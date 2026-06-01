from functools import reduce
from typing import Any, Callable, List, Optional, Tuple, Union
import math
import threading

import diffrax
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import equinox as eqx

from qml_essentials.script import Script  # noqa: F401
from qml_essentials.operations import (
    Hermitian,
    ParametrizedHamiltonian,
    Operation,
    PauliZ,
)

import logging

log = logging.getLogger(__name__)


class Jaqsi:
    # Module-level cache for JIT-compiled ODE solvers.  Keyed on
    # (coeff_fn_id, dim, atol, rtol, max_steps, throw) so that all
    # evolve() calls with the same pulse shape function and matrix size
    # share one compiled XLA program.  This turns O(n_gates) JIT
    # compilations into O(n_distinct_pulse_shapes) during pulse-mode
    # circuit building.
    _evolve_solver_cache: dict = {}
    _evolve_solver_cache_lock = threading.Lock()

    # Default solver knobs for parametrized (time-dependent) evolution.
    # These can be overridden per-call via the **odeint_kwargs of
    # ``evolve()`` or globally via :meth:`set_solver_defaults`.
    #
    # ``max_steps`` is the hard cap on accepted ODE steps.  Pulse-level
    # workloads at on-resonance carriers (ω_c ≈ ω_q) require many more
    # steps than the diffrax default during JIT — 2**13 = 8192 is
    # large enough for realistic single- and two-qubit pulses while
    # remaining cheap to compile.
    #
    # ``throw`` controls whether diffrax raises on solver failure
    # (e.g. ``MaxStepsReached``).  When set to ``False`` the gate
    # factory instead returns a NaN-filled unitary so the calling
    # optimiser sees a well-defined (but useless) result and can
    # gracefully reject the candidate.

    # ``solver`` selects the time-integration backend for the
    # interaction-picture ODE ``dU/dt = -i H_I(t) U``:
    #
    #   * ``"dopri8"`` (default) — adaptive Dormand-Prince 8(7) via
    #     diffrax.  Robust but expensive on highly oscillatory drives
    #     because the step controller resolves every fast cycle.
    #   * ``"dopri5"`` — TODO description
    #   * ``"magnus2"`` — commutator-free Magnus, 2nd order (midpoint
    #     rule) on a fixed ``magnus_steps`` grid via ``jax.lax.scan``.
    #     One ``expm`` per step.  Preserves unitarity to machine
    #     precision and fuses into a single XLA program.
    #   * ``"magnus4"`` — commutator-free Magnus, 4th order (CFM4:2 of
    #     Blanes & Moan) on a fixed ``magnus_steps`` grid.  Two ``H``
    #     evaluations and two ``expm`` per step; typically the best
    #     accuracy/cost trade-off for smooth oscillatory pulse drives.
    #
    # ``magnus_steps`` is the number of fixed substeps for the Magnus
    # integrators (ignored for ``dopri8``).  Choose it so that ``h =
    # T/N`` resolves the fastest oscillation in ``H(t)`` (~few steps
    # per period of the highest frequency).
    _solver_defaults: dict = {
        "max_steps": 2**13,
        "throw": True,
        "solver": "dopri8",
        "magnus_steps": 256,
    }
    _valid_solvers = ("dopri8", "dopri5", "magnus2", "magnus4")

    @classmethod
    def set_solver_defaults(
        cls,
        max_steps: Optional[int] = None,
        throw: Optional[bool] = None,
        solver: Optional[str] = None,
        magnus_steps: Optional[int] = None,
    ) -> dict:
        """Update class-level solver defaults; return the previous values.

        The returned dictionary is suitable for restoring the previous
        defaults via ``set_solver_defaults(**prev)``.

        Args:
            max_steps: New default for ``max_steps`` (ignored if ``None``).
            throw: New default for ``throw`` (ignored if ``None``).

        Returns:
            Dictionary with the previous values of the updated keys.
        """
        prev: dict = {}
        if max_steps is not None:
            prev["max_steps"] = cls._solver_defaults["max_steps"]
            cls._solver_defaults["max_steps"] = int(max_steps)
        if throw is not None:
            prev["throw"] = cls._solver_defaults["throw"]
            cls._solver_defaults["throw"] = bool(throw)
        if solver is not None:
            if solver not in cls._valid_solvers:
                raise ValueError(
                    f"Unknown solver {solver!r}; expected one of {cls._valid_solvers}"
                )
            prev["solver"] = cls._solver_defaults["solver"]
            cls._solver_defaults["solver"] = solver
        if magnus_steps is not None:
            prev["magnus_steps"] = cls._solver_defaults["magnus_steps"]
            cls._solver_defaults["magnus_steps"] = int(magnus_steps)
        return prev

    @classmethod
    def _store_evolve_solver(cls, cache_key: tuple, solve: Callable) -> Callable:
        """Cache a compiled evolve solver unless another thread won the race."""
        with cls._evolve_solver_cache_lock:
            existing = cls._evolve_solver_cache.get(cache_key)
            if existing is not None:
                return existing
            cls._evolve_solver_cache[cache_key] = solve
        return solve

    @classmethod
    def clear_evolve_solver_cache(cls) -> None:
        """Drop every cached compiled evolve solver.

        Call this whenever the coefficient functions referenced by the
        cache keys are rebuilt (e.g. when :class:`PulseGates` swaps in
        a new pulse envelope, RWA flag or frame).  Without an explicit
        eviction the cache keeps the old code objects alive and would
        also retain XLA programs that no longer match any active
        coefficient function.
        """
        with cls._evolve_solver_cache_lock:
            cls._evolve_solver_cache.clear()

    @classmethod
    def _parse_evolve_solver_options(cls, odeint_kwargs: dict) -> tuple:
        """Pop and validate solver options from ``evolve(..., **odeint_kwargs)``."""
        default_tol = 1.0e-10 if jax.config.x64_enabled else 1.4e-8
        atol = odeint_kwargs.pop("atol", default_tol)
        rtol = odeint_kwargs.pop("rtol", default_tol)
        max_steps = int(
            odeint_kwargs.pop("max_steps", cls._solver_defaults["max_steps"])
        )
        throw = bool(odeint_kwargs.pop("throw", cls._solver_defaults["throw"]))
        solver_name = str(odeint_kwargs.pop("solver", cls._solver_defaults["solver"]))
        if solver_name not in cls._valid_solvers:
            raise ValueError(
                f"Unknown solver {solver_name!r}; expected one of {cls._valid_solvers}"
            )
        magnus_steps = int(
            odeint_kwargs.pop("magnus_steps", cls._solver_defaults["magnus_steps"])
        )
        return atol, rtol, max_steps, throw, solver_name, magnus_steps

    @classmethod
    def _build_magnus_evolve_solver(
        cls,
        cache_key: tuple,
        coeff_fns: Tuple[Callable, ...],
        n_terms: int,
        dim: int,
        solver_name: str,
        magnus_steps: int,
    ) -> Callable:
        """Build and cache a fixed-step commutator-free Magnus solver."""
        _coeff_fns = coeff_fns
        _cdtype_local = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
        n_steps = magnus_steps
        solver_name_local = solver_name

        @eqx.filter_jit
        def _solve(neg_iH_split, params, t0, t1):
            # Reconstruct the per-term complex matrices ``-i H_i`` from their
            # split (Re, Im) representation so the coefficient sum is a single
            # complex tensordot.
            A_all = neg_iH_split[:, 0]
            B_all = neg_iH_split[:, 1]
            neg_iH = (A_all + 1j * B_all).astype(_cdtype_local)

            h = (t1 - t0) / n_steps

            def H_at(t):
                c = jnp.stack(
                    [
                        jnp.asarray(_coeff_fns[i](params[i], t)).reshape(())
                        for i in range(n_terms)
                    ]
                ).astype(_cdtype_local)
                return jnp.tensordot(c, neg_iH, axes=1)

            if solver_name_local == "magnus2":

                def step(U, n):
                    tn = t0 + n * h
                    Omega = h * H_at(tn + 0.5 * h)
                    return jax.scipy.linalg.expm(Omega) @ U, None

            else:
                sqrt3 = math.sqrt(3.0)
                c1 = 0.5 - sqrt3 / 6.0
                c2 = 0.5 + sqrt3 / 6.0
                a1 = 0.25 + sqrt3 / 6.0
                a2 = 0.25 - sqrt3 / 6.0

                def step(U, n):
                    tn = t0 + n * h
                    H1 = H_at(tn + c1 * h)
                    H2 = H_at(tn + c2 * h)
                    Omega_a = h * (a1 * H1 + a2 * H2)
                    Omega_b = h * (a2 * H1 + a1 * H2)
                    # CFM4:2 ordering (Blanes & Moan 2006, Table II):
                    # U_{n+1} = exp(Ω_b) · exp(Ω_a) · U_n.
                    U_next = (
                        jax.scipy.linalg.expm(Omega_b)
                        @ jax.scipy.linalg.expm(Omega_a)
                        @ U
                    )
                    return U_next, None

            U0 = jnp.eye(dim, dtype=_cdtype_local)
            U_final, _ = jax.lax.scan(step, U0, jnp.arange(n_steps))
            return U_final

        return cls._store_evolve_solver(cache_key, _solve)

    @classmethod
    def _build_diffrax_evolve_solver(
        cls,
        cache_key: tuple,
        coeff_fns: Tuple[Callable, ...],
        n_terms: int,
        dim: int,
        atol: float,
        rtol: float,
        max_steps: int,
        throw: bool,
        solver_name: str,
        _rdtype,
    ) -> Callable:
        """Build and cache an adaptive diffrax-based evolve solver."""
        solver = diffrax.Dopri8() if solver_name == "dopri8" else diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(atol=atol, rtol=rtol)
        _coeff_fns = coeff_fns

        @eqx.filter_jit
        def _solve(neg_iH_split, params, t0, t1):
            """Solve dU/dt = sum_i f_i(p_i, t) * (-iH_i) * U from t0 to t1.

            ``neg_iH_split`` has shape ``(n_terms, 2, dim, dim)`` with
            ``[:, 0]`` = Re(-iH_i) and ``[:, 1]`` = Im(-iH_i).
            ``params`` is a list/tuple of length ``n_terms`` carrying
            each term's coefficient parameters.  The state ``y`` has
            shape ``(2, dim, dim)`` with ``y[0] = Re(U)`` and
            ``y[1] = Im(U)``.
            """
            A_all = neg_iH_split[:, 0]
            B_all = neg_iH_split[:, 1]

            def rhs(t, y, args):
                # Each coefficient function must return a scalar value; some
                # call sites pass a shape-(1,) param array, so coerce to a
                # true scalar before stacking.
                c = jnp.stack(
                    [
                        jnp.asarray(_coeff_fns[i](args[i], t)).reshape(())
                        for i in range(n_terms)
                    ]
                )
                u_re = y[0]
                u_im = y[1]
                A_eff = jnp.tensordot(c, A_all, axes=1)
                B_eff = jnp.tensordot(c, B_all, axes=1)
                du_re = A_eff @ u_re - B_eff @ u_im
                du_im = A_eff @ u_im + B_eff @ u_re
                return jnp.stack([du_re, du_im], axis=0)

            y0 = jnp.stack(
                [
                    jnp.eye(dim, dtype=_rdtype),
                    jnp.zeros((dim, dim), dtype=_rdtype),
                ],
                axis=0,
            )

            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(rhs),
                solver,
                t0=t0,
                t1=t1,
                dt0=None,
                y0=y0,
                args=params,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
                throw=throw,
            )

            y_final = sol.ys[0]
            U = y_final[0] + 1j * y_final[1]

            if not throw:
                successful = sol.result == diffrax.RESULTS.successful
                U = jnp.where(successful, U, jnp.full_like(U, jnp.nan))
            return U

        return cls._store_evolve_solver(cache_key, _solve)

    @staticmethod
    def _partial_trace_single(
        rho: jnp.ndarray,
        n_qubits: int,
        keep: List[int],
    ) -> jnp.ndarray:
        """Partial trace of a single density matrix (no batch dimension)."""
        shape = (2,) * (2 * n_qubits)
        rho_t = rho.reshape(shape)

        trace_out = sorted(set(range(n_qubits)) - set(keep))

        for q in reversed(trace_out):
            n_remaining = rho_t.ndim // 2
            rho_t = jnp.trace(rho_t, axis1=q, axis2=q + n_remaining)

        dim = 2 ** len(keep)
        return rho_t.reshape(dim, dim)

    @classmethod
    def partial_trace(
        cls,
        rho: jnp.ndarray,
        n_qubits: int,
        keep: List[int],
    ) -> jnp.ndarray:
        """Partial trace of a density matrix, keeping only the specified qubits.

        Supports both single density matrices of shape ``(2**n, 2**n)`` and
        batched density matrices of shape ``(B, 2**n, 2**n)``.

        Args:
            rho: Density matrix of shape ``(2**n, 2**n)`` or ``(B, 2**n, 2**n)``.
            n_qubits: Total number of qubits.
            keep: List of qubit indices to *keep* (0-indexed).

        Returns:
            Reduced density matrix of shape ``(2**k, 2**k)`` or ``(B, 2**k, 2**k)``
            where *k* = ``len(keep)``.
        """

        dim = 2**n_qubits
        if rho.shape == (dim, dim):
            return cls._partial_trace_single(rho, n_qubits, keep)
        # Batched: shape (B, dim, dim)
        return jax.vmap(lambda r: cls._partial_trace_single(r, n_qubits, keep))(rho)

    @staticmethod
    def _marginalize_probs_single(
        probs: jnp.ndarray,
        target_shape: Tuple[int],
        trace_out: Tuple[int],
    ) -> jnp.ndarray:
        """Marginalize a single probability vector (no batch dimension)."""
        probs_t = probs.reshape(target_shape)

        for q in trace_out:
            probs_t = probs_t.sum(axis=q)

        return probs_t.ravel()

    @classmethod
    def marginalize_probs(
        cls,
        probs: jnp.ndarray,
        n_qubits: int,
        keep: Tuple[int],
    ) -> jnp.ndarray:
        """Marginalize a probability vector to keep only the specified qubits.

        Supports both single probability vectors of shape ``(2**n,)`` and
        batched vectors of shape ``(B, 2**n)``.

        Args:
            probs: Probability vector of shape ``(2**n,)`` or ``(B, 2**n)``.
            n_qubits: Total number of qubits.
            keep: List of qubit indices to *keep* (0-indexed).

        Returns:
            Marginalized probability vector of shape ``(2**k,)`` or ``(B, 2**k)``
            where *k* = ``len(keep)``.
        """

        dim = 2**n_qubits
        trace_out = tuple(q for q in range(n_qubits - 1, -1, -1) if q not in keep)
        target_shape = (2,) * n_qubits

        return jax.vmap(
            lambda p: cls._marginalize_probs_single(p, target_shape, trace_out)
        )(probs.reshape(-1, dim))

    @classmethod
    def build_parity_observable(
        cls,
        qubit_group: List[int],
    ) -> Hermitian:
        """Build a multi-qubit parity observable.

        Args:
            qubit_group: List of qubit indices for the parity measurement.

        Returns:
            A :class:`Hermitian` operation whose matrix is the Z parity
            tensor product and whose wires match the given qubits.
        """
        Z = PauliZ._matrix
        mat = reduce(jnp.kron, [Z] * len(qubit_group))
        return Hermitian(matrix=mat, wires=qubit_group, record=False)

    @classmethod
    def evolve(
        cls,
        hamiltonian: Union["Hermitian", "ParametrizedHamiltonian"],
        name: Optional[str] = None,
        **odeint_kwargs: Any,
    ) -> Callable:
        """Return a gate-factory for Hamiltonian time evolution.

        Supports two modes:

        Static — when *hamiltonian* is a :class:`Hermitian`::

            gate = evolve(Hermitian(H_mat, wires=0))
            gate(t=0.5)            # U = exp(-i*0.5*H)

        Time-dependent — when *hamiltonian* is a
        :class:`ParametrizedHamiltonian` (created via ``coeff_fn * Hermitian``)::

            H_td = coeff_fn * Hermitian(H_mat, wires=0)
            gate = evolve(H_td)
            gate([A, sigma], T)    # U via ODE: dU/dt = -i f(p,t) H * U

        The time-dependent case solves the Schrödinger equation numerically
        using ``diffrax.diffeqsolve`` with a Dopri8 adaptive Runge-Kutta
        solver

        All computations are pure JAX and fully differentiable with
        ``jax.grad``.

        Args:
            hamiltonian: Either a :class:`Hermitian` (static evolution) or a
                :class:`ParametrizedHamiltonian` (time-dependent evolution).
            **odeint_kwargs: Extra keyword arguments.  Recognised keys:

                - ``atol``, ``rtol`` — absolute/relative tolerances for the
                adaptive step-size controller (default ``1.4e-8``).

        Returns:
            A callable gate factory.  Signature depends on the mode:

            - Static: ``(t, wires=0) -> Operation``
            - Time-dependent: ``(coeff_args, T) -> Operation``

        Raises:
            TypeError: If *hamiltonian* is neither ``Hermitian`` nor
                ``ParametrizedHamiltonian``.
        """
        if isinstance(hamiltonian, Hermitian):
            return cls._evolve_static(hamiltonian, name=name)
        elif isinstance(hamiltonian, ParametrizedHamiltonian):
            return cls._evolve_parametrized(hamiltonian, name=name, **odeint_kwargs)
        else:
            raise TypeError(
                f"evolve() expects a Hermitian or ParametrizedHamiltonian, "
                f"got {type(hamiltonian)}"
            )

    @staticmethod
    def _evolve_static(hermitian: Hermitian, name: Optional[str] = None) -> Callable:
        """Gate factory for static Hamiltonian evolution U = exp(-i t H)."""
        H_mat = hermitian.matrix

        def _apply(t: float, wires: Union[int, List[int]] = 0) -> Operation:
            U = jax.scipy.linalg.expm(-1j * t * H_mat)
            return Operation(wires=wires, matrix=U, name=name)

        return _apply

    @classmethod
    def _evolve_parametrized(
        cls,
        ph: ParametrizedHamiltonian,
        name: Optional[str] = None,
        **odeint_kwargs: Any,
    ) -> Callable:
        """Gate factory for time-dependent (multi-term) Hamiltonian evolution.

        Solves the matrix ODE

            dU/dt = -i [\\sum_i f_i(params_i, t) * H_i] * U,    U(0) = I

        with ``diffrax.diffeqsolve`` (Dopri8 adaptive RK).  The Hamiltonian
        may contain one or more ``coeff_fn * Hermitian`` terms (see
        :class:`ParametrizedHamiltonian`); the single-term case is the
        usual ``coeff_fn * Hermitian`` and is fully backward compatible.

        Implementation notes:

        - To avoid diffrax's experimental complex dtype path, the ODE is
          reformulated in real arithmetic.  Writing ``-iH_i = A_i + i B_i``
          and ``U = U_re + i U_im``, each term contributes::

              d(U_re)/dt += f_i(p_i,t) * (A_i @ U_re - B_i @ U_im)
              d(U_im)/dt += f_i(p_i,t) * (A_i @ U_im + B_i @ U_re)

        - ``-iH_i`` is precomputed once per term and stacked into a
          ``(n_terms, 2, dim, dim)`` real array, contracted via
          ``einsum`` against the per-step coefficient vector
          ``c = [f_0(p_0,t), ..., f_{n-1}(p_{n-1},t)]``.

        - The JIT-compiled solver is cached per coefficient-function code
          tuple (and ``dim``, tolerances) so multiple ``evolve()`` calls
          with the same pulse shape — but different Hamiltonian matrices
          or parameters — reuse the same compiled XLA program.

        TODO: switch back once diffrax is stable with complex arithmetic.

        Args:
            ph: A :class:`ParametrizedHamiltonian` (one or more terms).
            **odeint_kwargs: Keyword arguments forwarded to
                ``diffrax.diffeqsolve``.  Recognised keys:

                - ``atol``, ``rtol`` — absolute/relative tolerances for the
                  step-size controller (default ``1.4e-8`` in fp32 mode,
                  ``1.0e-10`` in fp64 mode).
                - ``max_steps`` — hard cap on accepted ODE steps
                  (default :attr:`cls._solver_defaults['max_steps']`,
                  currently ``2**14``).  Increase this if the integrator
                  raises ``MaxStepsReached`` for a stiff/oscillatory
                  pulse Hamiltonian.
                - ``throw`` — whether to raise on solver failure
                  (default :attr:`cls._solver_defaults['throw']`,
                  currently ``True``).  When ``False``, a failed
                  integration returns a NaN-filled unitary instead of
                  raising; this is the recommended setting for inner
                  loops of an optimiser (e.g. QOC Stage 0) so a single
                  pathological candidate cannot abort the whole run.
        """
        coeff_fns = ph.coeff_fns  # tuple of callables
        H_mats = ph.H_mats  # tuple of (dim, dim)
        wires = ph.wires
        n_terms = ph.n_terms
        dim = H_mats[0].shape[0]

        # Pre-compute -i*H_i for each term and split into real / imaginary
        # parts so the ODE RHS uses only real arithmetic.  Final shape:
        # (n_terms, 2, dim, dim).
        neg_iH_split_per_term = []
        for H_mat in H_mats:
            neg_iH = -1j * H_mat
            neg_iH_split_per_term.append(
                jnp.stack([jnp.real(neg_iH), jnp.imag(neg_iH)], axis=0)
            )
        neg_iH_split = jnp.stack(neg_iH_split_per_term, axis=0)

        # Real dtype matching the precision mode
        # consider decreasing if no convergence
        _rdtype = jnp.float64 if jax.config.x64_enabled else jnp.float32

        # Pick tolerances according to precision + some headroom
        atol, rtol, max_steps, throw, solver_name, magnus_steps = (
            cls._parse_evolve_solver_options(odeint_kwargs)
        )

        # Cache key:  every coeff fn's code object (same shape of pulse
        # fns -> same JIT program) plus dim, tolerances, and solver
        # budget / throw flag (different budgets mean different XLA
        # programs).  We use the code object itself (hashable, identity-
        # equal) rather than ``id(fn.__code__)``: ids can be reused for
        # later code objects after the original is garbage-collected,
        # which would silently return a stale compiled solver for a
        # different pulse shape.  Holding the code object in the cache
        # keeps it alive for as long as the cached program is valid.
        cache_key = (
            tuple(fn.__code__ for fn in coeff_fns),
            dim,
            atol,
            rtol,
            max_steps,
            throw,
            solver_name,
            magnus_steps,
        )

        with cls._evolve_solver_cache_lock:
            _solve = cls._evolve_solver_cache.get(cache_key)
        if _solve is None:
            if solver_name in ("magnus2", "magnus4"):
                _solve = cls._build_magnus_evolve_solver(
                    cache_key=cache_key,
                    coeff_fns=coeff_fns,
                    n_terms=n_terms,
                    dim=dim,
                    solver_name=solver_name,
                    magnus_steps=magnus_steps,
                )
            else:
                _solve = cls._build_diffrax_evolve_solver(
                    cache_key=cache_key,
                    coeff_fns=coeff_fns,
                    n_terms=n_terms,
                    dim=dim,
                    atol=atol,
                    rtol=rtol,
                    max_steps=max_steps,
                    throw=throw,
                    solver_name=solver_name,
                    _rdtype=_rdtype,
                )

        def _apply(coeff_args, T) -> Operation:
            """Evolve under the (multi-term) time-dependent Hamiltonian.

            Args:
                coeff_args: List/tuple of parameter sets, one per term.
                    For single-term Hamiltonians the legacy form
                    ``[params]`` works unchanged; ``params`` is forwarded
                    to the sole coefficient function.
                T: Total evolution time.  Scalar -> integrate on
                    ``[0, T]``; 2-element -> integrate on ``[T[0], T[1]]``.

            Returns:
                An :class:`Operation` wrapping the computed unitary.
            """
            # Normalise to a tuple of length n_terms.  Accept a bare
            # single-term arg for backward compat.
            if isinstance(coeff_args, (list, tuple)):
                params = tuple(coeff_args)
            else:
                params = (coeff_args,)

            if len(params) != n_terms:
                raise ValueError(
                    f"Expected {n_terms} parameter set(s) for a "
                    f"{n_terms}-term ParametrizedHamiltonian, "
                    f"got {len(params)}."
                )

            # Build time span — resolve at Python level to avoid traced
            # branching.  ``T`` is either a Python scalar / 0-d array (=> integrate
            # on [0, T]) or a 2-element sequence/array (=> integrate on [T[0], T[1]]).
            # Let ``_solve`` cast t0/t1 to its working dtype; we only need the
            # array form to know the rank.
            T_arr = jnp.asarray(T, dtype=_rdtype)
            if T_arr.ndim == 0:
                t0 = _rdtype(0.0)
                t1 = T_arr
            else:
                t0 = T_arr[0]
                t1 = T_arr[1]

            U = _solve(neg_iH_split, params, t0, t1)

            return Operation(wires=wires, matrix=U, name=name)

        return _apply


# Convenience access to internal classmethods
partial_trace = Jaqsi.partial_trace
evolve = Jaqsi.evolve
marginalize_probs = Jaqsi.marginalize_probs
build_parity_observable = Jaqsi.build_parity_observable
