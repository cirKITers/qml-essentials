import argparse
import csv
import itertools
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
from jax import numpy as jnp
import numpy as np
import optax

from qml_essentials.gates import Gates, PulseInformation, PulseEnvelope
from qml_essentials import operations as op
from qml_essentials import yaqsi as ys
from qml_essentials.math import phase_difference, fidelity

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


def _build_optimizer(schedule, grad_clip: float):
    """Build the AdamW chain used by both stage-0 and stage-1.

    Adds a global-norm gradient-clip step when ``grad_clip`` is a
    finite, strictly positive value; otherwise returns plain AdamW.
    """
    use_clip = grad_clip and grad_clip > 0 and jnp.isfinite(grad_clip)
    if use_clip:
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adamw(schedule),
        )
    return optax.adamw(schedule)


def _safe_eval(cost_fn: Callable, params: jnp.ndarray) -> jnp.ndarray:
    """Evaluate ``cost_fn(params)``; map non-finite results to ``+inf``."""
    loss = cost_fn(params)
    return jnp.where(jnp.isfinite(loss), loss, jnp.inf)


def _with_basis_prep(circuit_fn: Callable, k: int, n_wires: int) -> Callable:
    """Wrap ``circuit_fn`` so it first prepares basis state ``|k⟩``.

    The wrapped circuit applies ``PauliX`` on every wire whose bit in
    ``k`` is set (MSB first) before delegating to ``circuit_fn``.  Used
    by both per-gate and joint optimisation paths to build the
    column-stacked unitary required by :func:`unitary_cost_fn`.
    """
    bits = [(k >> (n_wires - 1 - i)) & 1 for i in range(n_wires)]

    def prepared(*args, **kwargs):
        for i, bit in enumerate(bits):
            if bit:
                op.PauliX(wires=i)
        circuit_fn(*args, **kwargs)

    prepared.__name__ = f"basis{k}_{circuit_fn.__name__}"
    return prepared


def _sample_rotation_angles(n_samples: int) -> jnp.ndarray:
    """Boundary-biased sample of rotation angles in ``[0, 2π)``.

    The pulse-vs-target residual scales roughly linearly with rotation
    angle, so a uniform sample over ``[0, 2π)`` underweights the
    high-residual band that dominates failing tests (typical large-w
    test points: π/2, π).  We stratify the samples into

      * a uniform component covering the full ``[0, 2π)`` circle, and
      * a focus component packed in ``[π/2, 3π/2]``

    so the central band is sampled at roughly twice the density of the
    tails.  Returns at least one angle even for ``n_samples == 1``;
    when ``n_samples == 1`` the legacy uniform behaviour is preserved
    (single sample at ``w = 0``) to avoid surprising callers.
    """
    if n_samples <= 1:
        return jnp.linspace(0.0, 2.0 * jnp.pi, max(n_samples, 1), endpoint=False)
    # ~1/3 of samples in the central [π/2, 3π/2] band on top of a full
    # uniform sweep.  Sub-sample counts are rounded so both components
    # are non-empty for any ``n_samples >= 2``.
    k_focus = max(1, n_samples // 3)
    k_uniform = n_samples - k_focus
    ws_uniform = jnp.linspace(0.0, 2.0 * jnp.pi, k_uniform, endpoint=False)
    ws_focus = jnp.linspace(
        0.5 * jnp.pi, 1.5 * jnp.pi, k_focus, endpoint=False
    )
    return jnp.concatenate([ws_uniform, ws_focus])


class Cost:
    """Weighted wrapper around a cost function.

    Combines a cost callable with a scalar or tuple weight and optional
    constant keyword arguments.  Multiple ``Cost`` instances can be
    composed via the ``+`` operator to build a combined objective.

    Args:
        cost: Callable ``(pulse_params, **ckwargs) -> scalar | tuple``.
        weight: Scalar or tuple of per-component weights.
        ckwargs: Constant keyword arguments injected into every call.
    """

    def __init__(
        self,
        cost: Callable,
        weight: Union[float, Tuple],
        ckwargs: Optional[dict] = None,
    ):
        self.cost = cost
        self.weight = weight
        self.ckwargs = ckwargs if ckwargs is not None else {}

    def __call__(self, *args, **kwargs):
        """Evaluate the cost function with injected kwargs and apply weights."""
        cost = self.cost(*args, **kwargs, **self.ckwargs)
        if isinstance(self.weight, tuple):
            return jnp.array(
                [c * w for c, w in zip(cost, self.weight, strict=True)]
            ).sum()
        return cost * self.weight

    def __add__(self, other):
        """Compose two cost terms into a single callable that sums them."""
        if other is None:
            return lambda *args, **kwargs: self(*args, **kwargs)
        if callable(other):
            return lambda *args, **kwargs: self(*args, **kwargs) + other(
                *args, **kwargs
            )
        raise TypeError(f"Cannot add Cost and {type(other)}")


def fidelity_cost_fn(
    pulse_params: jnp.ndarray,
    pulse_scripts: Union[ys.Script, List[ys.Script]],
    target_scripts: Union[ys.Script, List[ys.Script]],
    n_samples: int,
) -> Tuple[float, float]:
    """
    Cost function returning ``(1 - fidelity, 1 - cos(phase_difference))``
    averaged over ``n_samples`` uniformly spaced rotation angles in
    ``[0, 2π)`` and across one or more (pulse, target) script pairs.

    Multiple script pairs let the optimiser probe sensitivity from
    multiple initial states (e.g. ``|0⟩`` and ``|+⟩``).  This makes
    rotation-axis tilt observable to the cost: from ``|0⟩`` alone an
    RX/RY pulse with a small Z-component is largely degenerate with
    the correct pulse, but from ``|+⟩`` the same tilt produces a
    visible state-vector deviation.

    Uses batched (vmapped) circuit execution per script: all
    ``n_samples`` rotation angles are evaluated in a single vectorised
    call per script, replacing ``n_samples`` sequential Python-level
    circuit executions with one JIT-compiled XLA program each.

    The phase term uses ``1 - cos(Δφ)`` rather than ``|Δφ|`` so that
    it is differentiable everywhere (including at the optimum) and
    well-behaved at the ``±π`` wrap-around — important because Stage 0
    now sees the same cost as Stage 1.

    Args:
        pulse_params: Pulse parameters for evaluation.
        pulse_scripts: One or a list of yaqsi scripts with pulse
            parameters.  If a list is supplied, the cost is averaged
            element-wise with ``target_scripts`` (which must have the
            same length).
        target_scripts: One or a list of yaqsi target scripts.
        n_samples: Number of parameter samples.

    Returns:
        Tuple of ``(abs_diff, phase_diff)`` averaged across script pairs.
    """
    if not isinstance(pulse_scripts, (list, tuple)):
        pulse_scripts = [pulse_scripts]
    if not isinstance(target_scripts, (list, tuple)):
        target_scripts = [target_scripts]
    assert len(pulse_scripts) == len(target_scripts), (
        f"pulse_scripts and target_scripts must have the same length "
        f"({len(pulse_scripts)} vs {len(target_scripts)})."
    )

    ws = _sample_rotation_angles(n_samples)

    abs_diffs = []
    phase_diffs = []
    for p_script, t_script in zip(pulse_scripts, target_scripts):
        pulse_states = p_script.execute(
            type="state",
            args=(ws, pulse_params),
            in_axes=(0, None),
        )  # (n_samples, dim)

        target_states = t_script.execute(
            type="state",
            args=(ws,),
            in_axes=(0,),
        )  # (n_samples, dim)

        abs_diffs.append(
            jnp.mean(
                jnp.array(1.0, dtype=jnp.float64)
                - fidelity(pulse_states, target_states)
            )
        )
        phase_diffs.append(
            jnp.mean(
                jnp.array(1.0, dtype=jnp.float64)
                - jnp.cos(phase_difference(pulse_states, target_states))
            )
        )

    abs_diff = jnp.mean(jnp.stack(abs_diffs))
    phase_diff = jnp.mean(jnp.stack(phase_diffs))

    # TODO: in future we could consider some sort of log based loss for the small values
    # or utilize gradient ascent if we run into numerical limitations

    return (abs_diff, phase_diff)


def unitary_cost_fn(
    pulse_params: jnp.ndarray,
    pulse_basis_scripts: List[ys.Script],
    target_basis_scripts: List[ys.Script],
    n_samples: int,
    n_qubits: int,
) -> Tuple[float, float]:
    """Unitary-level cost based on the average gate (process) fidelity.

    Builds the full unitary of the pulse and target circuits at every
    sampled rotation angle by stacking ``2**n_qubits`` basis-state
    evolutions as columns (``U[:, k] = circuit(|k⟩)``).  Returns

        (1 - |Tr(E)|² / d²,  1 - cos(angle(Tr(E))))

    where ``E = U_target† · U_pulse`` and ``d = 2**n_qubits``.

    The first component is the standard process-infidelity (which is
    *global-phase invariant*).  The second component captures the
    residual global phase between pulse and target — without it the
    optimiser cannot distinguish ``U_pulse`` and ``e^{iα} U_pulse``,
    which leaves systematic phase errors in composed gates (e.g. the
    H-CZ-H decomposition of CX).

    Compared to the state-vector ``fidelity_cost_fn``, this cost
    captures rotation-axis tilt and off-diagonal coherent error in a
    single number, regardless of which probe state(s) one chooses.

    Args:
        pulse_params: Pulse parameters under optimisation.
        pulse_basis_scripts: List of ``d`` scripts; the k-th script
            prepares ``|k⟩`` (via ``PauliX`` gates) and then applies
            the pulse-level circuit.
        target_basis_scripts: Same for the target circuit.
        n_samples: Number of rotation-angle samples in ``[0, 2π)``.
        n_qubits: Number of qubits the gate acts on.

    Returns:
        Tuple ``(process_loss, phase_loss)`` averaged over rotation
        angles.
    """
    d = 2**n_qubits
    assert len(pulse_basis_scripts) == d, (
        f"pulse_basis_scripts must have {d} entries (one per basis "
        f"state); got {len(pulse_basis_scripts)}."
    )
    assert len(target_basis_scripts) == d, (
        f"target_basis_scripts must have {d} entries (one per basis "
        f"state); got {len(target_basis_scripts)}."
    )

    ws = _sample_rotation_angles(n_samples)

    pulse_cols = []
    target_cols = []
    for k in range(d):
        ps = pulse_basis_scripts[k].execute(
            type="state",
            args=(ws, pulse_params),
            in_axes=(0, None),
        )  # (n_samples, d)
        ts = target_basis_scripts[k].execute(
            type="state",
            args=(ws,),
            in_axes=(0,),
        )  # (n_samples, d)
        pulse_cols.append(ps)
        target_cols.append(ts)

    # Stack basis-state outputs as columns of U at every sampled angle.
    # Resulting shape (n_samples, d, d) with U[s, :, k] = column k.
    U_pulse = jnp.stack(pulse_cols, axis=-1)
    U_target = jnp.stack(target_cols, axis=-1)

    # E = U_target^† U_pulse, shape (n_samples, d, d)
    E = jnp.einsum("sji,sjk->sik", jnp.conj(U_target), U_pulse)
    trE = jnp.einsum("sii->s", E)

    F_pro = jnp.abs(trE) ** 2 / float(d) ** 2
    process_loss = jnp.mean(jnp.array(1.0, dtype=jnp.float64) - F_pro)
    phase_loss = jnp.mean(
        jnp.array(1.0, dtype=jnp.float64) - jnp.cos(jnp.angle(trE))
    )

    return (process_loss, phase_loss)


def joint_unitary_cost_fn(
    pulse_params: jnp.ndarray,
    gate_specs: List[dict],
    n_samples: int,
) -> Tuple[float, float]:
    """Joint unitary-level cost summed over multiple target gates.

    Each entry in ``gate_specs`` is a dictionary describing one target
    gate that shares the joint parameter vector ``pulse_params``::

        {
            "name":                 str,             # gate name (debug)
            "n_qubits":             int,
            "weight":               float,           # per-gate weight
            "assembler":            Callable,        # theta -> per-gate flat params
            "pulse_basis_scripts":  List[ys.Script], # 2**n_qubits scripts
            "target_basis_scripts": List[ys.Script],
        }

    The total return value is a ``(process_loss, phase_loss)`` tuple
    where each component is ``Σ_g w_g · loss_g(theta)`` divided by the
    sum of weights.  Sharing the leaf parameters across all target
    gates pulls the optimum into a basin that is good for *every*
    use-site (composite gates as well as standalone leaves) — fixing
    the failure mode where per-gate optimisation pushes a leaf into a
    "selfish" basin that is optimal for its standalone use but breaks
    composites that contain it.

    Args:
        pulse_params: Joint leaf parameter vector (theta).
        gate_specs: List of per-gate spec dicts (see above).
        n_samples: Number of rotation-angle samples per gate.

    Returns:
        Tuple ``(process_loss, phase_loss)`` averaged over angles and
        weighted-summed over gates.
    """
    total_proc = jnp.array(0.0, dtype=jnp.float64)
    total_phase = jnp.array(0.0, dtype=jnp.float64)
    total_w = 0.0

    for spec in gate_specs:
        per_gate_pp = spec["assembler"](pulse_params)
        proc_loss, phase_loss = unitary_cost_fn(
            per_gate_pp,
            spec["pulse_basis_scripts"],
            spec["target_basis_scripts"],
            n_samples,
            spec["n_qubits"],
        )
        w = spec["weight"]
        total_proc = total_proc + w * proc_loss
        total_phase = total_phase + w * phase_loss
        total_w += w

    if total_w > 0:
        total_proc = total_proc / total_w
        total_phase = total_phase / total_w

    return (total_proc, total_phase)


def pulse_width_cost_fn(
    pulse_params: jnp.ndarray,
    envelope: str,
) -> jnp.ndarray:
    """
    Cost function penalising the pulse width (sigma / width).

    The pulse width is taken as the last envelope parameter. For
    envelopes with no envelope parameters (e.g. ``"general"``), the cost
    is zero.

    Args:
        pulse_params: Pulse parameters for the gate.
        envelope: Name of the active pulse envelope.

    Returns:
        Scalar pulse-width cost.
    """
    envelope_info = PulseEnvelope.get(envelope)
    n_envelope_params = envelope_info["n_envelope_params"]

    if n_envelope_params > 0:
        pulse_width = pulse_params[n_envelope_params - 1]
    else:
        pulse_width = 0

    return jnp.array(pulse_width, dtype=jnp.float64)


def evolution_time_cost_fn(
    pulse_params: jnp.ndarray,
    t_target: float,
) -> jnp.ndarray:
    """
    Cost function penalising deviation of the evolution time from a target.

    The evolution time is always the last element of the pulse parameter
    vector.  The cost is the squared relative deviation from ``t_target``:

        cost = ((t - t_target) / t_target) ** 2

    This encourages all independently optimized gates to converge towards a
    common evolution time, making them compatible when composed into a
    circuit.

    Args:
        pulse_params: Pulse parameters for the gate.
        t_target: Target evolution time.

    Returns:
        Scalar evolution-time cost.
    """
    t = pulse_params[-1]
    return ((t - t_target) / t_target) ** 2


def spectral_density_cost_fn(
    pulse_params: jnp.ndarray,
    envelope: str,
    n_fft: int = 1024,
) -> jnp.ndarray:
    """
    Cost function penalising the spectral width of a given pulse.

    Samples the pulse envelope in the time domain over ``[0, t_evol]``
    (where ``t_evol`` is the last element of pulse_params), computes its
    power spectral density via FFT, and returns the normalised RMS bandwidth
    (square root of the second central moment of the PSD).

    Pulses with narrow spectra (e.g. Gaussian, DRAG) receive a low cost,
    whereas pulses with wide spectra (e.g. rectangular) are penalised more
    heavily.

    For envelopes with no envelope parameters (e.g. ``"general"``), the
    cost is zero.

    Args:
        pulse_params: Pulse parameters for the gate.  Envelope parameters
            occupy ``pulse_params[:n_envelope_params]`` and the evolution
            time is ``pulse_params[-1]``.
        envelope: Name of the active pulse envelope.
        n_fft: Number of time-domain samples used for the FFT
            (default 1024).

    Returns:
        Scalar spectral-width cost (RMS bandwidth normalised by the
        Nyquist frequency so the value is in [0, 1]).
    """
    envelope_info = PulseEnvelope.get(envelope)
    n_envelope_params = envelope_info["n_envelope_params"]
    envelope_fn = envelope_info["fn"]

    # Nothing to penalise for envelopes without tuneable shape params
    if n_envelope_params == 0 or envelope_fn is None:
        return jnp.array(0.0, dtype=jnp.float64)

    # Extract envelope parameters and evolution time
    env_params = pulse_params[:n_envelope_params]
    t_evol = pulse_params[-1]
    t_c = t_evol / 2.0

    t_samples = jnp.linspace(0.0, t_evol, n_fft)
    signal = jax.vmap(lambda t: envelope_fn(env_params, t, t_c))(t_samples)

    spectrum = jnp.fft.rfft(signal)
    psd = jnp.abs(spectrum) ** 2
    psd = psd / (jnp.sum(psd) + 1e-12)  # normalise to a distribution

    freqs = jnp.linspace(0.0, 1.0, len(psd))

    mean_freq = jnp.sum(freqs * psd)
    rms_bw = jnp.sqrt(jnp.sum((freqs - mean_freq) ** 2 * psd))

    return jnp.array(rms_bw, dtype=jnp.float64)


class CostFnRegistry:
    """Registry of cost functions available for pulse optimisation.

    Use :meth:`register` to add new cost functions at runtime and
    :meth:`get` / :meth:`available` to query them.
    """

    _REGISTRY: Dict[str, dict] = {
        "fidelity": {
            "fn": fidelity_cost_fn,
            "default_weight": (0.5, 0.5),
            "ckwargs_keys": ["pulse_scripts", "target_scripts", "n_samples"],
        },
        "unitary": {
            "fn": unitary_cost_fn,
            "default_weight": (0.5, 0.5),
            "ckwargs_keys": [
                "pulse_basis_scripts",
                "target_basis_scripts",
                "n_samples",
                "n_qubits",
            ],
        },
        "pulse_width": {
            "fn": pulse_width_cost_fn,
            "default_weight": 1.0,
            "ckwargs_keys": ["envelope"],
        },
        "evolution_time": {
            "fn": evolution_time_cost_fn,
            "default_weight": 1.0,
            "ckwargs_keys": ["t_target"],
        },
        "spectral_density": {
            "fn": spectral_density_cost_fn,
            "default_weight": 1.0,
            "ckwargs_keys": ["envelope"],
        },
    }

    @classmethod
    def available(cls) -> List[str]:
        """Return the names of all registered cost functions."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def get(cls, name: str) -> dict:
        """Look up cost-function metadata by name.

        Args:
            name: Registered cost function name.

        Returns:
            Metadata dict with keys ``fn``,
            ``default_weight``, ``ckwargs_keys``.

        Raises:
            ValueError: If name is not registered.
        """
        if name not in cls._REGISTRY:
            raise ValueError(
                f"Unknown cost function '{name}'. " f"Available: {cls.available()}"
            )
        return cls._REGISTRY[name]

    @classmethod
    def parse_cost_arg(
        cls, spec: Union[str, Tuple]
    ) -> Tuple[str, Union[float, Tuple[float, ...]]]:
        """Parse a ``"name:w1,w2,..."`` CLI string into ``(name, weight)``.
        If a tuple is provided, it is returned directly.

        If the weight part is omitted the default weight from the registry
        is used.  A single-component weight is returned as a float;
        multi-component weights are returned as a tuple of floats.

        Args:
            spec: A string of the form ``"name"`` or ``"name:w1,w2,..."``.

        Returns:
            A tuple of ``(name, weight)``.

        Raises:
            ValueError: If the name is unknown or the number of weight
                components does not match the ones in ``default_weight``.
        """
        if isinstance(spec, tuple):
            return spec

        if ":" in spec:
            name, weight_str = spec.split(":", 1)
            parts = [float(x) for x in weight_str.split(",")]
            weight: Union[float, Tuple[float, ...]] = (
                parts[0] if len(parts) == 1 else tuple(parts)
            )
        else:
            name = spec
            weight = cls.get(name)["default_weight"]

        # Validate weight count
        got = len(weight) if isinstance(weight, tuple) else 1
        default_weight = cls.get(name)["default_weight"]
        expected = len(default_weight) if isinstance(default_weight, tuple) else 1

        if got != expected:
            raise ValueError(
                f"Cost function '{name}' expects {expected} weight(s), " f"got {got}."
            )

        return name, weight


class QOC:
    """Quantum Optimal Control for pulse-level gate synthesis.

    Optimises pulse parameters to reproduce the unitary of standard
    quantum gates using a two-stage strategy.

    Attributes:
        GATES_1Q: Names of supported single-qubit gates.
        GATES_2Q: Names of supported two-qubit gates.
        DEFAULT_PARAM_RANGES: Default parameter ranges for each gate.
    """

    GATES_1Q: List[str] = ["RX", "RY", "RZ", "Rot", "H"]
    GATES_2Q: List[str] = ["CX", "CY", "CZ", "CRX", "CRY", "CRZ"]

    DEFAULT_PARAM_RANGES = {
        1: [(0.05, 3.0)],  # evolution time
        2: [(0.05, 3.0), (0.05, 3.0)],  # not typically used
        3: [(0.05, 3.0), (0.05, 3.0), (0.05, 3.0)],  # [A, sigma, t]
        4: [(0.05, 3.0), (0.05, 3.0), (0.05, 3.0), (0.05, 3.0)],  # [A, beta, sigma, t]
    }

    def __init__(
        self,
        envelope: str,
        cost_fns: List[Tuple[str, Union[float, Tuple[float, ...]]]],
        t_target: float,
        n_steps: int,
        n_samples: int,
        learning_rate: float,
        log_interval: int = 50,
        file_dir: str = None,
        warmup_ratio: float = 0.0,
        end_lr_ratio: float = 1.0,
        n_restarts: int = 1,
        restart_noise_scale: float = 0.5,
        grad_clip: float = 1.0,
        random_seed: int = 42,
        scan_steps: int = 0,
        scan_grid_size: int = 5,
        scan_ranges: Optional[List[Tuple[float, float]]] = None,
        log_scale_params: Optional[List[int]] = None,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
        plot: bool = False,
    ):
        """
        Initialize Quantum Optimal Control with Pulse-level Gates.

        Args:
            envelope (str): Pulse envelope shape to use for optimization.
                Must be one of the registered envelopes in PulseEnvelope
                (e.g. 'gaussian', 'square', 'cosine', 'drag', 'sech').
            cost_fns (list): List of ``(name, weight)`` tuples that select
                which cost functions to use and their weights.  name must
                be a key in :class:`CostFnRegistry`.  *weight* is either a
                single float or a tuple of floats matching the number of
                return values of the cost function.
            t_target (float, optional): Target evolution time for the
                ``evolution_time`` cost function.  Required when
                ``"evolution_time"`` is among the selected cost functions.
            n_steps (int): Number of steps in optimization.
            n_samples (int): Number of parameter samples per step.
            learning_rate (float): Peak learning rate for AdamW. When a
                warmup/decay schedule is active this is the maximum LR
                reached after the warmup phase.
            log_interval (int): Interval for logging.
            file_dir (str): Directory to save results.
            warmup_ratio (float): Fraction of ``n_steps`` used for linear
                warmup (0.0 - 1.0).  Set to 0.0 to disable warmup and use
                a constant learning rate throughout.  A value of e.g. 0.05
                means the first 5 % of steps linearly ramp the LR from
                ``end_lr_ratio * learning_rate`` to ``learning_rate``.
            end_lr_ratio (float): The final learning rate is
                ``end_lr_ratio * learning_rate``.  Also used as the initial
                LR at the start of warmup.  Set to 0.0 for full cosine
                decay to zero; set to 1.0 (together with
                ``warmup_ratio=0.0``) to recover a constant LR.
            n_restarts (int): Number of random restarts for the
                optimisation.  The first run uses the initial parameters
                as-is; subsequent runs add scaled random perturbations.
                The best result across all restarts is kept.
                Set to 1 to disable restarts (default behaviour).
            restart_noise_scale (float): Standard deviation of the
                Gaussian noise added to the initial parameters for each
                restart (relative to the absolute value of each parameter).
                Defaults to 0.5 (50 % relative perturbation).  Note that
                the package-level default in ``default_qoc_params`` is a
                much smaller ``0.01`` because the QOC loss landscape is
                highly sensitive to initial conditions and large
                perturbations routinely move restarts into useless
                basins; tune up only if you have reason to believe the
                initial point is far from any good basin.
            grad_clip (float): Maximum global gradient norm.  Gradients
                are clipped to this value before being passed to the
                optimiser, which stabilises training when the loss
                landscape has steep regions.  Set to ``float('inf')`` or
                0.0 to disable.  Defaults to 1.0.
            random_seed (int): Base random seed for generating restart
                perturbations.  Defaults to 42.
            scan_steps (int): Number of short gradient-descent steps to
                run for each candidate in the coarse grid search
                (Stage 0).  Set to 0 to disable the grid scan entirely
                and rely solely on restarts.  A value of 20-50 is
                usually enough to identify promising basins.  Defaults
                to 0.
            scan_grid_size (int): Number of points per parameter
                dimension in the coarse grid.  The total number of
                candidates is ``scan_grid_size ** n_params``, so keep
                this small for high-dimensional parameter spaces.
                Defaults to 5.
            scan_ranges (Optional[List[Tuple[float, float]]]): Per-
                parameter ``(lo, hi)`` ranges for the grid scan.  If
                ``None``, heuristic ranges are used based on the
                envelope type: amplitude in ``[0.5, 30]``, width/sigma
                in ``[0.05, 2]``, and evolution time in ``[0.05, 2]``.
                Must have length equal to the number of pulse parameters
                if provided.
            log_scale_params (Optional[List[int]]): Indices of pulse
                parameters that should be optimised in log-space.  For
                these parameters the optimizer sees ``log(p)`` and the
                actual parameter used in the simulation is ``exp(log_p)``.
                This dramatically improves convergence when the optimal
                value may differ from the initial value by an order of
                magnitude (e.g. amplitude, evolution time).
                If ``None``, defaults to ``[0, -1]`` (amplitude and
                evolution time) for envelopes with ≥ 2 envelope params,
                or ``[]`` otherwise.
            early_stop_patience (int): Number of consecutive
                Stage-1 steps with no improvement greater than
                ``early_stop_min_delta`` after which optimisation
                exits early.  Set to ``0`` (default) to disable.
                Only honoured in the single-restart (sequential)
                path; when ``n_restarts > 1`` the parallel
                vmap+scan path always runs the full ``n_steps``.
            early_stop_min_delta (float): Minimum decrease in loss
                that counts as an improvement for the early-stopping
                patience counter.  Defaults to ``0.0`` (any strict
                improvement resets the counter).
            plot (bool): If ``True``, save a loss-landscape figure after
                Phase 0 and a loss-curve figure after Phase 1 to
                ``file_dir``.  Requires ``matplotlib`` to be installed.
                Defaults to ``False``.
        """
        self.envelope = envelope
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.end_lr_ratio = end_lr_ratio
        self.log_interval = log_interval
        self.file_dir = (
            file_dir if file_dir else os.path.dirname(os.path.realpath(__file__))
        )
        self.t_target = t_target
        self.n_restarts = max(1, n_restarts)
        self.restart_noise_scale = restart_noise_scale
        self.grad_clip = grad_clip
        self.random_key = jax.random.PRNGKey(random_seed)
        self.scan_steps = scan_steps
        self.scan_grid_size = scan_grid_size
        self.scan_ranges = scan_ranges

        # Determine log-scale param indices
        envelope_info = PulseEnvelope.get(envelope)
        n_env = envelope_info["n_envelope_params"]
        if log_scale_params is not None:
            self.log_scale_params = log_scale_params
        elif n_env >= 2:
            # Default: amplitude (index 0) and evolution time (last)
            self.log_scale_params = [0, -1]
        else:
            self.log_scale_params = []

        # Mask cache used by ``_to_log_space``/``_from_log_space``;
        # rebuilt lazily because the mask length depends on the size of
        # the param vector being converted (per-gate vs joint).
        self._log_mask_cache: Dict[int, jnp.ndarray] = {}

        self.early_stop_patience = max(0, int(early_stop_patience))
        self.early_stop_min_delta = float(early_stop_min_delta)

        self.plot = plot

        log.info(
            f"Training parameters: {self.n_steps} steps, "
            f"{self.n_samples} samples, {self.learning_rate} learning rate"
        )
        log.info(
            f"LR schedule: warmup_ratio={self.warmup_ratio}, "
            f"end_lr_ratio={self.end_lr_ratio}"
        )

        log.info(f"Envelope: {self.envelope}")
        log.info(f"Target evolution time: {self.t_target}")
        log.info(
            f"Restarts: {self.n_restarts}, noise_scale={self.restart_noise_scale}, "
            f"grad_clip={self.grad_clip}"
        )
        if self.early_stop_patience > 0:
            log.info(
                f"Early stopping: patience={self.early_stop_patience}, "
                f"min_delta={self.early_stop_min_delta:g}"
            )
        log.info(
            f"Grid scan: scan_steps={self.scan_steps}, "
            f"scan_grid_size={self.scan_grid_size}, "
            f"log_scale_params={self.log_scale_params}"
        )
        log.info(f"Using cost function(s) {cost_fns}")

        # Validate each entry against the registry
        summed_weights = 0
        for name, _weight in cost_fns:
            CostFnRegistry.get(name)  # raises ValueError if unknown
            summed_weights += sum(_weight) if isinstance(_weight, tuple) else _weight
        assert jnp.isclose(
            summed_weights, 1.0, rtol=1e-8
        ), f"Cost function weights must sum to 1. Got {summed_weights}"

        self.cost_fns = cost_fns

        # Configure the pulse system with the selected envelope
        PulseInformation.set_envelope(self.envelope)

    def save_results(self, gate: str, fidelity: float, pulse_params) -> None:
        """Save optimised pulse parameters and fidelity for a gate to CSV.

        If the gate already exists in the file, its entry is overwritten
        regardless of whether the new fidelity is higher.  A warning is
        logged when the existing fidelity was better.

        Args:
            gate: Name of the gate (e.g. ``"RX"``).
            fidelity: Achieved fidelity of the optimised pulse.
            pulse_params (jnp.ndarray): Optimised pulse parameters for the gate.
        """
        if self.file_dir is not None:
            os.makedirs(self.file_dir, exist_ok=True)
            filename = os.path.join(self.file_dir, f"qoc_results_{self.envelope}.csv")

            reader = None
            if os.path.isfile(filename):
                with open(filename, mode="r", newline="") as f:
                    reader = csv.reader(f.readlines())

            entry = [gate] + [fidelity] + list(map(float, pulse_params))

            with open(filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                match = False
                if reader is not None:
                    for row in reader:
                        # gate already exists
                        if row[0] == gate:
                            if fidelity <= float(row[1]):
                                log.warning(
                                    f"Pulse parameters for {gate} already exist with "
                                    f"higher fidelity ({row[1]} >= {fidelity})"
                                )
                            writer.writerow(entry)
                            match = True
                        # any other gate
                        else:
                            writer.writerow(row)
                # gate does not exist
                if not match:
                    writer.writerow(entry)

    def _log_mask(self, n: int) -> jnp.ndarray:
        """Return a boolean mask of length ``n`` marking log-scaled indices."""
        cached = self._log_mask_cache.get(n)
        if cached is not None and cached.shape[0] == n:
            return cached
        mask = np.zeros(n, dtype=bool)
        for idx in self.log_scale_params:
            i = idx if idx >= 0 else n + idx
            if 0 <= i < n:
                mask[i] = True
        out = jnp.asarray(mask)
        self._log_mask_cache[n] = out
        return out

    def _to_log_space(self, params: jnp.ndarray) -> jnp.ndarray:
        """Convert selected parameters to log-space for optimisation.

        Parameters at indices in ``self.log_scale_params`` are replaced
        by ``log(|p| + eps)`` so the optimiser operates on a
        logarithmic scale.  All other parameters are left unchanged.
        """
        if not self.log_scale_params:
            return params
        mask = self._log_mask(params.shape[0])
        log_vals = jnp.log(jnp.abs(params) + 1e-12)
        return jnp.where(mask, log_vals, params)

    def _from_log_space(self, log_params: jnp.ndarray) -> jnp.ndarray:
        """Convert selected parameters back from log-space.

        Inverse of :meth:`_to_log_space`.  Parameters at indices in
        ``self.log_scale_params`` are exponentiated; all others are
        passed through unchanged.
        """
        if not self.log_scale_params:
            return log_params
        mask = self._log_mask(log_params.shape[0])
        return jnp.where(mask, jnp.exp(log_params), log_params)

    # Multiplicative factors used to build a centred grid around the
    # supplied init parameters when no explicit ``scan_ranges`` are
    # given.  ``1.0`` is included so the init point itself is always a
    # candidate (Stage 0 cannot otherwise re-evaluate it as a grid
    # point — only as the baseline ``best_scan_loss``).
    SCAN_REL_FACTORS: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)

    def _build_scan_grid(
        self,
        n_params: int,
        init_pulse_params: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """Build a coarse parameter grid for the initial scan phase.

        If the user supplied ``scan_ranges`` they take precedence and
        a log-spaced grid is built within those bounds.  Otherwise, when
        ``init_pulse_params`` is available, a **multiplicative grid
        centred on the init point** is used (each axis spans
        ``init * SCAN_REL_FACTORS``) so that already-optimised init
        params are always re-evaluated and only their immediate
        neighbourhood is explored.  This avoids the failure mode where
        the global ``DEFAULT_PARAM_RANGES`` brackets exclude the actual
        optimum (the previous default range was ``(0.05, 3.0)`` per
        axis, which clipped DRAG amplitudes around 3.1 and made the
        scan systematically worse than the init point).

        Args:
            n_params: Number of pulse parameters.
            init_pulse_params: Optional init params used to centre the
                multiplicative grid when ``scan_ranges`` is ``None``.

        Returns:
            Tuple of:
            - Array of shape ``(n_candidates, n_params)`` with grid points.
            - List of 1-D arrays, one per parameter axis.
        """
        if self.scan_ranges is not None:
            ranges = self.scan_ranges
            assert len(ranges) == n_params, (
                f"scan_ranges has {len(ranges)} entries but gate has "
                f"{n_params} parameters."
            )
            # Build log-spaced grids for each parameter
            axes = []
            for lo, hi in ranges:
                axes.append(
                    jnp.logspace(
                        jnp.log10(lo), jnp.log10(hi), self.scan_grid_size
                    )
                )
        elif init_pulse_params is not None:
            # Multiplicative grid centred on init params.  We pick
            # ``scan_grid_size`` factors symmetric around 1.0.  When
            # ``scan_grid_size`` matches the static SCAN_REL_FACTORS
            # length we use those; otherwise build a symmetric linspace.
            if self.scan_grid_size == len(self.SCAN_REL_FACTORS):
                factors = jnp.array(self.SCAN_REL_FACTORS, dtype=jnp.float64)
            else:
                half = (self.scan_grid_size - 1) / 2.0
                if half <= 0:
                    factors = jnp.array([1.0], dtype=jnp.float64)
                else:
                    factors = jnp.linspace(
                        1.0 - 0.5,
                        1.0 + 0.5,
                        self.scan_grid_size,
                        dtype=jnp.float64,
                    )
            axes = [factors * float(p) for p in init_pulse_params]
        else:
            # Fall back to legacy log-spaced default ranges
            ranges = self.DEFAULT_PARAM_RANGES.get(
                n_params,
                [(0.1, 10.0)] * n_params,
            )
            axes = []
            for lo, hi in ranges:
                axes.append(
                    jnp.logspace(
                        jnp.log10(lo), jnp.log10(hi), self.scan_grid_size
                    )
                )

        # Cartesian product of all axes
        grid = jnp.array(list(itertools.product(*axes)))
        return grid, axes

    def stage_0_opt(
        self, init_pulse_params: jnp.ndarray, total_cost: Callable
    ) -> Tuple[jnp.ndarray, Optional[Tuple[List[jnp.ndarray], list]]]:
        """Run the coarse grid-scan phase (Stage 0).

        Evaluates a Cartesian grid of parameter candidates using the
        **full weighted cost** (fidelity + phase, plus any other
        registered terms) — the same objective as Stage 1.  Each
        candidate is refined with a few fast gradient steps.  Returns
        the best-found parameters.

        Sharing the objective with Stage 1 prevents the grid scan from
        landing in a basin that has high fidelity but a biased phase
        which Adam then has to migrate out of (the previous
        fidelity-only scan caused exactly this failure mode for RX/RY,
        whose phase residuals compounded in the CRX decomposition).

        Robustness: candidates that produce a non-finite loss (e.g. when
        the underlying pulse drives the integrator into a NaN — typical
        for very narrow DRAG envelopes) are skipped with a warning.  For
        the duration of the scan, :class:`qml_essentials.yaqsi.Yaqsi` is
        switched into ``throw=False`` mode so a single bad candidate
        cannot abort the loop with ``MaxStepsReached``; the previous
        defaults are restored on exit.

        Args:
            init_pulse_params: Initial pulse parameters to compare against.
            total_cost: Combined cost callable (same as Stage 1).

        Returns:
            Tuple of:
            - Best pulse parameters found during the scan.
            - ``(grid_axes, landscape_data)`` if the grid scan ran, else
              ``None``.  ``landscape_data`` is a list of
              ``(candidate_index, original_params, loss)`` tuples for
              every successful scan candidate.
        """

        def total_cost_log(log_params, *args):
            return total_cost(self._from_log_space(log_params), *args)

        best_scan_params = init_pulse_params
        best_scan_loss = _safe_eval(total_cost, init_pulse_params)
        if not jnp.isfinite(best_scan_loss):
            log.warning(
                "Stage 0: initial pulse parameters produced a non-finite "
                "loss; falling back to a placeholder loss of +inf."
            )

        landscape_data: list = []
        axes_out: Optional[List[jnp.ndarray]] = None

        if self.scan_steps > 0:
            log.info(
                f"Stage 0: Grid scan with {self.scan_grid_size}^"
                f"{len(init_pulse_params)} candidates, "
                f"{self.scan_steps} steps each"
            )

            grid, axes_out = self._build_scan_grid(
                len(init_pulse_params),
                init_pulse_params=init_pulse_params,
            )
            log.info(f"  Total candidates: {len(grid)}")

            # Use a fast Adam for the scan phase.  The aggressive 5×
            # multiplier originally used here tended to push refined
            # candidates *out* of good basins; 2× keeps the refinement
            # localised.  Always-evaluate-the-raw-candidate below
            # additionally guards against this.
            scan_optimizer = optax.chain(
                optax.clip_by_global_norm(
                    self.grad_clip if self.grad_clip > 0 else 1.0
                ),
                optax.adam(self.learning_rate * 2),
            )

            @jax.jit
            def refine_candidate(log_candidate):
                """Run ``self.scan_steps`` Adam steps on a single candidate.

                Fused into a single ``jax.lax.scan`` so the whole
                refinement is one XLA program — no per-step host
                syncs, no Python-loop dispatch.  Returns the final
                log-params and a scalar bool ``failed`` flag (set if
                any intermediate update produced a non-finite value).
                """

                opt_state0 = scan_optimizer.init(log_candidate)

                def body(carry, _):
                    log_p, opt_state, failed = carry
                    loss, grads = jax.value_and_grad(total_cost_log)(log_p)
                    updates, opt_state = scan_optimizer.update(
                        grads, opt_state, log_p
                    )
                    new_log_p = optax.apply_updates(log_p, updates)
                    new_failed = failed | (~jnp.all(jnp.isfinite(new_log_p)))
                    # Freeze on failure so subsequent steps cannot
                    # propagate NaNs further.
                    new_log_p = jnp.where(new_failed, log_p, new_log_p)
                    return (new_log_p, opt_state, new_failed), loss

                (final_log_p, _, failed), _ = jax.lax.scan(
                    body,
                    (log_candidate, opt_state0, jnp.bool_(False)),
                    None,
                    length=self.scan_steps,
                )
                return final_log_p, failed

            # Switch the underlying ODE solver to non-throwing mode for
            # the duration of the scan so candidates that exceed the step
            # budget produce NaN unitaries (and therefore +inf losses)
            # rather than aborting the whole grid loop.
            prev_solver_defaults = ys.Yaqsi.set_solver_defaults(throw=False)
            n_skipped = 0
            n_raw_better = 0
            try:
                for ci, candidate in enumerate(grid):
                    log_candidate = self._to_log_space(candidate)

                    # Evaluate the raw (unrefined) candidate so an
                    # over-aggressive refinement step cannot discard
                    # an already-good grid point.
                    raw_loss = _safe_eval(total_cost, candidate)

                    try:
                        log_p, failed_flag = refine_candidate(log_candidate)
                    except Exception as exc:  # pragma: no cover - defensive
                        log.debug(
                            f"  Candidate {ci + 1}/{len(grid)} "
                            f"raised during refinement: {exc}; skipping."
                        )
                        physical_p = candidate
                        loss = raw_loss
                    else:
                        if bool(failed_flag):
                            physical_p = candidate
                            loss = raw_loss
                        else:
                            physical_p = self._from_log_space(log_p)
                            if not jnp.all(jnp.isfinite(physical_p)):
                                physical_p = candidate
                                loss = raw_loss
                            else:
                                loss = _safe_eval(total_cost, physical_p)

                    # Keep the better of (raw, refined) for this candidate.
                    if jnp.isfinite(raw_loss) and (
                        not jnp.isfinite(loss) or raw_loss < loss
                    ):
                        physical_p = candidate
                        loss = raw_loss
                        n_raw_better += 1

                    if not jnp.isfinite(loss):
                        n_skipped += 1
                        continue

                    landscape_data.append((ci, candidate, float(loss)))

                    if loss < best_scan_loss:
                        best_scan_loss = loss
                        best_scan_params = physical_p
                        log.info(
                            f"  Candidate {ci + 1}/{len(grid)}: "
                            f"loss={float(loss):.6e} improved with "
                            f"params={physical_p}"
                        )
            finally:
                # Always restore the previous solver defaults so other
                # callers (including Stage 1) are unaffected.
                if prev_solver_defaults:
                    ys.Yaqsi.set_solver_defaults(**prev_solver_defaults)

            if n_skipped:
                log.warning(
                    f"Stage 0: skipped {n_skipped}/{len(grid)} candidates "
                    f"due to solver failure or non-finite loss "
                    f"(typical for very narrow / very large-amplitude "
                    f"DRAG pulses)."
                )
            if n_raw_better:
                log.info(
                    f"Stage 0: {n_raw_better}/{len(grid)} candidates "
                    f"were better unrefined than after the {self.scan_steps}-"
                    f"step refinement; raw values were kept."
                )

            log.info(
                f"Stage 0 complete. Best loss: "
                f"{float(best_scan_loss):.6e}, "
                f"params: {best_scan_params}"
            )

        scan_data = (axes_out, landscape_data) if self.scan_steps > 0 else None
        return best_scan_params, scan_data

    def stage_1_opt(
        self, best_scan_params: jnp.ndarray, total_costs: Callable
    ) -> Tuple[jnp.ndarray, list, jnp.ndarray]:
        """Run multi-restart gradient optimisation (Stage 1).

        Performs ``n_restarts`` independent AdamW runs with the full
        (weighted) cost function.  The first restart uses
        ``best_scan_params`` directly; subsequent restarts add random
        perturbations.  Parameters specified in ``log_scale_params`` are
        optimised in log-space.

        When ``n_restarts == 1`` we keep the original single-restart
        Python loop (it preserves per-step ``log.info`` granularity
        and avoids the vmap/scan compilation overhead).  When
        ``n_restarts > 1`` we ``vmap`` the optimiser over restarts and
        run the inner step loop with :func:`jax.lax.scan`, fusing all
        ``n_restarts × n_steps`` steps into a single XLA program.

        Args:
            best_scan_params: Starting parameters (typically from Stage 0).
            total_costs: Combined cost callable.

        Returns:
            Tuple of ``(best_params, loss_history, best_loss)`` from the
            best restart.
        """

        # Wrap the cost function with log-space reparameterisation
        def total_costs_log(log_params):
            return total_costs(self._from_log_space(log_params))

        # Build learning rate schedule
        warmup_steps = int(self.n_steps * self.warmup_ratio)
        end_value = self.learning_rate * self.end_lr_ratio

        if warmup_steps > 0 or self.end_lr_ratio < 1.0:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=(end_value if warmup_steps > 0 else self.learning_rate),
                peak_value=self.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=self.n_steps,
                end_value=end_value,
            )
        else:
            schedule = self.learning_rate

        optimizer = _build_optimizer(schedule, self.grad_clip)

        if self.n_restarts <= 1:
            return self._stage_1_sequential(
                best_scan_params, total_costs, total_costs_log, optimizer
            )
        return self._stage_1_parallel(
            best_scan_params, total_costs, total_costs_log, optimizer
        )

    def _perturb_starts(
        self, start_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Pre-build the ``(n_restarts, n_params)`` matrix of restart starts.

        Restart 0 is the unperturbed start; subsequent restarts add
        Gaussian noise scaled by ``max(|start|, 0.1) *
        restart_noise_scale``.  Indices that are optimised in
        log-space (plus the evolution time at index ``-1``) are kept
        positive via ``jnp.abs`` so the subsequent ``log`` is safe.
        """
        n_params = start_params.shape[0]
        keys = jax.random.split(self.random_key, self.n_restarts)
        # Shape (n_restarts, n_params); restart 0 is intentionally zero
        # noise so the unperturbed start is preserved.
        noise = jax.vmap(
            lambda k: jax.random.normal(k, shape=(n_params,))
        )(keys)
        noise = noise.at[0].set(0.0)
        scale = jnp.maximum(jnp.abs(start_params), 0.1) * self.restart_noise_scale
        starts = start_params[None, :] + noise * scale[None, :]

        # Keep the evolution time and any log-scaled indices positive.
        positive_mask = np.zeros(n_params, dtype=bool)
        positive_mask[-1] = True
        for idx in self.log_scale_params:
            i = idx if idx >= 0 else n_params + idx
            if 0 <= i < n_params:
                positive_mask[i] = True
        positive_mask_j = jnp.asarray(positive_mask)
        starts = jnp.where(positive_mask_j[None, :], jnp.abs(starts), starts)
        return starts

    def _stage_1_sequential(
        self,
        start_params: jnp.ndarray,
        total_costs: Callable,
        total_costs_log: Callable,
        optimizer,
    ) -> Tuple[jnp.ndarray, list, jnp.ndarray]:
        """Single-restart Stage 1, fused into a single ``jax.lax.scan``.

        The whole optimisation loop (n_steps × Adam updates) compiles
        to one XLA program, eliminating the per-step Python overhead
        and per-step host/device syncs that the previous Python ``for``
        loop incurred.  Early stopping is preserved via *masked
        updates*: once the patience condition trips, subsequent steps
        leave the parameters and loss unchanged.  Compute is not
        skipped (lax.scan has fixed length) but the optimiser state
        and parameter trajectory freeze, matching the previous
        early-stop semantics modulo wall-clock savings.
        """

        params = start_params
        log_params = self._to_log_space(params)
        opt_state = optimizer.init(log_params)

        init_loss = total_costs(params)
        min_delta = self.early_stop_min_delta
        patience = self.early_stop_patience
        # ``patience <= 0`` ⇒ early stopping disabled.  Use a large
        # constant so the masked-update path is never triggered.
        eff_patience = patience if patience > 0 else self.n_steps + 1

        def scan_body(carry, _):
            (
                log_params,
                opt_state,
                best_loss,
                best_log_params,
                steps_since_improve,
                stopped_flag,
                stopped_step,
                step_idx,
            ) = carry

            loss, grads = jax.value_and_grad(total_costs_log)(log_params)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, log_params
            )
            stepped_log_params = optax.apply_updates(log_params, updates)

            # Improvement test (uses the pre-update loss, matching the
            # original semantics where the loss recorded on step *i*
            # corresponds to the params *before* that step's update).
            improved = loss < best_loss - min_delta
            best_loss = jnp.where(improved, loss, best_loss)
            # Save the params that *produced* the improving loss
            # (i.e. the pre-update ``log_params``).  ``improved`` is a
            # scalar bool and broadcasts against the 1-D params arrays.
            best_log_params = jnp.where(
                improved, log_params, best_log_params
            )
            steps_since_improve = jnp.where(
                improved, jnp.int32(0), steps_since_improve + jnp.int32(1)
            )

            # Latch the early-stop flag once it fires.
            trigger = steps_since_improve >= jnp.int32(eff_patience)
            new_stopped_flag = stopped_flag | trigger
            stopped_step = jnp.where(
                stopped_flag,
                stopped_step,
                jnp.where(trigger, step_idx + jnp.int32(1), stopped_step),
            )

            # Mask the update once stopped: freeze params/optimiser.
            new_log_params = jnp.where(
                new_stopped_flag, log_params, stepped_log_params
            )
            new_opt_state_kept = jax.tree_util.tree_map(
                lambda new, old: jnp.where(new_stopped_flag, old, new),
                new_opt_state,
                opt_state,
            )

            new_carry = (
                new_log_params,
                new_opt_state_kept,
                best_loss,
                best_log_params,
                steps_since_improve,
                new_stopped_flag,
                stopped_step,
                step_idx + jnp.int32(1),
            )
            return new_carry, loss

        init_carry = (
            log_params,                       # log_params
            opt_state,                        # opt_state
            init_loss,                        # best_loss
            log_params,                       # best_log_params
            jnp.int32(0),                     # steps_since_improve
            jnp.bool_(False),                 # stopped_flag
            jnp.int32(self.n_steps),          # stopped_step (default = n_steps)
            jnp.int32(0),                     # step_idx
        )

        @jax.jit
        def run_scan(carry):
            return jax.lax.scan(scan_body, carry, None, length=self.n_steps)

        final_carry, step_losses = run_scan(init_carry)
        (
            _,
            _,
            best_loss,
            best_log_params,
            _,
            stopped_flag,
            stopped_step,
            _,
        ) = final_carry

        # One sync: pull just what we need for logging in a single
        # device->host transfer instead of a per-step ``.item()`` call.
        host_step_losses, host_best_loss, host_stopped, host_stopped_step = (
            jax.device_get((step_losses, best_loss, stopped_flag, stopped_step))
        )

        # Periodic progress log (replaces the per-step inline log;
        # cheap because step losses already live on the host).
        for step in range(0, self.n_steps, max(1, self.log_interval)):
            log.info(
                f"Step {step}/{self.n_steps}, "
                f"Loss: {float(host_step_losses[step]):.3e}"
            )
        if bool(host_stopped):
            log.info(
                f"Early stop at step {int(host_stopped_step)}/{self.n_steps} "
                f"(no improvement > {min_delta:g} for "
                f"{self.early_stop_patience} steps)."
            )

        log.info(
            f"Restart 1/1 finished with best loss: {float(host_best_loss):.3e}"
            + (
                f" (early stopped at step {int(host_stopped_step)})"
                if bool(host_stopped)
                else ""
            )
        )

        # Reconstruct the historical loss list shape: leading entry is
        # the initial (pre-step-0) loss, followed by one entry per
        # scan step.  Match the previous return type (``list``) so
        # downstream plotting code is unchanged.
        loss_history = [init_loss] + list(step_losses)

        best_pulse_params = self._from_log_space(best_log_params)
        return best_pulse_params, loss_history, best_loss

    def _stage_1_parallel(
        self,
        start_params: jnp.ndarray,
        total_costs: Callable,
        total_costs_log: Callable,
        optimizer,
    ) -> Tuple[jnp.ndarray, list, jnp.ndarray]:
        """Vmap+scan Stage 1: all restarts × all steps in one XLA program.

        Always runs the full ``n_steps``: an early-stop break would
        require either chunking the scan (extra Python overhead) or
        masking updates inside the scan (no compute saved), and
        because every restart would have to plateau before we could
        break, the win is small.  Sequential mode (``n_restarts == 1``)
        does honour ``early_stop_patience``.
        """

        # (n_restarts, n_params) starting points (restart 0 unperturbed).
        params_batch = self._perturb_starts(start_params)
        log.info(
            f"Stage 1 (parallel): vmapping {self.n_restarts} restarts × "
            f"{self.n_steps} steps in a single fused program."
        )
        if self.early_stop_patience > 0:
            log.info(
                "Note: early_stop_patience is ignored in the parallel "
                "(n_restarts > 1) path; the full n_steps will run."
            )

        log_params_batch = jax.vmap(self._to_log_space)(params_batch)
        opt_state_batch = jax.vmap(optimizer.init)(log_params_batch)

        # Initial losses (per-restart) so loss_history[0] matches the
        # per-restart sequential semantics.
        init_losses = jax.vmap(total_costs)(params_batch)

        def opt_step(log_params, opt_state):
            loss, grads = jax.value_and_grad(total_costs_log)(log_params)
            updates, opt_state = optimizer.update(grads, opt_state, log_params)
            log_params = optax.apply_updates(log_params, updates)
            return log_params, opt_state, loss

        v_opt_step = jax.vmap(opt_step, in_axes=(0, 0))

        def scan_body(carry, _):
            log_params, opt_state, prev_log_params, best_loss, best_log_params = carry
            new_log_params, new_opt_state, loss = v_opt_step(log_params, opt_state)
            # Track best loss (and the params that *produced* it,
            # which are the pre-update ``prev_log_params`` — same
            # rationale as the sequential path).
            improved = loss < best_loss
            best_loss = jnp.where(improved, loss, best_loss)
            best_log_params = jnp.where(
                improved[:, None], prev_log_params, best_log_params
            )
            new_carry = (
                new_log_params,
                new_opt_state,
                log_params,  # becomes prev for the next step
                best_loss,
                best_log_params,
            )
            return new_carry, loss

        init_carry = (
            log_params_batch,
            opt_state_batch,
            log_params_batch,
            init_losses,
            log_params_batch,
        )

        @jax.jit
        def run_scan(carry):
            return jax.lax.scan(scan_body, carry, None, length=self.n_steps)

        final_carry, step_losses = run_scan(init_carry)
        # step_losses shape (n_steps, n_restarts); each row is the
        # cross-restart loss vector at one optimisation step.
        _, _, _, best_losses, best_log_params_batch = final_carry

        # Periodic batch summary so the operator still sees progress.
        # Pull the small per-step loss matrix to host once, then format
        # without further device→host transfers.
        host_step_losses = jax.device_get(step_losses)
        for step in range(0, self.n_steps, max(1, self.log_interval)):
            row = host_step_losses[step]
            log.info(
                f"Step {step}/{self.n_steps}, "
                f"loss min/mean/max: {float(row.min()):.3e} / "
                f"{float(row.mean()):.3e} / {float(row.max()):.3e}"
            )

        # Per-restart final summary (single sync for ``best_losses``).
        host_best_losses = jax.device_get(best_losses)
        for r in range(self.n_restarts):
            log.info(
                f"Restart {r + 1}/{self.n_restarts} finished "
                f"with best loss: {float(host_best_losses[r]):.3e}"
            )

        winner = int(jnp.argmin(best_losses))
        global_best_loss = best_losses[winner]
        global_best_params = self._from_log_space(best_log_params_batch[winner])

        # Build a per-step loss history for the winning restart so the
        # downstream API (and the loss-curve plot) keeps the same
        # shape as before.
        winner_history = [init_losses[winner]]
        winner_history.extend(step_losses[:, winner])
        return global_best_params, winner_history, global_best_loss

    def plot_loss_landscape(
        self,
        gate_name: str,
        grid_axes: List[jnp.ndarray],
        landscape_data: list,
    ) -> None:
        """Save a loss-landscape figure for the Phase-0 grid scan.

        The visualisation adapts to the number of pulse parameters:

        - **1 parameter**: line/scatter plot (param value vs. loss).
        - **2 parameters**: 2-D heatmap (param₀ × param₁, colour = loss).
        - **≥ 3 parameters**: horizontal scatter sorted by ascending loss
          with the best candidate highlighted.

        The figure is saved to ``{file_dir}/{gate_name}_loss_landscape.png``.

        Args:
            gate_name: Name of the gate being optimised (e.g. ``"RX"``).
            grid_axes: Per-parameter 1-D arrays that span the scan grid.
            landscape_data: List of ``(candidate_index, params, loss)``
                tuples for every successful scan candidate.
        """
        import matplotlib.pyplot as plt  # lazy — matplotlib is dev-only

        if not landscape_data:
            log.warning("plot_loss_landscape: no landscape data to plot, skipping.")
            return

        os.makedirs(self.file_dir, exist_ok=True)
        n_params = len(grid_axes)
        indices, _params_list, losses = zip(*landscape_data)
        losses_arr = np.array(losses, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))

        if n_params == 1:
            x = np.array([float(grid_axes[0][i]) for i in indices])
            sc = ax.scatter(x, losses_arr, c=losses_arr, cmap="viridis_r", s=60, zorder=3)
            fig.colorbar(sc, ax=ax, label="Loss")
            best_i = int(np.argmin(losses_arr))
            ax.scatter(
                x[best_i], losses_arr[best_i],
                marker="*", s=200, color="red", zorder=4, label="best",
            )
            ax.set_xlabel("Parameter value")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()

        elif n_params == 2:
            n = self.scan_grid_size
            loss_grid = np.full((n, n), np.nan)
            for ci, _, loss in landscape_data:
                row = ci // n
                col = ci % n
                loss_grid[row, col] = loss
            masked = np.ma.masked_invalid(loss_grid)
            cmap = plt.cm.viridis_r.copy()
            cmap.set_bad(color="lightgrey")
            im = ax.imshow(
                masked, origin="lower", cmap=cmap, aspect="auto",
                extent=[
                    float(grid_axes[1][0]), float(grid_axes[1][-1]),
                    float(grid_axes[0][0]), float(grid_axes[0][-1]),
                ],
            )
            fig.colorbar(im, ax=ax, label="Loss")
            ax.set_xlabel("Parameter 1")
            ax.set_ylabel("Parameter 0")

        else:  # n_params >= 3: sorted scatter
            order = np.argsort(losses_arr)
            sorted_losses = losses_arr[order]
            sorted_indices = np.array(indices)[order]  # original trial numbers
            ranks = np.arange(len(sorted_losses))
            sc = ax.scatter(
                sorted_losses, ranks, c=sorted_indices, cmap="plasma", s=40, zorder=3,
            )
            fig.colorbar(sc, ax=ax, label="Trial number")
            ax.scatter(
                sorted_losses[0], ranks[0],
                marker="*", s=200, color="red", zorder=4, label="best",
            )
            ax.set_xlabel("Loss")
            ax.set_ylabel("Candidate rank (0 = best)")
            ax.set_xscale("log")
            ax.legend()

        ax.set_title(f"Loss Landscape (Phase 0) — {gate_name}")
        fig.tight_layout()
        path = os.path.join(self.file_dir, f"{gate_name}_loss_landscape.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Loss landscape saved to {path}")

    def plot_loss_curve(
        self,
        gate_name: str,
        loss_history: list,
    ) -> None:
        """Save a training-loss curve figure for the Phase-1 optimisation.

        Shows loss vs. optimisation step on a log y-scale with a dashed
        horizontal line at the minimum achieved loss.

        The figure is saved to ``{file_dir}/{gate_name}_loss_curve.png``.

        Args:
            gate_name: Name of the gate being optimised (e.g. ``"RX"``).
            loss_history: Sequence of loss values, one per step (including
                the initial loss at index 0).
        """
        import matplotlib.pyplot as plt  # lazy — matplotlib is dev-only

        if not loss_history:
            log.warning("plot_loss_curve: empty loss history, skipping.")
            return

        os.makedirs(self.file_dir, exist_ok=True)
        losses = [float(v) for v in loss_history]
        best = min(losses)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(losses, linewidth=1.2, label="Loss")
        ax.axhline(best, color="red", linestyle="--", linewidth=1.0,
                   label=f"Best: {best:.3e}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title(f"Training Loss (Phase 1) — {gate_name}")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(self.file_dir, f"{gate_name}_loss_curve.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Loss curve saved to {path}")

    def optimize(self, wires: int) -> Callable:
        """Decorator factory that optimises pulse parameters for a gate.

        Usage::

            opt = qoc.optimize(wires=1)
            best_params, loss_history = opt(qoc.create_RX)()

        Args:
            wires: Number of qubits the gate acts on.

        Returns:
            A decorator that accepts a circuit-factory function and
            returns a callable ``(init_pulse_params=None) ->
            (best_params, loss_history)``.
        """

        def decorator(create_circuits):
            def wrapper(init_pulse_params: jnp.ndarray = None):
                """
                Optimise pulse parameters for a quantum gate using a
                multi-phase strategy:

                Stage 0 - Grid scan (if ``scan_steps > 0``):
                    Evaluate a coarse grid of parameter candidates using
                    the same weighted cost as Stage 1.  Each candidate
                    is refined with a few fast gradient steps.  The
                    best candidate becomes the starting point for
                    Stage 1, unless the user-supplied init_pulse_params
                    are already better.

                Stage 1 - Multi-restart gradient optimisation:
                    Run ``n_restarts`` independent Adam optimisation runs
                    with the full cost function.  The first restart uses
                    the best point found so far; subsequent restarts add
                    random perturbations.  Parameters at indices in
                    ``log_scale_params`` are optimised in log-space to
                    handle order-of-magnitude differences in scale.

                Args:
                    init_pulse_params (array): Initial pulse parameters.
                        If ``None``, uses the envelope defaults from
                        :class:`PulseInformation`.

                Returns:
                    tuple: ``(best_params, loss_history)`` from the best
                        restart.
                """
                pulse_circuit, target_circuit = create_circuits()

                # Build a second pair that prepends a Hadamard on every
                # wire so the cost is also evaluated from the
                # ``|+⟩^⊗n`` initial state.  Probing two non-collinear
                # initial states exposes rotation-axis tilt to the
                # optimiser: an RX/RY pulse with a residual Z component
                # is partly degenerate from ``|0⟩`` alone but produces
                # a clearly distinguishable trajectory from ``|+⟩``.
                # Both circuits get the same preparation so the target
                # remains exact.
                def _with_plus_prep(circuit_fn):
                    def prepared(*args, **kwargs):
                        for q in range(wires):
                            op.H(wires=q)
                        circuit_fn(*args, **kwargs)
                    prepared.__name__ = f"plus_{circuit_fn.__name__}"
                    return prepared

                pulse_circuit_plus = _with_plus_prep(pulse_circuit)
                target_circuit_plus = _with_plus_prep(target_circuit)

                pulse_scripts = [
                    ys.Script(pulse_circuit, n_qubits=wires),
                    ys.Script(pulse_circuit_plus, n_qubits=wires),
                ]
                target_scripts = [
                    ys.Script(target_circuit, n_qubits=wires),
                    ys.Script(target_circuit_plus, n_qubits=wires),
                ]

                d_basis = 2**wires
                pulse_basis_scripts = [
                    ys.Script(_with_basis_prep(pulse_circuit, k, wires), n_qubits=wires)
                    for k in range(d_basis)
                ]
                target_basis_scripts = [
                    ys.Script(_with_basis_prep(target_circuit, k, wires), n_qubits=wires)
                    for k in range(d_basis)
                ]

                gate_name = create_circuits.__name__.split("_")[1]

                if init_pulse_params is None:
                    init_pulse_params = PulseInformation.gate_by_name(gate_name).params
                log.debug(
                    f"Initial pulse parameters for {gate_name}: {init_pulse_params}"
                )

                all_ckwargs = {
                    "pulse_scripts": pulse_scripts,
                    "target_scripts": target_scripts,
                    "pulse_basis_scripts": pulse_basis_scripts,
                    "target_basis_scripts": target_basis_scripts,
                    "envelope": self.envelope,
                    "n_samples": self.n_samples,
                    "n_qubits": wires,
                    "t_target": self.t_target,
                }

                def _build_cost(name, weight):
                    """Build a Cost from a registry entry, filtering ckwargs."""
                    meta = CostFnRegistry.get(name)
                    return Cost(
                        cost=meta["fn"],
                        weight=weight,
                        ckwargs={
                            k: v
                            for k, v in all_ckwargs.items()
                            if k in meta["ckwargs_keys"]
                        },
                    )

                total_costs = None
                for name, weight in self.cost_fns:
                    total_costs = _build_cost(name, weight) + total_costs

                # Stage 0 now uses the same weighted objective as Stage 1
                # so the two phases share a basin (previously Stage 0
                # used a fidelity-only cost which led the grid scan into
                # local minima with biased phase residuals — see plan
                # notes for details).
                best_scan_params, scan_data = self.stage_0_opt(
                    init_pulse_params,
                    total_costs,
                )

                global_best_params, global_best_history, global_best_loss = (
                    self.stage_1_opt(
                        best_scan_params,
                        total_costs,
                    )
                )
                self.save_results(
                    gate=gate_name,
                    fidelity=1 - global_best_loss.item(),
                    pulse_params=global_best_params,
                )

                if self.plot:
                    if scan_data is not None:
                        grid_axes, landscape_items = scan_data
                        self.plot_loss_landscape(gate_name, grid_axes, landscape_items)
                    self.plot_loss_curve(gate_name, global_best_history)

                return global_best_params, global_best_history

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Per-gate (pulse, target) circuit factories
    # ------------------------------------------------------------------
    #
    # Each entry maps a gate name to a ``(pulse_circuit, target_circuit)``
    # pair.  The per-gate variants prepend a symmetry-breaking
    # preparation (e.g. ``op.H``/``op.RY``) so the *state-vector* cost
    # is sensitive to rotation-axis tilt.  The joint-mode variants drop
    # those preps because the unitary cost already captures axis tilt
    # without probe-state trickery (see :meth:`_create_joint_pair_for`).

    @staticmethod
    def _gate_factories() -> Dict[str, Tuple[Callable, Callable]]:
        """Return the ``{gate_name: (pulse_fn, target_fn)}`` table.

        Constructed lazily inside a staticmethod so the closures
        capture the imported gate symbols at call time.
        """
        # Single-qubit rotations
        def rx_p(w, pp):
            Gates.RX(w, 0, pulse_params=pp, gate_mode="pulse")
        def rx_t(w):
            op.RX(w, wires=0)

        def ry_p(w, pp):
            Gates.RY(w, 0, pulse_params=pp, gate_mode="pulse")
        def ry_t(w):
            op.RY(w, wires=0)

        def rz_p(w, pp):
            op.H(wires=0)
            Gates.RZ(w, 0, pulse_params=pp, gate_mode="pulse")
            op.H(wires=0)
        def rz_t(w):
            op.H(wires=0)
            op.RZ(w, wires=0)
            op.H(wires=0)

        def h_p(w, pp):
            op.RY(w, wires=0)
            Gates.H(0, pulse_params=pp, gate_mode="pulse")
        def h_t(w):
            op.RY(w, wires=0)
            op.H(wires=0)

        def rot_p(w, pp):
            op.H(wires=0)
            Gates.Rot(w, w * 2, w * 3, 0, pulse_params=pp, gate_mode="pulse")
        def rot_t(w):
            op.H(wires=0)
            op.Rot(w, w * 2, w * 3, wires=0)

        # Two-qubit gates — preps shaped to expose tilt for each gate.
        def _two_q(prep, pulse_gate, target_gate, prep_w_sets_angle=False):
            def pulse_circuit(w, pp):
                prep(w)
                pulse_gate(w, pp)
            def target_circuit(w):
                prep(w)
                target_gate(w)
            return pulse_circuit, target_circuit

        def cx_prep(w):
            op.RY(w, wires=0)
            op.H(wires=1)
        def cy_prep(w):
            op.RX(w, wires=0)
            op.H(wires=1)
        def cz_prep(w):
            op.RY(w, wires=0)
            op.H(wires=1)
        def cr_single_prep(w):
            op.H(wires=0)
        def crz_prep(w):
            op.H(wires=0)
            op.H(wires=1)

        cx = _two_q(
            cx_prep,
            lambda w, pp: Gates.CX(wires=[0, 1], pulse_params=pp, gate_mode="pulse"),
            lambda w: op.CX(wires=[0, 1]),
        )
        cy = _two_q(
            cy_prep,
            lambda w, pp: Gates.CY(wires=[0, 1], pulse_params=pp, gate_mode="pulse"),
            lambda w: op.CY(wires=[0, 1]),
        )
        cz = _two_q(
            cz_prep,
            lambda w, pp: Gates.CZ(wires=[0, 1], pulse_params=pp, gate_mode="pulse"),
            lambda w: op.CZ(wires=[0, 1]),
        )
        crx = _two_q(
            cr_single_prep,
            lambda w, pp: Gates.CRX(w, wires=[0, 1], pulse_params=pp, gate_mode="pulse"),
            lambda w: op.CRX(w, wires=[0, 1]),
        )
        cry = _two_q(
            cr_single_prep,
            lambda w, pp: Gates.CRY(w, wires=[0, 1], pulse_params=pp, gate_mode="pulse"),
            lambda w: op.CRY(w, wires=[0, 1]),
        )
        crz = _two_q(
            crz_prep,
            lambda w, pp: Gates.CRZ(w, wires=[0, 1], pulse_params=pp, gate_mode="pulse"),
            lambda w: op.CRZ(w, wires=[0, 1]),
        )

        return {
            "RX": (rx_p, rx_t),
            "RY": (ry_p, ry_t),
            "RZ": (rz_p, rz_t),
            "H":  (h_p,  h_t),
            "Rot": (rot_p, rot_t),
            "CX": cx,
            "CY": cy,
            "CZ": cz,
            "CRX": crx,
            "CRY": cry,
            "CRZ": crz,
        }

    @staticmethod
    def _joint_gate_factories() -> Dict[str, Tuple[Callable, Callable]]:
        """``(pulse, target)`` pairs without any symmetry-breaking preps.

        Used by :meth:`_create_joint_pair_for`: the unitary cost
        already exposes rotation-axis tilt without a probe state, and
        leaving the preps in actively *hides* certain errors (e.g.
        ``op.H(wires=1)`` puts the target qubit of CX into a CX
        eigenstate, so the column-stacked unitary becomes insensitive
        to the pulse error).  ``Rot`` and ``CY`` are intentionally
        absent because the joint optimiser does not target them.
        """
        def rx_p(w, pp):
            Gates.RX(w, wires=0, pulse_params=pp, gate_mode="pulse")
        def rx_t(w):
            op.RX(w, wires=0)
        def ry_p(w, pp):
            Gates.RY(w, wires=0, pulse_params=pp, gate_mode="pulse")
        def ry_t(w):
            op.RY(w, wires=0)
        def rz_p(w, pp):
            Gates.RZ(w, wires=0, pulse_params=pp, gate_mode="pulse")
        def rz_t(w):
            op.RZ(w, wires=0)
        def h_p(w, pp):
            Gates.H(0, pulse_params=pp, gate_mode="pulse")
        def h_t(w):
            op.H(wires=0)
        def cz_p(w, pp):
            Gates.CZ(wires=[0, 1], pulse_params=pp, gate_mode="pulse")
        def cz_t(w):
            op.CZ(wires=[0, 1])
        def cx_p(w, pp):
            Gates.CX(wires=[0, 1], pulse_params=pp, gate_mode="pulse")
        def cx_t(w):
            op.CX(wires=[0, 1])
        def crx_p(w, pp):
            Gates.CRX(w, wires=[0, 1], pulse_params=pp, gate_mode="pulse")
        def crx_t(w):
            op.CRX(w, wires=[0, 1])
        def cry_p(w, pp):
            Gates.CRY(w, wires=[0, 1], pulse_params=pp, gate_mode="pulse")
        def cry_t(w):
            op.CRY(w, wires=[0, 1])
        def crz_p(w, pp):
            Gates.CRZ(w, wires=[0, 1], pulse_params=pp, gate_mode="pulse")
        def crz_t(w):
            op.CRZ(w, wires=[0, 1])

        return {
            "RX": (rx_p, rx_t), "RY": (ry_p, ry_t), "RZ": (rz_p, rz_t),
            "H":  (h_p,  h_t),
            "CZ": (cz_p, cz_t), "CX": (cx_p, cx_t),
            "CRX": (crx_p, crx_t), "CRY": (cry_p, cry_t), "CRZ": (crz_p, crz_t),
        }

    def _create_pair(self, gate_name: str) -> Tuple[Callable, Callable]:
        """Look up the per-gate ``(pulse, target)`` pair from the table."""
        try:
            return self._gate_factories()[gate_name]
        except KeyError as exc:
            raise ValueError(f"No factory for gate {gate_name!r}.") from exc

    # Thin compatibility wrappers around :meth:`_create_pair` so existing
    # code (and tests) that call ``qoc.create_<gate>`` keep working.
    def create_RX(self):  return self._create_pair("RX")
    def create_RY(self):  return self._create_pair("RY")
    def create_RZ(self):  return self._create_pair("RZ")
    def create_H(self):   return self._create_pair("H")
    def create_Rot(self): return self._create_pair("Rot")
    def create_CX(self):  return self._create_pair("CX")
    def create_CY(self):  return self._create_pair("CY")
    def create_CZ(self):  return self._create_pair("CZ")
    def create_CRX(self): return self._create_pair("CRX")
    def create_CRY(self): return self._create_pair("CRY")
    def create_CRZ(self): return self._create_pair("CRZ")

    def optimize_all(self, sel_gates: str, make_log: bool) -> None:
        """Optimise all selected gates and optionally write a log CSV.

        Args:
            sel_gates: Comma-separated gate names or ``"all"``.
            make_log: If ``True``, write per-gate loss histories to
                ``qml_essentials/qoc_logs.csv``.
        """
        # Joint mode (Round 3) is now implemented in :meth:`optimize_joint`.
        # The `--joint` CLI flag selects it instead of this per-gate loop.
        log_history: Dict[str, list] = {}

        for gate in self.GATES_1Q + self.GATES_2Q:
            if gate in sel_gates or "all" in sel_gates:
                n_wires = 1 if gate in self.GATES_1Q else 2
                opt = self.optimize(wires=n_wires)
                gate_factory = getattr(self, f"create_{gate}")
                log.info(f"Optimizing {gate} gate...")
                optimized_pulse_params, loss_history = opt(gate_factory)()
                log.info(f"Optimized parameters for {gate}: {optimized_pulse_params}")
                best_fid = 1 - min(float(loss) for loss in loss_history)
                log.info(f"Best achieved fidelity: {best_fid * 100:.5f}%")
                log_history[gate] = log_history.get(gate, []) + loss_history

        if make_log:
            # write log history to file
            with open("qml_essentials/qoc_logs.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(log_history.keys())
                writer.writerows(zip(*log_history.values()))

    # ------------------------------------------------------------------
    # Joint composite-aware optimisation (Round 3)
    # ------------------------------------------------------------------

    # Default leaf set whose parameters are jointly optimised.  Order
    # matters — it determines the layout of the joint parameter vector
    # (theta).  Excluding a leaf from this list freezes it at its
    # current PulseInformation default during joint optimisation.
    JOINT_LEAVES_DEFAULT: Tuple[str, ...] = ("RX", "RY", "RZ", "CZ")

    # Default set of target gates whose unitary cost is summed during
    # joint optimisation.  Composite gates back-propagate into the
    # shared leaves; leaf-gate terms keep the standalone fidelity
    # acceptable.  CZ is excluded from the default targets because it
    # is implemented as a static diagonal-Hamiltonian evolution
    # (``H_CZ = π·|11⟩⟨11|``, t=1) that is structurally exact and
    # cannot be improved by tuning leaf parameters — including it only
    # adds ballast to the averaged loss.
    JOINT_TARGETS_DEFAULT: Tuple[str, ...] = (
        "RX", "RY", "RZ", "H", "CX", "CRX", "CRY", "CRZ",
    )

    # Default per-target weights for the joint objective.  Weights are
    # normalised inside :func:`joint_unitary_cost_fn`.  Composites are
    # up-weighted because (a) they are what fails the tightened tests
    # and (b) standalone leaves already start near-perfect, so the
    # averaged loss would otherwise be dominated by the cheap leaves
    # and the optimiser would happily refuse to move.  Within
    # composites, CR_ are weighted higher than H/CX because they are
    # the longest decompositions (2 CX + ~6 single-qubit gates) so
    # their leaf-error compounding is worst.
    JOINT_WEIGHTS_DEFAULT: Dict[str, float] = {
        "RX": 0.3, "RY": 0.3, "RZ": 0.3,
        "H": 1.0, "CX": 2.0,
        "CRX": 3.0, "CRY": 3.0, "CRZ": 3.0,
    }

    # Leaves that are physically identical up to a static carrier-phase
    # offset (RX uses cos(ω_c t), RY uses cos(ω_c t + π/2)) and therefore
    # *should* share the same envelope parameters.  Tying them here in
    # the QOC layout — rather than in :mod:`pulses` — keeps the per-gate
    # decomposition tree intact while ensuring joint optimisation cannot
    # drift their envelopes apart.  Empirically RY is the dominant
    # contributor to H/CX residuals, so leaving it un-tied lets the
    # joint loss settle into a basin where RX is well-tuned but RY is
    # ~3× worse; tying them removes that asymmetry.
    JOINT_TIED_GROUPS_DEFAULT: Tuple[Tuple[str, ...], ...] = (
        ("RX", "RY"),
    )

    def _build_joint_layout(
        self, leaf_names: Tuple[str, ...],
        tied_groups: Optional[Tuple[Tuple[str, ...], ...]] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, slice], List[int]]:
        """Build the joint parameter layout.

        Args:
            leaf_names: Ordered names of the leaf gates that participate
                in the joint optimisation.
            tied_groups: Optional tuple of leaf-name groups whose
                parameters are forced to share a single slice in
                ``theta``.  Defaults to
                :pyattr:`JOINT_TIED_GROUPS_DEFAULT` (ties RX/RY).  Only
                leaves that are present in ``leaf_names`` participate —
                a group becomes a no-op if fewer than two of its
                members are listed.

        Returns:
            Tuple ``(init_theta, leaf_slices, log_scale_indices)``:
              * ``init_theta`` — concatenated init parameters from
                ``PulseInformation.<leaf>.params`` in the given order.
                For tied groups, the representative leaf is the *first*
                member in the group (the group's mean of current params
                is used as the shared init so neither side dominates).
              * ``leaf_slices`` — mapping leaf-name → ``slice`` into
                ``init_theta``.  Tied leaves map to the *same* slice.
              * ``log_scale_indices`` — indices into ``init_theta`` that
                should be optimised in log-space (amplitude + evolution
                time per envelope leaf, mirroring the per-gate default
                ``[0, -1]`` rule).
        """
        if tied_groups is None:
            tied_groups = self.JOINT_TIED_GROUPS_DEFAULT

        # Build leaf_name -> representative_name lookup.  Members of a
        # tied group are routed to the group's first member that is
        # actually present in ``leaf_names``.
        rep_of: Dict[str, str] = {n: n for n in leaf_names}
        leaf_set = set(leaf_names)
        for group in tied_groups:
            present = [n for n in group if n in leaf_set]
            if len(present) < 2:
                continue
            head = present[0]
            for member in present[1:]:
                rep_of[member] = head
                log.info(
                    f"  Joint layout: tying leaf {member!r} to {head!r} "
                    f"(shared slice in theta)."
                )

        envelope_info = PulseEnvelope.get(self.envelope)
        n_env = envelope_info["n_envelope_params"]

        leaf_slices: Dict[str, slice] = {}
        init_chunks = []
        log_idx: List[int] = []
        offset = 0
        for name in leaf_names:
            rep = rep_of[name]
            if rep != name:
                # Tied member — point at the representative's slice.
                leaf_slices[name] = leaf_slices[rep]
                continue

            pp = PulseInformation.gate_by_name(name)
            assert pp is not None and pp.is_leaf, (
                f"_build_joint_layout: {name!r} is not a leaf gate"
            )
            # For tied groups the shared init is the elementwise mean
            # of the current params across all present members; this
            # avoids biasing toward whichever member happens to be the
            # group representative.
            tied_members = [m for m in leaf_names if rep_of[m] == name]
            if len(tied_members) > 1:
                stacked = jnp.stack([
                    jnp.asarray(
                        PulseInformation.gate_by_name(m).params,
                        dtype=jnp.float64,
                    )
                    for m in tied_members
                ])
                chunk = jnp.mean(stacked, axis=0)
            else:
                chunk = jnp.asarray(pp.params, dtype=jnp.float64)
            n_p = chunk.shape[0]
            leaf_slices[name] = slice(offset, offset + n_p)
            init_chunks.append(chunk)
            # Log-scale rule per leaf: only leaves that come from the
            # *envelope* (RX, RY) get log-scaled amplitude+time.  RZ
            # and CZ use the "general" registry with a single tuning
            # scalar — leave them in linear space.
            if name in ("RX", "RY") and n_env >= 2:
                log_idx.append(offset)            # amplitude
                log_idx.append(offset + n_p - 1)  # evolution time
            offset += n_p

        init_theta = jnp.concatenate(init_chunks)
        return init_theta, leaf_slices, log_idx

    @staticmethod
    def _assemble_for_gate(
        theta: jnp.ndarray,
        pp_obj,
        leaf_slices: Dict[str, slice],
    ) -> jnp.ndarray:
        """Assemble the per-gate flat ``pulse_params`` from ``theta``.

        Walks the gate's decomposition tree (recursing through
        composites) and concatenates the appropriate slice of ``theta``
        for each leaf occurrence.  Mirrors :pyattr:`PulseParams.params`
        getter logic but pulls leaf data from the joint vector
        ``theta`` rather than the leaves' own ``_params``.
        """
        if pp_obj.is_leaf:
            sl = leaf_slices.get(pp_obj.name)
            if sl is None:
                # Leaf is frozen — use its current PulseInformation
                # value directly.
                return jnp.asarray(pp_obj.params, dtype=jnp.float64)
            return theta[sl]
        return jnp.concatenate(
            [QOC._assemble_for_gate(theta, child, leaf_slices) for child in pp_obj.childs]
        )

    def _joint_stage_0_coord_descent(
        self,
        init_theta: jnp.ndarray,
        leaf_slices: Dict[str, slice],
        total_cost: Callable,
    ) -> jnp.ndarray:
        """Coordinate-descent grid scan over leaf-axis blocks.

        For each leaf in ``leaf_slices`` (in order), sweep a centred
        multiplicative grid over that leaf's params (using the existing
        :meth:`_build_scan_grid` machinery) while holding the other
        leaves at the current best.  Greedily accept any improvement.

        This avoids the combinatorial explosion of a Cartesian
        product over all leaf axes simultaneously: instead of
        ``Π_i scan_grid_size**k_i`` candidates, only ``Σ_i
        scan_grid_size**k_i`` are evaluated.

        Args:
            init_theta: Starting joint parameter vector.
            leaf_slices: Mapping leaf-name → slice into ``init_theta``.
            total_cost: Joint cost callable taking ``theta`` and
                returning a scalar loss.

        Returns:
            Best joint parameter vector found.
        """
        if self.scan_steps <= 0:
            log.info("Joint Stage 0: scan disabled (scan_steps=0); skipping.")
            return init_theta

        current = init_theta
        best_loss = _safe_eval(total_cost, current)
        log.info(
            f"Joint Stage 0: coordinate-descent over {len(leaf_slices)} leaves, "
            f"init_loss={float(best_loss):.6e}"
        )

        prev_solver_defaults = ys.Yaqsi.set_solver_defaults(throw=False)
        try:
            seen_slices: set = set()
            for leaf_name, sl in leaf_slices.items():
                # Tied leaves share a slice — only scan the unique
                # (start, stop) range once to avoid wasted evaluations.
                key = (sl.start, sl.stop)
                if key in seen_slices:
                    continue
                seen_slices.add(key)
                leaf_init = current[sl]
                n_p = int(leaf_init.shape[0])
                if n_p == 0:
                    continue
                grid, _ = self._build_scan_grid(n_p, init_pulse_params=leaf_init)
                n_better = 0
                for cand in grid:
                    new_theta = current.at[sl].set(cand)
                    loss = _safe_eval(total_cost, new_theta)
                    if loss < best_loss:
                        best_loss = loss
                        current = new_theta
                        n_better += 1
                log.info(
                    f"  Joint scan after leaf {leaf_name} "
                    f"({len(grid)} candidates, {n_better} improved): "
                    f"best_loss={float(best_loss):.6e}"
                )
        finally:
            if prev_solver_defaults:
                ys.Yaqsi.set_solver_defaults(**prev_solver_defaults)

        return current

    def _create_joint_pair_for(self, gate_name: str):
        """Return a prep-free ``(pulse, target)`` pair for joint mode.

        Looks up :meth:`_joint_gate_factories` first; falls back to the
        per-gate (preps included) variant via :meth:`_create_pair_for`
        with a warning if the gate is not in the joint table.  See the
        joint-table docstring for why preps are dropped.
        """
        table = self._joint_gate_factories()
        if gate_name in table:
            return table[gate_name]
        log.warning(
            f"_create_joint_pair_for: no prep-free factory for {gate_name!r}; "
            f"falling back to create_{gate_name} (preps may hide errors)."
        )
        return self._create_pair_for(gate_name)

    def _create_pair_for(self, gate_name: str):
        """Return ``(pulse_circuit, target_circuit)`` for a target gate.

        Reuses :meth:`_create_pair` so the joint mode targets exactly
        the same circuits as the per-gate mode.
        """
        return self._create_pair(gate_name)

    def optimize_joint(
        self,
        target_gates: Optional[List[str]] = None,
        leaf_names: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, slice], list]:
        """Joint composite-aware optimisation of leaf pulse parameters.

        Optimises a single shared parameter vector ``theta`` (containing
        the concatenated leaf params for ``leaf_names``) against a
        weighted sum of unitary-cost terms over ``target_gates``.
        Composite gates back-propagate into the shared leaves; leaf
        terms keep the standalone fidelity acceptable.  CZ is omitted
        from the default targets because the ``PulseGates.CZ``
        implementation is a static diagonal-Hamiltonian evolution
        (``H_CZ = π·|11⟩⟨11|``, t=1) that is structurally exact and
        unaffected by any leaf re-tuning.

        Args:
            target_gates: Gates whose unitary cost contributes to the
                joint objective.  Defaults to
                :pyattr:`JOINT_TARGETS_DEFAULT` (RX, RY, RZ, H, CX,
                CRX, CRY, CRZ).
            leaf_names: Leaf gates whose parameters are jointly
                optimised.  Defaults to :pyattr:`JOINT_LEAVES_DEFAULT`
                (RX, RY, RZ, CZ).
            weights: Optional mapping ``gate_name → weight``.  Merged
                on top of :pyattr:`JOINT_WEIGHTS_DEFAULT` (composites
                up-weighted; leaves down-weighted).  All weights are
                normalised inside the cost.

        Returns:
            ``(best_theta, leaf_slices, loss_history)``.  Per-leaf
            results are also written to ``qoc_results_<envelope>.csv``
            via :meth:`save_results`.
        """
        target_gates = list(target_gates) if target_gates else list(self.JOINT_TARGETS_DEFAULT)
        leaf_names = list(leaf_names) if leaf_names else list(self.JOINT_LEAVES_DEFAULT)
        # Merge user-provided weights on top of class defaults so callers
        # can override only the gates they care about.
        merged_weights: Dict[str, float] = dict(self.JOINT_WEIGHTS_DEFAULT)
        if weights:
            merged_weights.update({k: float(v) for k, v in weights.items()})
        weights = merged_weights

        log.info(
            f"Joint optimisation: leaves={leaf_names}, targets={target_gates}"
        )

        init_theta, leaf_slices, joint_log_idx = self._build_joint_layout(
            tuple(leaf_names)
        )
        log.info(
            f"  Joint theta size: {init_theta.shape[0]}; "
            f"log-scale indices: {joint_log_idx}"
        )

        # Build per-gate specs (assembler + basis-prep scripts).
        gate_specs: List[dict] = []
        for gname in target_gates:
            pp_obj = PulseInformation.gate_by_name(gname)
            if pp_obj is None:
                log.warning(f"  Skipping unknown gate {gname!r}.")
                continue
            n_wires = 1 if gname in self.GATES_1Q else 2
            d_basis = 2 ** n_wires
            pulse_circuit, target_circuit = self._create_joint_pair_for(gname)

            pulse_basis_scripts = [
                ys.Script(_with_basis_prep(pulse_circuit, k, n_wires), n_qubits=n_wires)
                for k in range(d_basis)
            ]
            target_basis_scripts = [
                ys.Script(_with_basis_prep(target_circuit, k, n_wires), n_qubits=n_wires)
                for k in range(d_basis)
            ]

            # Closure capturing pp_obj + leaf_slices.  Defined here so
            # each spec carries its own assembler.
            def _make_assembler(pp_obj=pp_obj):
                def assemble(theta):
                    return QOC._assemble_for_gate(theta, pp_obj, leaf_slices)
                return assemble

            gate_specs.append({
                "name": gname,
                "n_qubits": n_wires,
                "weight": float(weights.get(gname, 1.0)),
                "assembler": _make_assembler(),
                "pulse_basis_scripts": pulse_basis_scripts,
                "target_basis_scripts": target_basis_scripts,
            })
            log.info(
                f"  Built spec for {gname}: n_qubits={n_wires}, "
                f"weight={gate_specs[-1]['weight']}"
            )

        # Build the joint cost as a Cost wrapper (so weight-tuple
        # collapsing into a scalar is shared with the per-gate path).
        # We use the same (process_loss, phase_loss) two-component
        # weighting as the standalone unitary cost — keeps the relative
        # importance of fidelity vs phase consistent.
        ((_, weight_tuple),) = (
            (n, w) for n, w in self.cost_fns if n == "unitary"
        ) if any(n == "unitary" for n, _ in self.cost_fns) else ((None, (0.5, 0.5)),)
        joint_cost = Cost(
            cost=joint_unitary_cost_fn,
            weight=weight_tuple,
            ckwargs={
                "gate_specs": gate_specs,
                "n_samples": self.n_samples,
            },
        )

        # Temporarily override log_scale_params to point at joint
        # vector indices (Stage 0 grid building + Stage 1 log-space
        # reparam both consult ``self.log_scale_params``).  Invalidate
        # the mask cache on either side of the swap so the joint
        # vector picks up the joint indices and per-gate runs revert
        # cleanly afterwards.
        prev_log_scale = self.log_scale_params
        self.log_scale_params = joint_log_idx
        self._log_mask_cache.clear()
        try:
            best_scan_theta = self._joint_stage_0_coord_descent(
                init_theta, leaf_slices, joint_cost
            )

            global_best_theta, global_best_history, global_best_loss = (
                self.stage_1_opt(best_scan_theta, joint_cost)
            )
        finally:
            self.log_scale_params = prev_log_scale
            self._log_mask_cache.clear()

        log.info(
            f"Joint optimisation done. final loss={float(global_best_loss):.6e}"
        )

        # Save per-leaf results to the CSV (one row per leaf).  The
        # fidelity column carries the *joint* fidelity; downstream code
        # that reads the CSV (or the user copy-pasting into pulses.py)
        # can use it as a coarse quality signal.
        joint_fid = float(1.0 - global_best_loss)
        for leaf_name, sl in leaf_slices.items():
            leaf_params = global_best_theta[sl]
            self.save_results(
                gate=leaf_name,
                fidelity=joint_fid,
                pulse_params=leaf_params,
            )

        # Update PulseInformation in-place so the new defaults are
        # active in this Python process (handy for diagnostic scripts
        # that import QOC and then evaluate the new gates).
        for leaf_name, sl in leaf_slices.items():
            pp = PulseInformation.gate_by_name(leaf_name)
            pp.params = global_best_theta[sl]

        return global_best_theta, leaf_slices, global_best_history


default_qoc_params = {
    "envelope": "drag",
    "cost_fns": [
        # Unitary-level cost (process infidelity + trace-phase term).
        # Captures rotation-axis tilt and global-phase residual that
        # the state-fidelity cost is blind to; required to keep two-CX
        # composites (CRX/CRY/CRZ) within tightened phase tolerances.
        ("unitary", (0.5, 0.5)),
        # ("fidelity", (0.5, 0.5)),  # legacy state-vector cost
        # ("pulse_width", 0.000000015),
        # ("evolution_time", 0.000000005),
    ],
    "t_target": 0.5,
    "n_steps": 800,
    "n_samples": 20,
    "learning_rate": 0.0001,
    "warmup_ratio": 0.05,
    "end_lr_ratio": 0.01,
    "log_interval": 50,
    "file_dir": None,
    "n_restarts": 5,
    "restart_noise_scale": 0.01,
    "grad_clip": 1.0,
    "random_seed": 1000,
    "scan_steps": 20,
    "scan_grid_size": 4,
    "scan_ranges": None,
    "log_scale_params": None,
    "early_stop_patience": 0,
    "early_stop_min_delta": 0.0,
}


def profile_pulse_pipeline(
    gate: str = "RX",
    n_samples: int = 3,
    rwa: Optional[bool] = None,
    n_qubits: int = 1,
) -> dict:
    """Profile a single pulse gate's forward + ``value_and_grad`` pass.

    Diagnostic helper for the JIT pipeline.  Builds a minimal
    :class:`Script` that applies the requested pulse gate, then
    times JIT compilation and steady-state evaluation of:

    1. one forward pass (``Script.execute(type="state", ...)``);
    2. one ``jax.value_and_grad`` of the squared overlap with the
       analytic ``operations.<gate>`` target.

    Use this to measure the impact of the RWA toggle
    (``rwa=True``) and of the scan/sync refactors documented in
    the patch notes:

        from qml_essentials.qoc import profile_pulse_pipeline
        profile_pulse_pipeline("RX", rwa=False)
        profile_pulse_pipeline("RX", rwa=True)

    Args:
        gate: Gate name to profile (default ``"RX"``).  Must be a
            single-qubit pulse-level gate (``RX`` / ``RY``).
        n_samples: Number of timed evaluations after warm-up.
        rwa: If not ``None``, temporarily switch the global RWA flag
            for the duration of the profile.  ``None`` keeps the
            current setting.
        n_qubits: Width of the script (kept at 1 for the single-
            qubit pulse gates).

    Returns:
        Dict with keys ``compile_fwd``, ``mean_fwd``, ``compile_grad``,
        ``mean_grad``, ``rwa``, ``loss``.
    """
    import time

    prev_rwa = PulseInformation.get_rwa()
    if rwa is not None:
        PulseInformation.set_rwa(bool(rwa))
    try:
        from qml_essentials.pulses import PulseGates
        gate_op = getattr(op, gate)
        gate_pulse = getattr(PulseGates, gate)

        def pulse_circuit(theta, pp):
            gate_pulse(theta, wires=0, pulse_params=pp)

        def target_circuit(theta):
            gate_op(theta, wires=0)

        pulse_script = ys.Script(pulse_circuit, n_qubits=n_qubits)
        target_script = ys.Script(target_circuit, n_qubits=n_qubits)

        theta = jnp.asarray(jnp.pi / 4)
        pp = PulseInformation.gate_by_name(gate).params
        target_state = target_script.execute(type="state", args=(theta,))
        target_state = jax.lax.stop_gradient(target_state)

        @jax.jit
        def fwd(theta, pp):
            return pulse_script.execute(type="state", args=(theta, pp))

        @jax.jit
        def loss_and_grad(pp):
            def loss_fn(p):
                state = pulse_script.execute(type="state", args=(theta, p))
                return 1.0 - jnp.abs(jnp.vdot(target_state, state)) ** 2

            return jax.value_and_grad(loss_fn)(pp)

        # Warm-up + compile timings.
        t0 = time.perf_counter()
        s = fwd(theta, pp)
        jax.block_until_ready(s)
        compile_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss, grads = loss_and_grad(pp)
        jax.block_until_ready(loss)
        jax.block_until_ready(grads)
        compile_grad = time.perf_counter() - t0

        fwd_t, grad_t = [], []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            s = fwd(theta, pp)
            jax.block_until_ready(s)
            fwd_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            loss, grads = loss_and_grad(pp)
            jax.block_until_ready(loss)
            jax.block_until_ready(grads)
            grad_t.append(time.perf_counter() - t0)

        result = {
            "gate": gate,
            "rwa": PulseInformation.get_rwa(),
            "compile_fwd": compile_fwd,
            "mean_fwd": float(np.mean(fwd_t)),
            "compile_grad": compile_grad,
            "mean_grad": float(np.mean(grad_t)),
            "loss": float(loss),
        }
        log.info(
            f"[profile] gate={gate} rwa={result['rwa']} "
            f"compile fwd/grad: {compile_fwd * 1e3:.1f}/"
            f"{compile_grad * 1e3:.1f} ms, "
            f"mean fwd/grad: {result['mean_fwd'] * 1e3:.1f}/"
            f"{result['mean_grad'] * 1e3:.1f} ms, "
            f"loss={result['loss']:.4e}"
        )
        return result
    finally:
        PulseInformation.set_rwa(prev_rwa)


if __name__ == "__main__":
    # argparse the selected gate
    parser = argparse.ArgumentParser(
        description="Quantum Optimal Control — pulse-level gate synthesis."
    )
    parser.add_argument(
        "--gates",
        type=str,
        nargs="+",
        default=["RX", "RY", "RZ", "CZ"],
        choices=QOC.GATES_1Q + QOC.GATES_2Q + ["all"],
        help="Gate(s) to optimize.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Log results to file (default: False).",
    )
    parser.add_argument(
        "--no-log",
        action="store_false",
        dest="log",
        help="Disable logging results to file.",
    )
    parser.add_argument(
        "--envelope",
        type=str,
        default=default_qoc_params["envelope"],
        choices=PulseEnvelope.available(),
        help="Pulse envelope shape to use for optimization.",
    )
    parser.add_argument(
        "--costs",
        type=str,
        nargs="+",
        default=default_qoc_params["cost_fns"],
        help=(
            "Cost functions and weights as 'name:w1,w2,...' strings. "
            "If weights are omitted the registry defaults are used. "
            f"Available: {CostFnRegistry.available()}. "
            "Example: --costs fidelity:0.5,0.3 pulse_width:0.2"
        ),
    )
    parser.add_argument(
        "--t_target",
        type=float,
        default=default_qoc_params["t_target"],
        help=(
            "Target evolution time for the 'evolution_time' cost function. "
            "All gates will be softly encouraged towards this common time."
        ),
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=default_qoc_params["n_steps"],
        help="Number of optimisation steps per gate.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=default_qoc_params["n_samples"],
        help="Number of parameter samples in [0, 2\\pi] for cost evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=default_qoc_params["learning_rate"],
        help="Peak learning rate for the AdamW optimiser.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=default_qoc_params["warmup_ratio"],
        help=(
            "Fraction of n_steps used for linear LR warmup (0.0-1.0). "
            "Set to 0 to start at the peak LR immediately."
        ),
    )
    parser.add_argument(
        "--end_lr_ratio",
        type=float,
        default=default_qoc_params["end_lr_ratio"],
        help=(
            "Final LR as a fraction of --learning_rate after cosine decay. "
            "Also used as the initial LR before warmup. "
            "Set to 1.0 (with --warmup_ratio 0) for a constant LR."
        ),
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=default_qoc_params["log_interval"],
        help="Log the current loss every N steps.",
    )
    parser.add_argument(
        "--file_dir",
        type=str,
        default=default_qoc_params["file_dir"],
        help="Directory to save qoc_results_[envelope].csv. " \
        "Defaults to the package directory.",
    )
    parser.add_argument(
        "--n_restarts",
        type=int,
        default=default_qoc_params["n_restarts"],
        help=(
            "Number of random restarts for the optimisation. "
            "The first run uses the initial parameters as-is; "
            "subsequent runs add random perturbations. "
            "The best result across all restarts is kept."
        ),
    )
    parser.add_argument(
        "--restart_noise_scale",
        type=float,
        default=default_qoc_params["restart_noise_scale"],
        help=(
            "Standard deviation of Gaussian noise added to the initial "
            "parameters for each restart, relative to parameter magnitude."
        ),
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=default_qoc_params["grad_clip"],
        help=(
            "Maximum global gradient norm. Gradients are clipped to this "
            "value before being passed to the optimiser. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=default_qoc_params["random_seed"],
        help="Base random seed for restart perturbations.",
    )
    parser.add_argument(
        "--scan_steps",
        type=int,
        default=default_qoc_params["scan_steps"],
        help=(
            "Number of short gradient-descent steps per candidate in the "
            "coarse grid scan (Stage 0).  Set to 0 to disable the grid scan."
        ),
    )
    parser.add_argument(
        "--scan_grid_size",
        type=int,
        default=default_qoc_params["scan_grid_size"],
        help=(
            "Number of points per parameter dimension in the coarse grid. "
            "Total candidates = scan_grid_size^n_params."
        ),
    )
    parser.add_argument(
        "--scan_ranges",
        type=str,
        nargs="*",
        default=default_qoc_params["scan_ranges"],
        help=(
            "Per-parameter (lo,hi) ranges for the grid scan, given as "
            "'lo,hi' strings. One pair per pulse parameter. "
            "Example: --scan_ranges 0.5,30.0 0.05,2.0 0.05,2.0 "
            "If omitted, heuristic defaults are used."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help=(
            "Save a loss-landscape plot (Phase 0) and a loss-curve plot "
            "(Phase 1) as PNG files in --file_dir for each optimised gate."
        ),
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=default_qoc_params["early_stop_patience"],
        help=(
            "Number of consecutive Stage-1 steps without improvement "
            "(> --early_stop_min_delta) after which optimisation exits "
            "early. 0 disables early stopping (default)."
        ),
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=default_qoc_params["early_stop_min_delta"],
        help=(
            "Minimum loss decrease that counts as an improvement for "
            "the --early_stop_patience counter (default 0.0)."
        ),
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        default=False,
        help=(
            "Use composite-aware joint optimisation: a single shared "
            "leaf parameter vector is optimised against the unitary "
            "cost summed over leaf and composite gates "
            "(default targets: RX, RY, RZ, CZ, H, CX, CRX, CRY, CRZ). "
            "Pulls leaves into a basin that works well in *every* "
            "use-site instead of only standalone, fixing the "
            "selfish-basin failure mode of per-gate optimisation. "
            "Ignores --gates."
        ),
    )
    parser.add_argument(
        "--joint_targets",
        nargs="+",
        type=str,
        default=None,
        help=(
            "(Used only with --joint.) Override the list of target "
            "gates whose unitary cost contributes to the joint "
            "objective."
        ),
    )
    parser.add_argument(
        "--joint_leaves",
        nargs="+",
        type=str,
        default=None,
        help=(
            "(Used only with --joint.) Override the list of leaf "
            "gates whose parameters are jointly optimised. "
            "Default: RX RY RZ CZ."
        ),
    )

    parser.add_argument(
        "--joint_weights",
        nargs="+",
        type=str,
        default=None,
        help=(
            "(Used only with --joint.) Override per-target weights as "
            "'gate:weight' strings (e.g. --joint_weights CRX:5 CX:3). "
            "Merged on top of QOC.JOINT_WEIGHTS_DEFAULT, so unspecified "
            "gates keep their default weight."
        ),
    )

    args = parser.parse_args()
    sel_gates = args.gates  # already a list from nargs="+"
    make_log = args.log

    # Parse scan_ranges from CLI (list of "lo,hi" strings -> list of tuples)
    scan_ranges = None
    if args.scan_ranges is not None:
        scan_ranges = []
        for pair in args.scan_ranges:
            lo, hi = pair.split(",")
            scan_ranges.append((float(lo), float(hi)))

    # Parse cost function specs from CLI
    cost_fns = [CostFnRegistry.parse_cost_arg(spec) for spec in args.costs]

    # create logger
    log = logging.getLogger("qml_essentials.qoc")

    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    qoc = QOC(
        envelope=args.envelope,
        cost_fns=cost_fns,
        t_target=args.t_target,
        n_steps=args.n_steps,
        n_samples=args.n_samples,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        end_lr_ratio=args.end_lr_ratio,
        log_interval=args.log_interval,
        file_dir=args.file_dir,
        n_restarts=args.n_restarts,
        restart_noise_scale=args.restart_noise_scale,
        grad_clip=args.grad_clip,
        random_seed=args.random_seed,
        scan_steps=args.scan_steps,
        scan_grid_size=args.scan_grid_size,
        scan_ranges=scan_ranges,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        plot=args.plot,
    )

    if args.joint:
        joint_weights = None
        if args.joint_weights:
            joint_weights = {}
            for spec in args.joint_weights:
                gname, w = spec.split(":")
                joint_weights[gname.strip()] = float(w)
        qoc.optimize_joint(
            target_gates=args.joint_targets,
            leaf_names=args.joint_leaves,
            weights=joint_weights,
        )
    else:
        qoc.optimize_all(sel_gates=sel_gates, make_log=make_log)
