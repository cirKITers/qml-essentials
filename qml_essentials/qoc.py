import argparse
import csv
import itertools
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
from jax import numpy as jnp
import optax

from qml_essentials.gates import Gates, PulseInformation, PulseEnvelope
from qml_essentials import operations as op
from qml_essentials import yaqsi as ys
from qml_essentials.math import phase_difference, fidelity

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


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
    pulse_script: ys.Script,
    target_script: ys.Script,
    n_samples: int,
) -> Tuple[float, float]:
    """
    Cost function returning (1 - fidelity) and |phase_difference| averaged
    over *n_samples* uniformly spaced rotation angles in [0, 2\\pi].

    Uses batched (vmapped) circuit execution: all *n_samples* rotation
    angles are evaluated in a single vectorised call per script, replacing
    ``n_samples`` sequential Python-level circuit executions with one
    JIT-compiled XLA program each.

    Args:
        pulse_params: Pulse parameters for evaluation.
        pulse_script: Yaqsi script with pulse parameters.
        target_script: Yaqsi script as target.
        n_samples: Number of parameter samples.

    Returns:
        Tuple of (abs_diff, phase_diff).
    """
    ws = jnp.linspace(0, 2 * jnp.pi, n_samples)

    pulse_states = pulse_script.execute(
        type="state",
        args=(ws, pulse_params),
        in_axes=(0, None),
    )  # (n_samples, dim)

    target_states = target_script.execute(
        type="state",
        args=(ws,),
        in_axes=(0,),
    )  # (n_samples, dim)

    abs_diff = jnp.mean(
        jnp.array(1.0, dtype=jnp.float64) - fidelity(pulse_states, target_states)
    )
    phase_diff = jnp.mean(jnp.abs(phase_difference(pulse_states, target_states)))

    return (abs_diff, phase_diff)


def pulse_width_cost_fn(
    pulse_params: jnp.ndarray,
    envelope: str,
) -> jnp.ndarray:
    """
    Cost function penalising the pulse width (sigma / width).

    The pulse width is taken as the **last** envelope parameter. For
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

    The evolution time is always the **last** element of the pulse parameter
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
    (where ``t_evol`` is the last element of *pulse_params*), computes its
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


# Backward-compatible alias for the old misspelled name
sepctral_density_cost_fn = spectral_density_cost_fn


class CostFnRegistry:
    """Registry of cost functions available for pulse optimisation.

    Use :meth:`register` to add new cost functions at runtime and
    :meth:`get` / :meth:`available` to query them.
    """

    _REGISTRY: Dict[str, dict] = {
        "fidelity": {
            "fn": fidelity_cost_fn,
            "default_weight": (0.5, 0.5),
            "ckwargs_keys": ["pulse_script", "target_script", "n_samples"],
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
        """Look up cost-function metadata by *name*.

        Args:
            name: Registered cost function name.

        Returns:
            Metadata dict with keys ``fn``,
            ``default_weight``, ``ckwargs_keys``.

        Raises:
            ValueError: If *name* is not registered.
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
    quantum gates using a two-stage strategy:

    * **Stage 0** – coarse grid scan (optional).
    * **Stage 1** – multi-restart gradient optimisation with AdamW.

    Attributes:
        GATES_1Q: Names of supported single-qubit gates.
        GATES_2Q: Names of supported two-qubit gates.
    """

    GATES_1Q: List[str] = ["RX", "RY", "RZ", "Rot", "H"]
    GATES_2Q: List[str] = ["CX", "CY", "CZ", "CRX", "CRY", "CRZ"]

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
    ):
        """
        Initialize Quantum Optimal Control with Pulse-level Gates.

        Args:
            envelope (str): Pulse envelope shape to use for optimization.
                Must be one of the registered envelopes in PulseEnvelope
                (e.g. 'gaussian', 'square', 'cosine', 'drag', 'sech').
            cost_fns (list): List of ``(name, weight)`` tuples that select
                which cost functions to use and their weights.  *name* must
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
                Defaults to 0.5 (50 % relative perturbation).
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
                and rely solely on restarts.  A value of 20–50 is
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
            pulse_params: Optimised pulse parameters for the gate.
        """
        if self.file_dir is not None:
            os.makedirs(self.file_dir, exist_ok=True)
            filename = os.path.join(self.file_dir, "qoc_results.csv")

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

    def _to_log_space(self, params: jnp.ndarray) -> jnp.ndarray:
        """Convert selected parameters to log-space for optimisation.

        Parameters at indices in ``self.log_scale_params`` are replaced
        by ``log(|p| + eps)`` so the optimiser operates on a
        logarithmic scale.  All other parameters are left unchanged.

        Args:
            params: Pulse parameters in physical space.

        Returns:
            Parameters with selected entries in log-space.
        """
        if not self.log_scale_params:
            return params
        n = len(params)
        log_params = params.copy()
        for idx in self.log_scale_params:
            # Normalise negative indices
            i = idx if idx >= 0 else n + idx
            log_params = log_params.at[i].set(jnp.log(jnp.abs(params[i]) + 1e-12))
        return log_params

    def _from_log_space(self, log_params: jnp.ndarray) -> jnp.ndarray:
        """Convert selected parameters back from log-space.

        Inverse of :meth:`_to_log_space`.  Parameters at indices in
        ``self.log_scale_params`` are exponentiated; all others are
        passed through unchanged.

        Args:
            log_params: Parameters with selected entries in log-space.

        Returns:
            Parameters in physical space (all positive for log-scaled
            entries).
        """
        if not self.log_scale_params:
            return log_params
        n = len(log_params)
        params = log_params.copy()
        for idx in self.log_scale_params:
            i = idx if idx >= 0 else n + idx
            params = params.at[i].set(jnp.exp(log_params[i]))
        return params

    def _build_scan_grid(self, n_params: int) -> jnp.ndarray:
        """Build a coarse parameter grid for the initial scan phase.

        Uses either user-supplied ``scan_ranges`` or heuristic defaults
        based on typical Gaussian pulse parameter ranges.

        Args:
            n_params: Number of pulse parameters.

        Returns:
            Array of shape ``(n_candidates, n_params)`` with grid points.
        """
        if self.scan_ranges is not None:
            ranges = self.scan_ranges
            assert len(ranges) == n_params, (
                f"scan_ranges has {len(ranges)} entries but gate has "
                f"{n_params} parameters."
            )
        else:
            # [amplitude, sigma/width, evolution_time]
            default_ranges = {
                1: [(0.05, 2.0)],  # evolution time only (general)
                2: [(0.5, 2.0), (0.05, 2.0)],  # not typically used
                3: [(0.5, 30.0), (0.05, 2.0), (0.05, 2.0)],  # A, σ, t
                4: [(0.5, 30.0), (0.05, 2.0), (0.01, 0.5), (0.05, 2.0)],  # DRAG
            }
            ranges = default_ranges.get(
                n_params,
                [(0.1, 10.0)] * n_params,  # fallback
            )

        # Build log-spaced grids for each parameter
        axes = []
        for lo, hi in ranges:
            axes.append(jnp.logspace(jnp.log10(lo), jnp.log10(hi), self.scan_grid_size))

        # Cartesian product of all axes
        grid = jnp.array(list(itertools.product(*axes)))
        return grid

    def stage_0_opt(self, init_pulse_params: jnp.ndarray, fidelity_only_cost):
        """Run the coarse grid-scan phase (Stage 0).

        Evaluates a Cartesian grid of parameter candidates using only the
        fidelity cost (ignoring phase).  Each candidate is refined with a
        few fast gradient steps.  Returns the best-found parameters.

        Args:
            init_pulse_params: Initial pulse parameters to compare against.
            fidelity_only_cost: Cost callable using fidelity only.

        Returns:
            Best pulse parameters found during the scan.
        """

        def fidelity_only_cost_log(log_params, *args):
            return fidelity_only_cost(self._from_log_space(log_params), *args)

        best_scan_params = init_pulse_params
        best_scan_loss = fidelity_only_cost(init_pulse_params)

        if self.scan_steps > 0:
            log.info(
                f"Stage 0: Grid scan with {self.scan_grid_size}^"
                f"{len(init_pulse_params)} candidates, "
                f"{self.scan_steps} steps each"
            )

            grid = self._build_scan_grid(len(init_pulse_params))
            log.info(f"  Total candidates: {len(grid)}")

            # Use a fast, constant-LR Adam for the scan phase
            scan_optimizer = optax.chain(
                optax.clip_by_global_norm(
                    self.grad_clip if self.grad_clip > 0 else 1.0
                ),
                optax.adam(self.learning_rate * 5),  # aggressive LR
            )

            @jax.jit
            def scan_step(opt_state, log_params):
                loss, grads = jax.value_and_grad(fidelity_only_cost_log)(log_params)
                updates, opt_state = scan_optimizer.update(grads, opt_state, log_params)
                log_params = optax.apply_updates(log_params, updates)
                return log_params, opt_state, loss

            for ci, candidate in enumerate(grid):
                log_candidate = self._to_log_space(candidate)
                opt_state = scan_optimizer.init(log_candidate)

                log_p = log_candidate
                for _ in range(self.scan_steps):
                    log_p, opt_state, loss = scan_step(opt_state, log_p)

                # Evaluate final loss
                physical_p = self._from_log_space(log_p)
                loss = fidelity_only_cost(physical_p)

                if loss < best_scan_loss:
                    best_scan_loss = loss
                    best_scan_params = physical_p
                    log.info(
                        f"  Candidate {ci + 1}/{len(grid)}: "
                        f"loss={loss.item():.3e} improved with "
                        f"params={physical_p}"
                    )

            log.info(
                f"Stage 0 complete. Best fidelity-only loss: "
                f"{best_scan_loss.item():.3e}, "
                f"params: {best_scan_params}"
            )

        return best_scan_params

    def stage_1_opt(self, best_scan_params: jnp.ndarray, total_costs):
        """Run multi-restart gradient optimisation (Stage 1).

        Performs ``n_restarts`` independent AdamW runs with the full
        (weighted) cost function.  The first restart uses
        ``best_scan_params`` directly; subsequent restarts add random
        perturbations.  Parameters specified in ``log_scale_params`` are
        optimised in log-space.

        Args:
            best_scan_params: Starting parameters (typically from Stage 0).
            total_costs: Combined cost callable.

        Returns:
            Tuple of ``(best_params, loss_history, best_loss)``.
        """

        # Wrap the cost function with log-space reparameterisation
        def total_costs_log(log_params, *args):
            return total_costs(self._from_log_space(log_params), *args)

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

        # Build optimiser chain with gradient clipping
        use_clip = (
            self.grad_clip and self.grad_clip > 0 and jnp.isfinite(self.grad_clip)
        )
        if use_clip:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.grad_clip),
                optax.adamw(schedule),
            )
        else:
            optimizer = optax.adamw(schedule)

        @jax.jit
        def opt_step(opt_state, log_params, *args):
            loss, grads = jax.value_and_grad(total_costs_log)(log_params, *args)
            updates, opt_state = optimizer.update(grads, opt_state, log_params)
            log_params = optax.apply_updates(log_params, updates)
            return log_params, opt_state, loss

        # Use the best from grid scan as starting point
        start_params = best_scan_params

        global_best_loss = jnp.inf
        global_best_params = start_params
        global_best_history = []
        restart_key = self.random_key

        for restart in range(self.n_restarts):
            if restart == 0:
                params = start_params
            else:
                # Perturb the starting point
                restart_key, sub_key = jax.random.split(restart_key)
                noise = jax.random.normal(sub_key, shape=start_params.shape)
                scale = (
                    jnp.maximum(jnp.abs(start_params), 0.1) * self.restart_noise_scale
                )
                params = start_params + noise * scale
                # Ensure log-scaled params remain positive before
                # conversion (evolution time at index -1 is always
                # included since _to_log_space uses jnp.abs anyway,
                # but we keep values positive for readability).
                params = params.at[-1].set(jnp.abs(params[-1]))
                for idx in self.log_scale_params:
                    i = idx if idx >= 0 else len(params) + idx
                    params = params.at[i].set(jnp.abs(params[i]))
                log.info(
                    f"Restart {restart + 1}/{self.n_restarts} "
                    f"with perturbed params: {params}"
                )

            # Convert to log-space for optimisation
            log_params = self._to_log_space(params)
            opt_state = optimizer.init(log_params)

            loss = total_costs(params)
            loss_history = [loss]
            best_loss = loss
            best_pulse_params = params

            for step in range(self.n_steps):
                if step % self.log_interval == 0:
                    restart_tag = (
                        f" [restart {restart + 1}/{self.n_restarts}]"
                        if self.n_restarts > 1
                        else ""
                    )
                    log.info(
                        f"Step {step}/{self.n_steps}, "
                        f"Loss: {loss_history[-1].item():.3e}"
                        f"{restart_tag}"
                    )

                log_params, opt_state, loss = opt_step(opt_state, log_params)

                if loss < best_loss:
                    log.debug(f"Best set of params found at step {step}")
                    best_loss = loss
                    best_pulse_params = self._from_log_space(log_params)

                loss_history.append(loss)

            log.info(
                f"Restart {restart + 1}/{self.n_restarts} finished "
                f"with best loss: {best_loss.item():.3e}"
            )

            if best_loss < global_best_loss:
                global_best_loss = best_loss
                global_best_params = best_pulse_params
                global_best_history = loss_history

        return global_best_params, global_best_history, global_best_loss

    def optimize(self, wires):
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
                    only the fidelity cost (ignoring phase).  Each
                    candidate is refined with a few fast gradient steps.
                    The best candidate becomes the starting point for
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

                pulse_script = ys.Script(pulse_circuit, n_qubits=wires)
                target_script = ys.Script(target_circuit, n_qubits=wires)

                gate_name = create_circuits.__name__.split("_")[1]

                if init_pulse_params is None:
                    init_pulse_params = PulseInformation.gate_by_name(gate_name).params
                log.debug(
                    f"Initial pulse parameters for {gate_name}: {init_pulse_params}"
                )

                all_ckwargs = {
                    "pulse_script": pulse_script,
                    "target_script": target_script,
                    "envelope": self.envelope,
                    "n_samples": self.n_samples,
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

                fidelity_only_cost = _build_cost(
                    "fidelity", (1.0, 0.0)  # 100% fidelity, 0% phase
                )

                best_scan_params = self.stage_0_opt(
                    init_pulse_params,
                    fidelity_only_cost,
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

                return global_best_params, global_best_history

            return wrapper

        return decorator

    def create_RX(self):
        """Create pulse and target circuits for the RX gate."""

        def pulse_circuit(w, pulse_params):
            Gates.RX(w, 0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RX(w, wires=0)

        return pulse_circuit, target_circuit

    def create_RY(self):
        """Create pulse and target circuits for the RY gate."""

        def pulse_circuit(w, pulse_params):
            Gates.RY(w, 0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)

        return pulse_circuit, target_circuit

    def create_RZ(self):
        """Create pulse and target circuits for the RZ gate.

        Both circuits are sandwiched between Hadamard gates to make the
        RZ rotation observable in the computational basis.
        """

        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.RZ(w, 0, pulse_params=pulse_params, gate_mode="pulse")
            op.H(wires=0)

        def target_circuit(w):
            op.H(wires=0)
            op.RZ(w, wires=0)
            op.H(wires=0)

        return pulse_circuit, target_circuit

    def create_H(self):
        """Create pulse and target circuits for the Hadamard gate.

        An RY rotation is prepended to break symmetry.
        """

        def pulse_circuit(w, pulse_params):
            op.RY(w, wires=0)
            Gates.H(0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)
            op.H(wires=0)

        return pulse_circuit, target_circuit

    def create_Rot(self):
        """Create pulse and target circuits for the general Rot gate."""

        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.Rot(w, w * 2, w * 3, 0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.Rot(w, w * 2, w * 3, wires=0)

        return pulse_circuit, target_circuit

    def create_CX(self):
        """Create pulse and target circuits for the CX (CNOT) gate."""

        def pulse_circuit(w, pulse_params):
            op.RY(w, wires=0)
            op.H(wires=1)
            Gates.CX(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)
            op.H(wires=1)
            op.CX(wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CY(self):
        """Create pulse and target circuits for the CY gate."""

        def pulse_circuit(w, pulse_params):
            op.RX(w, wires=0)
            op.H(wires=1)
            Gates.CY(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RX(w, wires=0)
            op.H(wires=1)
            op.CY(wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CZ(self):
        """Create pulse and target circuits for the CZ gate."""

        def pulse_circuit(w, pulse_params):
            op.RY(w, wires=0)
            op.H(wires=1)
            Gates.CZ(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)
            op.H(wires=1)
            op.CZ(wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CRX(self):
        """Create pulse and target circuits for the CRX gate."""

        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.CRX(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.CRX(w, wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CRY(self):
        """Create pulse and target circuits for the CRY gate."""

        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.CRY(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.CRY(w, wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CRZ(self):
        """Create pulse and target circuits for the CRZ gate."""

        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            op.H(wires=1)
            Gates.CRZ(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.H(wires=1)
            op.CRZ(w, wires=[0, 1])

        return pulse_circuit, target_circuit

    def optimize_all(self, sel_gates: str, make_log: bool) -> None:
        """Optimise all selected gates and optionally write a log CSV.

        Args:
            sel_gates: Comma-separated gate names or ``"all"``.
            make_log: If ``True``, write per-gate loss histories to
                ``qml_essentials/qoc_logs.csv``.
        """
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


default_qoc_params = {
    "envelope": "gaussian",
    "cost_fns": [
        ("fidelity", (0.49999999, 0.49999999)),
        # ("pulse_width", 0.000000015),
        # ("evolution_time", 0.000000005),
    ],
    "t_target": 0.5,
    "n_steps": 1500,
    "n_samples": 20,
    "learning_rate": 0.001,
    "warmup_ratio": 0.05,
    "end_lr_ratio": 0.01,
    "log_interval": 50,
    "file_dir": None,
    "n_restarts": 3,
    "restart_noise_scale": 0.5,
    "grad_clip": 1.0,
    "random_seed": 42,
    "scan_steps": 30,
    "scan_grid_size": 5,
    "scan_ranges": None,
    "log_scale_params": None,
}

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
        default=True,
        help="Log results to file (default: True).",
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
        help="Directory to save qoc_results.csv. Defaults to the package directory.",
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

    args = parser.parse_args()
    sel_gates = args.gates  # already a list from nargs="+"
    make_log = args.log

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
    )

    qoc.optimize_all(sel_gates=sel_gates, make_log=make_log)
