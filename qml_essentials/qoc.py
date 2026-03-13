# flake8: noqa: E402
import argparse
from typing import Dict, List, Callable, Optional, Union, Tuple

import os
import csv
import jax
from jax import numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from qml_essentials.gates import Gates, PulseInformation, PulseEnvelope
from qml_essentials import operations as op
from qml_essentials import yaqsi as ys
from qml_essentials.math import phase_difference, fidelity
import logging

log = logging.getLogger(__name__)


class Cost:
    def __init__(
        self,
        cost: Callable,
        weight: Union[float, Tuple],
        ckwargs: dict = {},
    ):
        self.cost = cost
        self.weight = weight
        self.ckwargs = ckwargs

    def cost(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # inject constant args in cost function
        cost = self.cost(*args, **kwargs, **self.ckwargs)
        if isinstance(self.weight, tuple):
            return jnp.array(
                [c * w for c, w in zip(cost, self.weight, strict=True)]
            ).sum()
        else:
            return cost * self.weight

    def __add__(self, other):
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
    Cost function returning (1 − fidelity) and |phase_difference| averaged
    over *n_samples* uniformly spaced rotation angles in [0, 2π].

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
        envelope: Name of the active pulse envelope (unused).
        t_target (float): Target evolution time (passed via ``**kwargs``).

    Returns:
        Scalar evolution-time cost.

    Raises:
        ValueError: If ``t_target`` is not provided in ``kwargs``.
    """
    t = pulse_params[-1]
    return ((t - t_target) / t_target) ** 2


class CostFnRegistry:
    """Registry of cost functions available for pulse optimisation.

    Use :meth:`register` to add new cost functions at runtime and
    :meth:`get` / :meth:`available` to query them.
    """

    _REGISTRY: Dict[str, dict] = {
        "fidelity": {
            "fn": fidelity_cost_fn,
            "n_weights": 2,
            "default_weight": (0.5, 0.5),
            "ckwargs_keys": ["pulse_script", "target_script", "n_samples"],
        },
        "pulse_width": {
            "fn": pulse_width_cost_fn,
            "n_weights": 1,
            "default_weight": 1.0,
            "ckwargs_keys": ["envelope"],
        },
        "evolution_time": {
            "fn": evolution_time_cost_fn,
            "n_weights": 1,
            "default_weight": 1.0,
            "ckwargs_keys": ["t_target"],
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
            Metadata dict with keys ``fn``, ``n_weights``,
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
                components does not match ``n_weights``.
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
        meta = cls.get(name)
        expected = meta["n_weights"]
        # TODO: maybe we can get rid of n_weights entirely; it's just for validation
        got = len(weight) if isinstance(weight, tuple) else 1
        if got != expected:
            raise ValueError(
                f"Cost function '{name}' expects {expected} weight(s), " f"got {got}."
            )

        return name, weight


class QOC:
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
        """
        self.ws = jnp.linspace(0, 2 * jnp.pi, n_samples)

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
        self.current_gate = None
        self.t_target = t_target

        log.info(
            f"Training parameters: {self.n_steps} steps, {self.n_samples} samples, {self.learning_rate} learning rate"
        )
        log.info(
            f"LR schedule: warmup_ratio={self.warmup_ratio}, end_lr_ratio={self.end_lr_ratio}"
        )

        log.info(f"Envelope: {self.envelope}")
        log.info(f"Target evolution time: {self.t_target}")
        log.info(f"Using cost function(s) {cost_fns}")

        # Validate each entry against the registry
        summed_weights = 0
        for name, _weight in cost_fns:
            CostFnRegistry.get(name)  # raises ValueError if unknown
            summed_weights += sum(_weight) if isinstance(_weight, tuple) else _weight
            # check sume of weights
        assert jnp.isclose(
            summed_weights, 1.0, rtol=1e-8
        ), f"Cost function weights must sum to 1. Got {summed_weights}"

        self.cost_fns = cost_fns

        # Configure the pulse system with the selected envelope
        PulseInformation.set_envelope(self.envelope)

    def save_results(self, gate, fidelity, pulse_params):
        """
        Saves the optimized pulse parameters and fidelity for a given gate to a CSV file

        Args:
            gate (str): Name of the gate.
            fidelity (float): Fidelity of the optimized pulse parameters.
            pulse_params (list): Optimized pulse parameters for the gate.

        Notes:
            If the gate already exists in the file and
            the newly optimized pulse parameters have a higher fidelity,
            the existing entry will be overwritten.
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

    def optimize(self, wires):
        def decorator(create_circuits):
            def wrapper(init_pulse_params: jnp.ndarray = None):
                """
                This function is a wrapper for the create_circuits method.
                It takes a simulator and wires as input and optimizes
                the pulse parameters using the cost function defined
                in the QOC class.

                Args:
                    create_circuits (callable): A function to generate the pulse and
                        target circuits for the gate.
                    init_pulse_params (array): Initial pulse parameters to use for
                        the pulse-based gate.

                Returns:
                    tuple: Optimized pulse parameters and list of loss values
                        at each iteration.
                """
                pulse_circuit, target_circuit = create_circuits()

                pulse_script = ys.Script(pulse_circuit, n_qubits=wires)
                target_script = ys.Script(target_circuit, n_qubits=wires)

                gate_name = create_circuits.__name__.split("_")[1]

                if init_pulse_params is None:
                    log.warning(
                        f"Using initial pulse parameters for {gate_name} \
                            from `ansaetze.py`"
                    )
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
                # Build the composite cost from self.cost_fns
                total_costs = None
                for name, weight in self.cost_fns:
                    meta = CostFnRegistry.get(name)
                    total_costs = (
                        Cost(
                            cost=meta["fn"],
                            weight=weight,
                            ckwargs={
                                k: v
                                for k, v in all_ckwargs.items()
                                if k in meta["ckwargs_keys"]
                            },
                        )
                        + total_costs
                    )

                params = init_pulse_params

                # Build learning rate schedule
                warmup_steps = int(self.n_steps * self.warmup_ratio)
                end_value = self.learning_rate * self.end_lr_ratio

                if warmup_steps > 0 or self.end_lr_ratio < 1.0:
                    schedule = optax.warmup_cosine_decay_schedule(
                        init_value=(
                            end_value if warmup_steps > 0 else self.learning_rate
                        ),
                        peak_value=self.learning_rate,
                        warmup_steps=warmup_steps,
                        decay_steps=self.n_steps,
                        end_value=end_value,
                    )
                else:
                    schedule = self.learning_rate

                optimizer = optax.adamw(schedule)
                opt_state = optimizer.init(params)

                loss = total_costs(params)
                loss_history = [loss]
                best_loss = loss
                best_pulse_params = params

                @jax.jit
                def opt_step(opt_state, params, *args):
                    loss, grads = jax.value_and_grad(total_costs)(params, *args)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return params, opt_state, loss

                for step in range(self.n_steps):
                    if step % self.log_interval == 0:
                        log.info(
                            f"Step {step}/{self.n_steps}, Loss: {loss_history[-1].item():.3e}"
                        )

                    params, opt_state, loss = opt_step(opt_state, params)

                    if loss < best_loss:
                        log.debug(f"Best set of params found at step {step}")
                        best_loss = loss
                        best_pulse_params = params

                    loss_history.append(loss)

                self.save_results(
                    gate=gate_name,
                    fidelity=1 - best_loss.item(),
                    pulse_params=best_pulse_params,
                )

                return best_pulse_params, loss_history

            return wrapper

        return decorator

    def create_RX(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            Gates.RX(w, 0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RX(w, wires=0)

        return pulse_circuit, target_circuit

    def create_RY(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            Gates.RY(w, 0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)

        return pulse_circuit, target_circuit

    def create_RZ(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.RZ(w, 0, pulse_params=pulse_params, gate_mode="pulse")
            op.H(wires=0)

        def target_circuit(w):
            op.H(wires=0)
            op.RZ(w, wires=0)
            op.H(wires=0)

        return pulse_circuit, target_circuit

    def create_H(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.RY(w, wires=0)
            Gates.H(0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)
            op.H(wires=0)

        return pulse_circuit, target_circuit

    def create_Rot(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.Rot(w, w * 2, w * 3, 0, pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.Rot(w, w * 2, w * 3, wires=0)

        return pulse_circuit, target_circuit

    def create_CX(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.RY(w, wires=0)
            op.H(wires=1)
            Gates.CX(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)
            op.H(wires=1)
            op.CX(wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CY(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.RX(w, wires=0)
            op.H(wires=1)
            Gates.CY(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RX(w, wires=0)
            op.H(wires=1)
            op.CY(wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CZ(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.RY(w, wires=0)
            op.H(wires=1)
            Gates.CZ(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.RY(w, wires=0)
            op.H(wires=1)
            op.CZ(wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CRX(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.CRX(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.CRX(w, wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CRY(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            Gates.CRY(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.CRY(w, wires=[0, 1])

        return pulse_circuit, target_circuit

    def create_CRZ(self, init_pulse_params: jnp.ndarray = None):
        def pulse_circuit(w, pulse_params):
            op.H(wires=0)
            op.H(wires=1)
            Gates.CRZ(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")

        def target_circuit(w):
            op.H(wires=0)
            op.H(wires=1)
            op.CRZ(w, wires=[0, 1])

        return pulse_circuit, target_circuit

    def optimize_all(self, sel_gates, make_log):
        log_history = {}

        gates_1q = ["RX", "RY", "RZ", "Rot", "H"]
        gates_2q = ["CX", "CY", "CZ", "CRX", "CRY", "CRZ"]

        for gate in gates_1q + gates_2q:
            if gate in sel_gates or "all" in sel_gates:
                opt = self.optimize(wires=1 if gate in gates_1q else 2)
                gate_factory = getattr(self, f"create_{gate}")
                log.info(f"Optimizing {gate} gate...")
                optimized_pulse_params, loss_history = opt(gate_factory)()
                log.info(f"Optimized parameters for {gate}: {optimized_pulse_params}")
                best_fid = 1 - min(float(l) for l in loss_history)
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
        ("pulse_width", 0.000000015),
        ("evolution_time", 0.000000005),
    ],
    "t_target": 0.5,
    "n_steps": 1500,
    "n_samples": 20,
    "learning_rate": 0.0001,
    "warmup_ratio": 0.05,
    "end_lr_ratio": 0.01,
    "log_interval": 50,
    "file_dir": None,
}

if __name__ == "__main__":
    # argparse the selected gate
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gates",
        type=str,
        default=["RX", "RY", "RZ", "CZ"],
        choices=["all", "RX", "RY", "RZ", "CZ"],
        help="Gate(s) to optimize.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=True,
        choices=[True, False],
        help="Log results to file.",
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
        default=default_qoc_params[
            "cost_fns"
        ],  # be aware of the numerical precision limit!
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
        default=default_qoc_params["t_target"],  # referenz is the RZ gate
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
        help="Number of parameter samples in [0, 2π] for cost evaluation.",
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
        help=(
            "Directory to save qoc_results.csv. " "Defaults to the package directory."
        ),
    )

    args = parser.parse_args()
    sel_gates = str(args.gates)
    make_log = bool(args.log)

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
    )

    qoc.optimize_all(sel_gates=sel_gates, make_log=make_log)
