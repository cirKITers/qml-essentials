# qa: disable

import os
import csv
import jax
from jax import numpy as jnp
import optax
import pennylane as qml
from qml_essentials.gates import Gates, PulseInformation
from qml_essentials import operations as op
from qml_essentials import yaqsi as ys
import argparse
from functools import partial
from typing import List, Callable, Union
import logging

log = logging.getLogger(__name__)


class QOC:
    def __init__(
        self,
        observable: Union[Callable, List[Callable]] = qml.state,
        n_steps: int = 1000,
        n_loops: int = 1,
        n_samples: int = 8,
        learning_rate: float = 0.01,
        log_interval: int = 50,
        skip_on_fidelity: bool = True,
    ):
        """
        Initialize Quantum Optimal Control with Pulse-level Gates.

        Args:
            observable (str): Observable to measure during optimization.
            n_steps (int): Number of steps in optimization.
            n_loops (int): Number of loops for optimization.
            n_samples (int): Number of parameter samples per step.
            learning_rate (float): Learning rate for Adam with
                weight decay regularization.
            log_interval (int): Interval for logging.
            skip_on_fidelity (bool): Skip writing to qoc_results if fidelity is lower?
        """
        self.ws = jnp.linspace(0, 2 * jnp.pi, n_samples)

        self.observable = observable
        self.n_steps = n_steps
        self.n_loops = n_loops
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.skip_on_fidelity = skip_on_fidelity

        self.current_gate = None

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
            If the fidelity is lower, the new entry will be skipped unless
            `skip_on_fidelity=False`.
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
                            if fidelity > float(row[1]):
                                writer.writerow(entry)
                            else:
                                log.warning(
                                    f"Pulse parameters for {gate} already exist with "
                                    f"higher fidelity ({row[1]} >= {fidelity})"
                                )
                                if not self.skip_on_fidelity:
                                    log.info("Overwriting parameters anyway")
                                    writer.writerow(entry)
                                else:
                                    writer.writerow(row)
                            match = True
                        # any other gate
                        else:
                            writer.writerow(row)
                # gate does not exist
                if not match:
                    writer.writerow(entry)

    def cost_fn(self, pulse_params, pulse_script, target_script) -> float:
        """
        Cost function for QOC optimization.

        The cost function is calculated as the average of the fidelity and
        phase difference between the pulse-based and unitary-based gates.

        Args:
            pulse_params (list or array): Optimized parameters to
                use for the pulse-based gate.
            pulse_script (ys.QuantumScript): Pulse-based gate QuantumScript.
            target_script (ys.QuantumScript): Unitary-based gate QuantumScript.

        Returns:
            float: Cost function value.
        """
        abs_diff = 0
        phase_diff = 0
        for w in jnp.arange(0, 2 * jnp.pi, (2 * jnp.pi) / self.n_samples):
            pulse_state = pulse_script.execute(type="state", args=(w, pulse_params))
            target_state = target_script.execute(type="state", args=(w,))
            dot_prod = jnp.vdot(target_state, pulse_state)
            abs_diff += 1 - jnp.abs(dot_prod) ** 2  # one if no diff
            phase_diff += jnp.abs(jnp.angle(dot_prod)) / jnp.pi  # zero if no diff

        abs_diff /= self.n_samples
        phase_diff /= self.n_samples

        return (abs_diff + phase_diff) / 2  # loss

    def multi_objective_cost_fn(
        self, pulse_params, pulse_scripts, target_scripts, leafs
    ) -> float:
        """
        Cost function for QOC optimization.

        The cost function is calculated as the average of the fidelity and
        phase difference between the pulse-based and unitary-based gates.

        Args:
            pulse_params (list or array): Optimized parameters to use for
                the pulse-based gate.
            pulse_scripts (list): List of pulse-based gate QuantumScripts.
            target_scripts (list): List of unitary-based gate QuantumScripts.

        Returns:
            float: Cost function value.
        """
        idx = 0
        for leaf in leafs:
            nidx = idx + leaf.size
            leaf.params = pulse_params[idx:nidx]
            idx = nidx

        abs_diff = 0
        phase_diff = 0
        for pulse_script, target_script in zip(pulse_scripts, target_scripts):
            for w in jnp.arange(0, 2 * jnp.pi, (2 * jnp.pi) / self.n_samples):
                pulse_state = pulse_script.execute(type="state", args=(w, None))
                target_state = target_script.execute(type="state", args=(w,))
                dot_prod = jnp.vdot(target_state, pulse_state)
                abs_diff += 1 - jnp.abs(dot_prod) ** 2  # one if no diff
                # phase_diff += jnp.abs(jnp.angle(dot_prod)) / jnp.pi  # zero if no diff

            abs_diff /= self.n_samples
            phase_diff /= self.n_samples

        n_nodes = len(pulse_scripts)
        return ((abs_diff + phase_diff) / 2) / n_nodes  # loss

    def run_optimization(
        self,
        cost,
        params,
        *args,
    ) -> tuple[jnp.ndarray, List]:
        """
        Run the optimization process.

        Args:
            cost (callable): Cost function to use for optimization.
            params (list or array): Initial parameters to use for
                the pulse-based gate.
            *args: Arguments to pass to the cost function.

        Returns:
            tuple[jnp.ndarray, List]: Optimized parameters and list of loss values
                at each iteration.
        """
        optimizer = optax.adamw(self.learning_rate)
        opt_state = optimizer.init(params)

        loss = cost(params, *args).item()
        loss_history = [loss]
        best_pulse_params = params

        @jax.jit
        def opt_step(params, opt_state, *args):
            loss, grads = jax.value_and_grad(cost)(params, *args)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for step in range(self.n_steps):
            if step % self.log_interval == 0:
                log.info(f"Step {step}/{self.n_steps}, Loss: {loss_history[-1]:.3e}")

            params, opt_state, loss = opt_step(params, opt_state, *args)

            if loss.item() < min(loss_history):
                log.debug(f"Best set of params found at step {step}")
                best_pulse_params = params

            loss_history.append(loss.item())

        return best_pulse_params, loss_history

    def optimize(self, simulator, wires):
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

                pulse_script = ys.QuantumScript(pulse_circuit, n_qubits=wires)
                target_script = ys.QuantumScript(target_circuit, n_qubits=wires)

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

                # Optimizing
                pulse_params, loss_history = self.run_optimization(
                    partial(
                        self.cost_fn,
                        pulse_script=pulse_script,
                        target_script=target_script,
                    ),
                    params=init_pulse_params,
                )

                self.save_results(
                    gate=gate_name,
                    fidelity=1 - min(loss_history),
                    pulse_params=pulse_params,
                )

                return pulse_params, loss_history

            return wrapper

        return decorator

    def optimize_multi_objective(self, simulator, wires):
        def decorator(create_circuits_array):
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
                    tuple: Optimized pulse parameters and list of
                        loss values at each iteration.
                """
                pulse_scripts = []
                target_scripts = []
                leafs = []
                for create_circuits in create_circuits_array:
                    pulse_circuit, target_circuit = create_circuits()

                    pulse_scripts.append(
                        ys.QuantumScript(pulse_circuit, n_qubits=wires)
                    )
                    target_scripts.append(
                        ys.QuantumScript(target_circuit, n_qubits=wires)
                    )

                    gate_name = create_circuits.__name__.split("_")[1]

                    leafs.append(PulseInformation.gate_by_name(gate_name).leafs)
                leafs = list(set(leafs))

                params = []
                for leaf in leafs:
                    params.extend(leaf.params)

                params = jnp.concatenate(params)

                # Optimizing
                pulse_params, loss_history = self.run_optimization(
                    partial(
                        self.multi_objective_cost_fn,
                        pulse_scripts=pulse_scripts,
                        target_scripts=target_scripts,
                        leafs=leafs,
                    ),
                    params=params,
                )

                idx = 0
                for leaf in leafs:
                    nidx = idx + leaf.size
                    # Saving the optimized parameters
                    self.save_results(
                        gate=leaf.name,
                        fidelity=1 - min(loss_history),
                        pulse_params=pulse_params[idx:nidx],
                    )
                    idx = nidx

                return pulse_params, loss_history

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
            return qml.state()

        return pulse_circuit, target_circuit

    def optimize_all(self, sel_gates, make_log):
        assert (
            self.observable == qml.state
        ), "Observable must be qml.state when doing optimization"

        log_history = {}
        optimize_1q = self.optimize("default.qubit", wires=1)
        optimize_2q = self.optimize("default.qubit", wires=2)

        # random_key = jax.random.key(seed=1000)
        # PulseInformation.shuffle_params(random_key)
        for loop in range(self.n_loops):
            log.info("Reading back optimized pulse parameters")
            # PulseInformation.update_params()

            log.info(f"Optimization loop {loop+1} of {self.n_loops}")

            if "RX" in sel_gates or "all" in sel_gates:
                log.info("Optimizing RX gate...")
                optimized_pulse_params, loss_history = optimize_1q(self.create_RX)()
                log.info(f"Optimized parameters for RX: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["RX"] = log_history.get("RX", []) + loss_history

            if "RY" in sel_gates or "all" in sel_gates:
                log.info("Optimizing RY gate...")
                optimized_pulse_params, loss_history = optimize_1q(self.create_RY)()
                log.info(f"Optimized parameters for RY: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["RY"] = log_history.get("RY", []) + loss_history

            if "RZ" in sel_gates or "all" in sel_gates:
                log.info("Optimizing RZ gate...")
                optimized_pulse_params, loss_history = optimize_1q(self.create_RZ)()
                log.info(f"Optimized parameters for RZ: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["RZ"] = log_history.get("RZ", []) + loss_history

            if "H" in sel_gates or "all" in sel_gates:
                log.info("Optimizing H gate...")
                optimized_pulse_params, loss_history = optimize_1q(self.create_H)()
                log.info(f"Optimized parameters for H: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["H"] = log_history.get("H", []) + loss_history

            if "Rot" in sel_gates or "all" in sel_gates:
                log.info("Optimizing Rot gate...")
                optimized_pulse_params, loss_history = optimize_1q(self.create_Rot)()
                log.info(f"Optimized parameters for Rot: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["Rot"] = log_history.get("Rot", []) + loss_history

            if "CX" in sel_gates or "all" in sel_gates:
                log.info("Optimizing CX gate...")
                optimized_pulse_params, loss_history = optimize_2q(self.create_CX)()
                log.info(f"Optimized parameters for CX: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["CX"] = log_history.get("CX", []) + loss_history

            if "CZ" in sel_gates or "all" in sel_gates:
                log.info("Optimizing CZ gate...")
                optimized_pulse_params, loss_history = optimize_2q(self.create_CZ)()
                log.info(f"Optimized parameters for CZ: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["CZ"] = log_history.get("CZ", []) + loss_history

            if "CY" in sel_gates or "all" in sel_gates:
                log.info("Optimizing CY gate...")
                optimized_pulse_params, loss_history = optimize_2q(self.create_CY)()
                log.info(f"Optimized parameters for CY: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["CY"] = log_history.get("CY", []) + loss_history

            if "CRX" in sel_gates or "all" in sel_gates:
                log.info("Optimizing CRX gate...")
                optimized_pulse_params, loss_history = optimize_2q(self.create_CRX)()
                log.info(f"Optimized parameters for CRX: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["CRX"] = log_history.get("CRX", []) + loss_history

            if "CRY" in sel_gates or "all" in sel_gates:
                log.info("Optimizing CRY gate...")
                optimized_pulse_params, loss_history = optimize_2q(self.create_CRY)()
                log.info(f"Optimized parameters for CRY: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["CRY"] = log_history.get("CRY", []) + loss_history

            if "CRZ" in sel_gates or "all" in sel_gates:
                log.info("Optimizing CRZ gate...")
                optimized_pulse_params, loss_history = optimize_2q(self.create_CRZ)()
                log.info(f"Optimized parameters for CRZ: {optimized_pulse_params}")
                log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.5f}%")
                log_history["CRZ"] = log_history.get("CRZ", []) + loss_history

            if make_log:
                # write log history to file
                with open("qml_essentials/qoc_logs.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(log_history.keys())
                    writer.writerows(zip(*log_history.values()))


if __name__ == "__main__":
    # argparse the selected gate
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", type=str, default=["RX", "RY", "RZ", "CZ"])
    parser.add_argument("--log", type=str, default=True)
    # TODO: add more arguments that take e.g. n_steps etc for initialization

    args = parser.parse_args()
    sel_gates = str(args.gates)
    make_log = bool(args.log)

    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    qoc = QOC(
        observable=qml.state,
    )

    qoc.optimize_all(sel_gates=sel_gates, make_log=make_log)
