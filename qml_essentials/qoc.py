import os
import csv
import jax
from jax import numpy as jnp
import optax
import pennylane as qml
from qml_essentials.ansaetze import Gates, PulseInformation
import matplotlib.pyplot as plt
import argparse
from functools import partial
from typing import List
import logging

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


class QOC:
    n_steps = 1000  # number of steps in optimization
    n_samples = 10  # number of parameter samples per step
    learning_rate = 0.01  # learning rate for adam with weight decay regularization
    log_interval = 100  # interval for logging
    skip_on_fidelity = True  # skip writing to qoc_results if fidelity is lower?

    # TODO: Potentially refactor all the create_*()... The only differences
    #   are the circuits
    def __init__(
        self,
        make_plots=False,
        file_dir="qoc/results",
        fig_dir="qoc/figures",
        fig_points=70,
    ):
        """
        Initialize Quantum Optimal Control with Pulse-level Gates.

        Args:
            log_dir (str): Directory for TensorBoard logs.
            make_plots (bool): Whether to generate and save plots.
            file_dir (str): Directory to save optimization results.
            fig_dir (str): Directory to save figures.
            fig_points (int): Number of points for plotting rotations.
        """
        self.ws = jnp.linspace(0, 2 * jnp.pi, fig_points)

        self.make_plots = make_plots
        self.file_dir = file_dir
        self.fig_dir = fig_dir

        self.current_gate = None

    def get_circuits(self):
        """
        Return pulse- and unitary-based circuits for the current gate.

        Returns:
            tuple: (pulse_circuit, target_circuit, operation_str)
        """
        dev = qml.device("default.qubit", wires=1)

        if self.current_gate in ["RX", "RY"]:

            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None):
                getattr(Gates, self.current_gate)(
                    w, 0, pulse_params=pulse_params, gate_mode="pulse"
                )
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0)),
                ]

            @qml.qnode(dev)
            def target_circuit(w):
                getattr(qml, self.current_gate)(w, wires=0)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0)),
                ]

            operation = f"{self.current_gate}(w)"

        elif self.current_gate == "RZ":

            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, *_):
                qml.RX(jnp.pi / 2, wires=0)
                getattr(Gates, self.current_gate)(w, 0, gate_mode="pulse")
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0)),
                ]

            @qml.qnode(dev)
            def target_circuit(w):
                qml.RX(jnp.pi / 2, wires=0)
                getattr(qml, self.current_gate)(w, wires=0)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0)),
                ]

            operation = f"RX(π / 2)·{self.current_gate}(w)"

        elif self.current_gate == "H":

            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None):
                qml.RX(w, wires=0)
                getattr(Gates, self.current_gate)(
                    0, pulse_params=pulse_params, gate_mode="pulse"
                )
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0)),
                ]

            @qml.qnode(dev)
            def target_circuit(w):
                qml.RX(w, wires=0)
                getattr(qml, self.current_gate)(wires=0)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0)),
                ]

            operation = f"RX(w)·{self.current_gate}"

        elif self.current_gate == "CZ":
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None):
                qml.RX(w, wires=0)
                qml.RX(w, wires=1)
                Gates.CZ([0, 1], pulse_params=pulse_params, gate_mode="pulse")
                qml.RX(-w, wires=1)
                qml.RX(-w, wires=0)
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1)),
                ]

            @qml.qnode(dev)
            def target_circuit(w):
                qml.RX(w, wires=0)
                qml.RX(w, wires=1)
                qml.CZ(wires=[0, 1])
                qml.RX(-w, wires=1)
                qml.RX(-w, wires=0)
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1)),
                ]

            operation = r"$RX_0(w)$·$RX_1(w)$·$CZ_{0, 1}$·$RX_1(-w)$·$RX_0(-w)$"

        elif self.current_gate == "CX":
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None):
                qml.RX(w, wires=0)
                Gates.CX([0, 1], pulse_params=pulse_params, gate_mode="pulse")
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1)),
                ]

            @qml.qnode(dev)
            def target_circuit(w):
                qml.RX(w, wires=0)
                qml.CNOT(wires=[0, 1])
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1)),
                ]

            operation = r"$RX_0(w)$·$CX_{0,1}$"

        return pulse_circuit, target_circuit, operation

    # TODO: Update method for new gates (Rot, CY, CRZ, CRY, CRX)
    def plot_rotation(self, pulse_params, pulse_qnode, target_qnode):
        """
        Plot expectation values of pulse- and unitary-based circuits for the
        current gate as a function of rotation angle.

        """
        # TODO: to make this functional, we have to change the measurement of the qnode
        # such that it matches those of `get_circuits`
        operation = ""

        pulse_expvals = [pulse_qnode(w, pulse_params) for w in self.ws]
        ideal_expvals = [target_qnode(w) for w in self.ws]

        pulse_expvals = jnp.array(pulse_expvals)
        ideal_expvals = jnp.array(ideal_expvals)

        fig, axs = plt.subplots(3, 1, figsize=(6, 12))

        bases = ["X", "Y", "Z"]
        for i, basis in enumerate(bases):
            axs[i].plot(self.ws, pulse_expvals[:, i], label="Pulse-based")
            axs[i].plot(self.ws, ideal_expvals[:, i], "--", label="Unitary-based")
            axs[i].set_xlabel("Rotation angle w (rad)")
            axs[i].set_ylabel(f"⟨{basis}⟩")
            axs[i].set_title(f"{operation} in {basis}-basis")
            axs[i].grid(True)
            axs[i].legend()

        xticks = [0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2, 2 * jnp.pi]
        xtick_labels = ["0", "π/2", "π", "3π/2", "2π"]
        for ax in axs:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)

        plt.tight_layout()
        os.makedirs(self.fig_dir, exist_ok=True)
        plt.savefig(f"{self.fig_dir}/qoc_{self.current_gate}(w).png")
        plt.close()

    def save_results(self, gate, fidelity, pulse_params):
        """
        Saves the optimized pulse parameters and fidelity for a given gate to a CSV file.

        Args:
            gate (str): Name of the gate.
            fidelity (float): Fidelity of the optimized pulse parameters.
            pulse_params (list): Optimized pulse parameters for the gate.

        Notes:
            If the gate already exists in the file and the newly optimized pulse parameters
            have a higher fidelity, the existing entry will be overwritten. If the fidelity is
            lower, the new entry will be skipped unless `skip_on_fidelity=False`.
        """
        if self.file_dir is not None:
            os.makedirs(self.file_dir, exist_ok=True)
            filename = os.path.join(self.file_dir, "qoc_results.csv")

            if os.path.isfile(filename):
                with open(filename, mode="r", newline="") as f:
                    reader = csv.reader(f.readlines())

            entry = [gate] + [fidelity] + list(map(float, pulse_params))

            with open(filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                match = False
                for row in reader:
                    # gate already exists
                    if row[0] == gate:
                        if fidelity > float(row[1]):
                            writer.writerow(entry)
                        else:
                            log.warning(
                                f"Pulse parameters for {gate} already exist with \
                                    higher fidelity ({row[1]} >= {fidelity})"
                            )
                            if not self.skip_on_fidelity:
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

    def cost_fn(self, pulse_params, pulse_qnode, target_qnode) -> float:
        """
        Cost function for QOC optimization.

        The cost function is calculated as the average of the fidelity and
        phase difference between the pulse-based and unitary-based gates.

        Args:
            pulse_params (list or array): Optimized parameters to use for the pulse-based gate.
            pulse_qnode (callable): Pulse-based gate qnode.
            target_qnode (callable): Unitary-based gate qnode.

        Returns:
            float: Cost function value.
        """
        fidelity = 0
        phase_diff = 0
        for w in jnp.arange(0, 2 * jnp.pi, (2 * jnp.pi) / self.n_samples):
            pulse_state = pulse_qnode(w, pulse_params)
            target_state = target_qnode(w)
            fidelity += (
                jnp.abs(jnp.vdot(target_state, pulse_state)) ** 2
            )  # one if no diff
            phase_diff += jnp.abs(jnp.angle(jnp.vdot(target_state, pulse_state))) / (
                jnp.pi
            )  # zero if no diff

        fidelity_n = 1 - (fidelity / self.n_samples)
        phase_diff = phase_diff / self.n_samples

        return (fidelity_n + phase_diff) / 2  # loss

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
            params (list or array): Initial parameters to use for the pulse-based gate.
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
            loss_history.append(loss.item())

            if loss.item() < min(loss_history):
                best_pulse_params = params

        return best_pulse_params, loss_history

    def optimize(simulator, wires):
        def decorator(create_circuits):
            def wrapper(self, init_pulse_params):
                """
                This function is a wrapper for the create_circuits method.
                It takes a simulator and wires as input and optimizes the pulse parameters
                  using the cost function defined in the QOC class.

                Args:
                    create_circuits (callable): A function to generate the pulse and
                        target circuits for the gate.
                    init_pulse_params (array): Initial pulse parameters to use for
                        the pulse-based gate.

                Returns:
                    tuple: Optimized pulse parameters and list of loss values at each iteration.
                """
                dev = qml.device(simulator, wires=wires)
                pulse_circuit, target_circuit = create_circuits(self, dev)

                pulse_qnode = qml.QNode(pulse_circuit, dev, interface="jax")
                target_qnode = qml.QNode(target_circuit, dev, interface="jax")

                # Optimizing
                pulse_params, loss_history = self.run_optimization(
                    partial(
                        self.cost_fn,
                        pulse_qnode=pulse_qnode,
                        target_qnode=target_qnode,
                    ),
                    params=init_pulse_params,
                )

                gate_name = create_circuits.__name__.split("_")[1]
                # Saving the optimized parameters
                self.save_results(
                    gate=gate_name,
                    fidelity=1 - min(loss_history),
                    pulse_params=pulse_params,
                )

                if self.make_plots:
                    self.plot_rotation(pulse_params, pulse_qnode, target_qnode)

                return pulse_params, loss_history

            return wrapper

        return decorator

    # def create_Rot(
    #     self,
    #
    #
    #     phi: float = jnp.pi / 2,
    #     theta: float = jnp.pi / 2,
    #     omega: float = jnp.pi / 2,
    #     init_pulse_params: jnp.array = jnp.array([0.5, 1.0, 15.0, 1.0, 0.5]),
    #
    # ):
    #     """
    #     Optimize pulse parameters for the Rot(theta, phi, lam) gate.

    #     Uses gradient-based optimization to minimize the difference between the
    #     pulse-based Rot(phi, theta, omega) circuit expectation value and the target
    #     unitary-based Rot(phi, theta, omega).

    #     Args:
    #         steps (int): Number of optimization steps. Default: 1000.
    #         patience (int): Patience for early stopping. Default: 100.
    #         theta, phi, lam (float): Rotation angles for the Rot gate.
    #             Default: π / 2 for all three.
    #         init_pulse_params (jnp.ndarray): Initial pulse parameters.
    #             Default: [0.5, 1.0, 15.0, 1.0, 0.5].
    #         log_interval (int): Frequency of printing loss.

    #     Returns:
    #         tuple: Optimized parameters and list of loss values.
    #     """
    #     self.current_gate = "Rot"
    #     w = (phi, theta, omega)

    #     dev = qml.device("default.qubit", wires=1)

    #     @qml.qnode(dev, interface="jax")
    #     def pulse_circuit(w, pulse_params):
    #         phi, theta, omega = w
    #         Gates.Rot(
    #             phi, theta, omega, 0, pulse_params=pulse_params, gate_mode="pulse"
    #         )
    #         return qml.state()

    #     @qml.qnode(dev)
    #     def target_circuit(w):
    #         phi, theta, omega = w
    #         qml.Rot(phi, theta, omega, wires=0)
    #         return qml.state()

    #     # Optimizing
    #     pulse_params, loss, losses = self.run_optimization(
    #         partial(
    #             self.cost_fn, pulse_circuit=pulse_circuit, target_circuit=target_circuit
    #         ),
    #         init_pulse_params=init_pulse_params,
    #         steps=steps,
    #
    #         log_interval=log_interval,
    #     )

    #     # Saving the optimized parameters
    #     self.save_results(pulse_params)

    #     # Plotting the rotation
    #     if self.make_plots:
    #         warnings.warn("Plotting not implemented yet", UserWarning)

    #     return pulse_params, loss, losses

    @optimize("default.qubit", wires=1)
    def create_RX(self, init_pulse_params):
        def pulse_circuit(w, pulse_params):
            Gates.RX(w, 0, pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.RX(w, wires=0)
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=1)
    def create_RY(
        self,
        init_pulse_params: jnp.array,
    ):
        def pulse_circuit(w, pulse_params):
            Gates.RY(w, 0, pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.RY(w, wires=0)
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=1)
    def create_RZ(self, init_pulse_params):
        return None, None

    @optimize("default.qubit", wires=1)
    def create_H(self, init_pulse_params: jnp.array):
        def pulse_circuit(w, pulse_params):
            Gates.H(0, pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.H(wires=0)
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=2)
    def create_CX(self, init_pulse_params: jnp.ndarray):
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            qml.RY(w, wires=1)
            Gates.CX(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.H(wires=0)
            qml.RX(w, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=2)
    def create_CY(self, init_pulse_params: jnp.ndarray):
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            qml.RY(w, wires=1)
            Gates.CY(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.H(wires=0)
            qml.RX(w, wires=1)
            qml.CY(wires=[0, 1])
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=2)
    def create_CZ(self, init_pulse_params: jnp.array):
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            qml.RX(w, wires=1)
            Gates.CZ(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.H(wires=0)
            qml.RX(w, wires=1)
            qml.CZ(wires=[0, 1])
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=2)
    def create_CRX(self, init_pulse_params: jnp.ndarray):
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            Gates.CRX(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.H(wires=0)
            qml.CRX(w, wires=[0, 1])
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=2)
    def create_CRY(self, init_pulse_params: jnp.ndarray):
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            Gates.CRY(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.H(wires=0)
            qml.CRY(w, wires=[0, 1])
            return qml.state()

        return pulse_circuit, target_circuit

    @optimize("default.qubit", wires=2)
    def create_CRZ(self, init_pulse_params: jnp.ndarray):
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            qml.H(wires=1)
            Gates.CRZ(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        def target_circuit(w):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.CRZ(w, wires=[0, 1])
            return qml.state()

        return pulse_circuit, target_circuit


if __name__ == "__main__":
    # argparse the selected gate
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=str, default="all")
    parser.add_argument("--loops", type=str, default=1)
    parser.add_argument("--log", type=str, default=True)

    args = parser.parse_args()
    gate = str(args.gate)
    loops = int(args.loops)
    make_log = bool(args.log)

    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler())

    qoc = QOC(
        make_plots=False,
        fig_points=40,
        fig_dir="docs/figures",
        file_dir="qml_essentials",
    )

    log_history = {}

    for loop in range(loops):
        log.info(f"Reading back optimized pulse parameters")
        PulseInformation.update_params()

        log.info(f"Optimization loop {loop+1} of {loops}")

        if gate == "RX" or gate == "all":
            log.info("Optimizing RX gate...")
            optimized_pulse_params, loss_history = qoc.create_RX(
                init_pulse_params=jnp.array(
                    [15.70989327341467, 29.5230665326707, 0.7499810441330634]
                ),
            )
            log.info(f"Optimized parameters for RX: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")
            log_history["RX"] = log_history.get("RX", []) + loss_history

        if gate == "RY" or gate == "all":
            log.info("Optimizing RY gate...")
            optimized_pulse_params, loss_history = qoc.create_RY(
                init_pulse_params=jnp.array(
                    [7.8787724942614235, 22.001319411513432, 1.098524473819202]
                ),
            )
            log.info(f"Optimized parameters for RY: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {1 - min(loss_history):.6f}")
            log_history["RY"] = log_history.get("RY", []) + loss_history

        # # if gate == "RZ" or gate == "all":
        # #     log.info("Plotting RZ gate rotation...")
        # #     qoc.create_RZ(None)
        # #     log.info("Plotted RZ gate rotation")

        if gate == "H" or gate == "all":
            log.info("Optimizing H gate...")
            optimized_pulse_params, loss_history = qoc.create_H(
                init_pulse_params=jnp.array(
                    [7.857992398977854, 21.572701026008765, 0.9000668764548863]
                )
            )
            log.info(f"Optimized parameters for H: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")

        if gate == "CX" or gate == "all":
            log.info("Optimizing CX gate...")
            optimized_pulse_params, loss_history = qoc.create_CX(
                init_pulse_params=jnp.array(
                    [
                        *PulseInformation.optimized_params("H"),
                        *PulseInformation.optimized_params("CZ"),
                        *PulseInformation.optimized_params("H"),
                    ]
                )
            )
            log.info(f"Optimized parameters for CX: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")

        if gate == "CZ" or gate == "all":
            log.info("Optimizing CZ gate...")
            optimized_pulse_params, loss_history = qoc.create_CZ(
                init_pulse_params=jnp.array([0.962596375687258])
            )
            log.info(f"Optimized parameters for CZ: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")

        if gate == "CY" or gate == "all":
            log.info("Optimizing CY gate...")
            optimized_pulse_params, loss_history = qoc.create_CY(
                init_pulse_params=jnp.array(
                    [
                        *PulseInformation.optimized_params("RZ"),
                        *PulseInformation.optimized_params("CX"),
                        *PulseInformation.optimized_params("RZ"),
                    ]
                ),
            )
            log.info(f"Optimized parameters for CY: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")

        if gate == "CRX" or gate == "all":
            log.info("Optimizing CRX gate...")
            optimized_pulse_params, loss_history = qoc.create_CRX(
                init_pulse_params=jnp.array(
                    [
                        *PulseInformation.optimized_params("RZ"),
                        *PulseInformation.optimized_params("RY"),
                        *PulseInformation.optimized_params("CX"),
                        *PulseInformation.optimized_params("RY"),
                        *PulseInformation.optimized_params("CX"),
                        *PulseInformation.optimized_params("RZ"),
                    ]
                ),
            )
            log.info(f"Optimized parameters for CRX: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")

        if gate == "CRY" or gate == "all":
            log.info("Optimizing CRY gate...")
            optimized_pulse_params, loss_history = qoc.create_CRY(
                init_pulse_params=jnp.array(
                    [
                        *PulseInformation.optimized_params("RY"),
                        *PulseInformation.optimized_params("CX"),
                        *PulseInformation.optimized_params("RX"),
                        *PulseInformation.optimized_params("CX"),
                    ]
                ),
            )
            log.info(f"Optimized parameters for CRY: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")

        if gate == "CRZ" or gate == "all":
            log.info("Optimizing CRZ gate...")
            optimized_pulse_params, loss_history = qoc.create_CRZ(
                init_pulse_params=jnp.array(
                    [
                        *PulseInformation.optimized_params("RZ"),
                        *PulseInformation.optimized_params("CX"),
                        *PulseInformation.optimized_params("RZ"),
                        *PulseInformation.optimized_params("CX"),
                    ]
                )
            )
            log.info(f"Optimized parameters for CRZ: {optimized_pulse_params}")
            log.info(f"Best achieved fidelity: {(1 - min(loss_history))*100:.3f}%")

        if make_log:
            # write log history to file
            with open("qml_essentials/qoc_results_log.csv", "w") as f:
                writer = csv.writer(f)
                # use keys in log_history as cols and values as rows such that each step is a new row
                writer.writerow(log_history.keys())
                writer.writerows(zip(*log_history.values()))
