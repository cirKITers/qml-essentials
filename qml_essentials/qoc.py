import os
import sys
import csv
import jax
from jax import numpy as jnp
import optax
import pennylane as qml
from qml_essentials.ansaetze import Gates, PulseInformation
import matplotlib.pyplot as plt
import warnings
import argparse
import logging

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


class QOC:
    # TODO: Potentially refactor all the optimize_*()... The only differences
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
            tuple: (pulse_circuit, unitary_circuit, operation_str)
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
            def unitary_circuit(w):
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
            def unitary_circuit(w):
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
            def unitary_circuit(w):
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
            def unitary_circuit(w):
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
            def unitary_circuit(w):
                qml.RX(w, wires=0)
                qml.CNOT(wires=[0, 1])
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1)),
                ]

            operation = r"$RX_0(w)$·$CX_{0,1}$"

        return pulse_circuit, unitary_circuit, operation

    # TODO: Update method for new gates (Rot, CY, CRZ, CRY, CRX)
    def plot_rotation(self, pulse_params):
        """
        Plot expectation values of pulse- and unitary-based circuits for the
        current gate as a function of rotation angle.

        Args:
            pulse_params: pulse parameters of pulse level gate.
        """
        pulse_circuit, unitary_circuit, operation = self.get_circuits()

        pulse_expvals = [pulse_circuit(w, pulse_params) for w in self.ws]
        ideal_expvals = [unitary_circuit(w) for w in self.ws]

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

    def save_results(self, opt_pulse_params):
        """
        Save optimized pulse parameters to CSV file.

        Args:
            opt_pulse_params (list or array): Optimized parameters to save.
            filename (str): Path to CSV file.
        """
        if self.file_dir is not None:
            os.makedirs(self.file_dir, exist_ok=True)
            filename = os.path.join(self.file_dir, "qoc_results.csv")

            if os.path.isfile(filename):
                with open(filename, mode="r", newline="") as f:
                    reader = csv.reader(f.readlines())

            with open(filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                match = False
                for row in reader:
                    if row[0] == self.current_gate:
                        writer.writerow(
                            [self.current_gate] + list(map(float, opt_pulse_params))
                        )
                        match = True
                    else:
                        writer.writerow(row)
                if not match:
                    writer.writerow(
                        [self.current_gate] + list(map(float, opt_pulse_params))
                    )

    def loss_fn(self, state, target_state):
        """
        Compute infidelity between two quantum states.

        Args:
            state (array): Output state from pulse circuit.
            target_state (array): Target state from unitary circuit.

        Returns:
            float: Infidelity (1 - fidelity).
        """
        fidelity = jnp.abs(jnp.vdot(target_state, state)) ** 2
        return 1 - fidelity

    def cost_fn(self, pulse_params, circuit, w, target_state):
        """
        Compute cost for optimization by evaluating circuit and loss.

        Args:
            pulse_params: pulse parameters of pulse level gate.
            circuit (callable): QNode circuit accepting (w, pulse_params).
            w (float): Rotation angle.
            target_state (array): Target quantum state.

        Returns:
            float: Computed loss.
        """
        state = circuit(w, pulse_params)
        return self.loss_fn(state, target_state)

    def run_optimization(
        self,
        cost,
        init_pulse_params,
        steps,
        patience,
        log_interval,
        *args,
    ):
        """
        Run gradient-based optimization on given cost function.

        Args:
            cost (callable): Cost function to minimize.
            init_pulse_params (array): Initial parameters.
            steps (int): Number of optimization steps.
            log_interval (int): Print frequency.
            *args: Extra args for cost.
            patience (int): Early stopping patience (default: 20).

        Returns:
            tuple: (optimized parameters, best loss, list of loss values)
        """
        optimizer = optax.adamw(0.1)
        opt_state = optimizer.init(init_pulse_params)
        pulse_params = init_pulse_params
        losses = []

        best_loss = float("inf")
        best_pulse_params = pulse_params
        no_improve_counter = 0

        @jax.jit
        def opt_step(params, opt_state, *args):
            loss, grads = jax.value_and_grad(cost)(params, *args)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for step in range(steps):
            pulse_params, opt_state, loss = opt_step(pulse_params, opt_state, *args)
            losses.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_pulse_params = pulse_params
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if (step + 1) % log_interval == 0:
                log.info(f"Step {step + 1}/{steps}, Loss: {loss:.2e}")

            if no_improve_counter >= patience:
                log.info(f"Early stopping at step {step + 1} due to no improvement.")
                break

        return best_pulse_params, best_loss, losses

    def optimize_Rot(
        self,
        steps: int = 1000,
        patience: int = 100,
        phi: float = jnp.pi / 2,
        theta: float = jnp.pi / 2,
        omega: float = jnp.pi / 2,
        init_pulse_params: jnp.array = jnp.array([0.5, 1.0, 15.0, 1.0, 0.5]),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the Rot(theta, phi, lam) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based Rot(phi, theta, omega) circuit expectation value and the target
        unitary-based Rot(phi, theta, omega).

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Patience for early stopping. Default: 100.
            theta, phi, lam (float): Rotation angles for the Rot gate.
                Default: π / 2 for all three.
            init_pulse_params (jnp.ndarray): Initial pulse parameters.
                Default: [0.5, 1.0, 15.0, 1.0, 0.5].
            log_interval (int): Frequency of printing loss.

        Returns:
            tuple: Optimized parameters and list of loss values.
        """
        self.current_gate = "Rot"
        w = (phi, theta, omega)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            phi, theta, omega = w
            Gates.Rot(
                phi, theta, omega, 0, pulse_params=pulse_params, gate_mode="pulse"
            )
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit(w):
            phi, theta, omega = w
            qml.Rot(phi, theta, omega, wires=0)
            return qml.state()

        target = unitary_circuit(w)

        # Optimizing
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=w, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        # Saving the optimized parameters
        self.save_results(pulse_params)

        # Plotting the rotation
        if self.make_plots:
            warnings.warn("Plotting not implemented yet", UserWarning)

        return pulse_params, loss, losses

    def optimize_RX(
        self,
        steps: int = 1000,
        patience: int = 100,
        w: float = jnp.pi,
        init_pulse_params: jnp.array = jnp.array([1.0, 15.0, 1.0]),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the RX(w) gate to best approximate
        the unitary RX(w) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based RX(w) circuit expectation value and the target gate-based RX(w).

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            w (float): Rotation angle in radians with which to run the optimization.
               Default: π.
            init_pulse_params (jnp.ndarray): Initial pulse parameters (A, sigma) and
                time. Default: [1.0, 15.0, 1.0].
            log_interval (int): Frequency of printing loss during optimization.
               Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values
                during optimization.
        """
        self.current_gate = "RX"

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            Gates.RX(w, 0, pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit(w):
            qml.RX(w, wires=0)
            return qml.state()

        target = unitary_circuit(w)

        # Optimizing
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=w, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        # Saving the optimized parameters
        self.save_results(pulse_params)

        # Plotting the RX rotation
        if self.make_plots:
            log.info("Plotting RX rotation...")
            self.plot_rotation(pulse_params)

        return pulse_params, loss, losses

    def optimize_RY(
        self,
        steps: int = 1000,
        patience: int = 100,
        w: float = jnp.pi,
        init_pulse_params: jnp.array = jnp.array([1.0, 15.0, 1.0]),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the RY(w) gate to best approximate
        the unitary RY(w) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based RY(w) circuit expectation value and the target unitary-based RY(w).

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            w (float): Rotation angle in radians with which to run the optimization.
                Default: π.
            init_pulse_params (jnp.ndarray): Initial pulse parameters (A, sigma) and
                time. Default: [1.0, 15.0, 1.0].
            log_interval (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values
                during optimization.
        """
        self.current_gate = "RY"

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            Gates.RY(w, 0, pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit(w):
            qml.RY(w, wires=0)
            return qml.state()

        target = unitary_circuit(w)

        # Optimizing
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=w, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        # Saving the optimized parameters
        self.save_results(pulse_params)

        # Plotting the RY rotation
        if self.make_plots:
            log.info("Plotting RY rotation...")
            self.plot_rotation(pulse_params)

        return pulse_params, loss, losses

    def optimize_RZ(self):
        """
        Plot the pulse level RZ rotation on the X basis.

        Note:
            No actual optimization is performed since the RZ gate
            does not have pulse parameters to optimize.

        Returns:
            tuple: (None, None)
        """
        self.current_gate = "RZ"
        if self.make_plots:
            log.info("Plotting RZ rotation...")
            self.plot_rotation([])

        return None, None

    def optimize_H(
        self,
        steps=1000,
        patience: int = 100,
        init_pulse_params: jnp.array = jnp.array([1.0, 15.0, 1.0]),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the Hadamard (H) gate to best approximate
        the unitary H gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H circuit output state and the target gate-based H state.

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            init_pulse_params (jnp.ndarray): Initial pulse parameters (A, sigma, t)
                Default: [1.0, 15.0, 1.0].
            log_interval (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray), best loss (float), and
                list of loss values during optimization.
        """
        self.current_gate = "H"

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            Gates.H(0, pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit():
            qml.H(wires=0)
            return qml.state()

        target = unitary_circuit()

        # Optimizing
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=None, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        # Saving the optimized parameters
        self.save_results(pulse_params)

        # Plotting the RX rotation
        if self.make_plots:
            log.info("Plotting H rotation...")
            self.plot_rotation(pulse_params)

        return pulse_params, loss, losses

    def optimize_CZ(
        self,
        steps=1000,
        patience: int = 100,
        init_pulse_params: jnp.ndarray = jnp.array([1.0]),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the CZ gate to best approximate
        the unitary CZ gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H_c · H_t · CZ circuit expectation value and the target
        unitary-based H_c · H_t · CZ.

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            init_pulse_params (jnp.ndarray): Initial pulse duration. Default: [1.0].
            log_interval (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values
                during optimization.
        """
        self.current_gate = "CZ"

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            qml.H(wires=1)
            Gates.CZ(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit():
            qml.H(wires=0)
            qml.H(wires=1)
            qml.CZ(wires=[0, 1])
            return qml.state()

        target = unitary_circuit()

        # Optimizing
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=None, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        # Saving the optimized parameters
        self.save_results(pulse_params)

        # Plotting the CZ rotation
        if self.make_plots:
            log.info("Plotting CZ rotation...")
            self.plot_rotation(pulse_params)

        return pulse_params, loss, losses

    def optimize_CY(
        self,
        steps=1000,
        patience: int = 100,
        init_pulse_params: jnp.ndarray = jnp.array(
            [0.5, 15.0, 10.0, 1.0, 15.0, 10.0, 1.0, 1.0, 0.5]
        ),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the CY gate to best approximate the
        unitary CY gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H_c · CY circuit expectation value and the target
        unitary-based H_c · CY.

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            init_pulse_params (jnp.ndarray): Initial pulse parameters.
                Default: [1.0, 15.0, 1.0, 1.0, 1.0, 15.0, 1.0].
            log_interval (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values during
                optimization.
        """
        self.current_gate = "CY"

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            Gates.CY(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit():
            qml.H(wires=0)
            qml.CY(wires=[0, 1])
            return qml.state()

        target = unitary_circuit()

        # Optimizing
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=None, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        # Saving the optimized parameters
        self.save_results(pulse_params)

        # Plotting the CY rotation
        if self.make_plots:
            warnings.warn("Plotting not implemented yet", UserWarning)

        return pulse_params, loss, losses

    def optimize_CX(
        self,
        steps=1000,
        patience: int = 100,
        init_pulse_params: jnp.ndarray = jnp.array(
            [1.0, 15.0, 1.0, 1.0, 1.0, 15.0, 1.0]
        ),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the CX gate to best approximate the
        unitary CX gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H_c · CX circuit expectation value and the target
        unitary-based H_c · CX.

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            init_pulse_params (jnp.ndarray): Initial pulse parameters.
                Default: [1.0, 15.0, 1.0, 1.0, 1.0, 15.0, 1.0].
            log_interval (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values during
                optimization.
        """
        self.current_gate = "CX"

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            Gates.CX(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit():
            qml.H(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        target = unitary_circuit()

        # Optimizing
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=None, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        # Saving the optimized parameters
        self.save_results(pulse_params)

        # Plotting the CX rotation
        if self.make_plots:
            log.info("Plotting CX rotation...")
            self.plot_rotation(pulse_params)

        return pulse_params, loss, losses

    def optimize_CRX(
        self,
        steps=1000,
        patience: int = 100,
        w: float = jnp.pi,
        init_pulse_params: jnp.ndarray = jnp.array(
            [10.0, 15.0, 1.0, 0.5, 1.0, 0.5, 10.0, 15.0, 1.0]
        ),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the CRX(w) gate to best approximate
        the unitary CRX(w) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H_c · CRX(w) circuit expectation value and the target
        unitary-based H_c · CRX(w).

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            w (float): Rotation angle.
            init_pulse_params (jnp.ndarray): Initial pulse parameters.
            log_interval (int): Frequency of printing loss.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values.
        """
        self.current_gate = "CRX"

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            Gates.CRX(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit(w):
            qml.H(wires=0)
            qml.CRX(w, wires=[0, 1])
            return qml.state()

        target = unitary_circuit(w)

        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=w, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        self.save_results(pulse_params)

        if self.make_plots:
            warnings.warn("Plotting not implemented yet", UserWarning)

        return pulse_params, loss, losses

    def optimize_CRY(
        self,
        steps=1000,
        patience: int = 100,
        w: float = jnp.pi,
        init_pulse_params: jnp.ndarray = jnp.array(
            [10.0, 15.0, 1.0, 0.5, 1.0, 0.5, 10.0, 15.0, 1.0]
        ),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the CRY(w) gate to best approximate
        the unitary CRY(w) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H_c · CRY(w) circuit expectation value and the target
        unitary-based H_c · CRY(w).

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Amount of epochs without improvement before early stopping.
                Default: 100.
            w (float): Rotation angle.
            init_pulse_params (jnp.ndarray): Initial pulse parameters.
            log_interval (int): Frequency of printing loss.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values.
        """
        self.current_gate = "CRY"

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            Gates.CRY(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit(w):
            qml.H(wires=0)
            qml.CRY(w, wires=[0, 1])
            return qml.state()

        target = unitary_circuit(w)

        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=w, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        self.save_results(pulse_params)

        if self.make_plots:
            warnings.warn("Plotting not implemented yet", UserWarning)

        return pulse_params, loss, losses

    def optimize_CRZ(
        self,
        steps=1000,
        patience: int = 100,
        w: float = jnp.pi,
        init_pulse_params: jnp.ndarray = jnp.array([0.5, 2.0, 0.5]),
        log_interval: int = 50,
    ):
        """
        Optimize pulse parameters for the CRZ(w) gate to best approximate
        the unitary CRZ(w) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H_c · H_t · CRZ(w) circuit expectation value and the target
        unitary-based H_c · H_t · CRZ(w).

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            patience (int): Early stopping patience. Default: 100.
            w (float): Rotation angle.
            init_pulse_params (jnp.ndarray): Initial pulse parameters.
            log_interval (int): Frequency of printing loss.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values.
        """
        self.current_gate = "CRZ"

        dev = qml.device("default.qubit", wires=2)

        # Pulse circuit with full parametric decomposition
        @qml.qnode(dev, interface="jax")
        def pulse_circuit(w, pulse_params):
            qml.H(wires=0)
            qml.H(wires=1)
            Gates.CRZ(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
            return qml.state()

        @qml.qnode(dev)
        def unitary_circuit(w):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.CRZ(w, wires=[0, 1])
            return qml.state()

        target = unitary_circuit(w)

        # Cost function
        def cost(pulse_params):
            return self.cost_fn(
                pulse_params, circuit=pulse_circuit, w=w, target_state=target
            )

        pulse_params, loss, losses = self.run_optimization(
            cost, init_pulse_params, steps, patience, log_interval
        )

        self.save_results(pulse_params)

        if self.make_plots:
            warnings.warn("Plotting not implemented yet", UserWarning)

        return pulse_params, loss, losses


if __name__ == "__main__":
    # argparse the selected gate
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=str, default="all")
    args = parser.parse_args()
    gate = args.gate

    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler())

    qoc = QOC(
        make_plots=False,
        fig_points=40,
        fig_dir="docs/figures",
        file_dir="qml_essentials",
    )

    if gate == "Rot" or gate == "all":
        log.info("Optimizing Rot gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_Rot(
            init_pulse_params=jnp.array(
                [0.5, 7.857992399021039, 21.57270102638842, 0.9000668764608991, 0.5]
            )
        )
        log.info(f"Optimized parameters for Rot: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "RX" or gate == "all":
        log.info("Optimizing RX gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_RX(
            w=jnp.pi,
            init_pulse_params=jnp.array(
                [15.70989327341467, 29.5230665326707, 0.7499810441330634]
            ),
        )
        log.info(f"Optimized parameters for RX: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "RY" or gate == "all":
        log.info("Optimizing RY gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_RY(
            w=jnp.pi,
            init_pulse_params=jnp.array(
                [7.8787724942614235, 22.001319411513432, 1.098524473819202]
            ),
        )
        log.info(f"Optimized parameters for RY: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss:.6f}")

    if gate == "RZ" or gate == "all":
        log.info("Plotting RZ gate rotation...")
        qoc.optimize_RZ()
        log.info("Plotted RZ gate rotation")

    if gate == "H" or gate == "all":
        log.info("Optimizing H gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_H(
            init_pulse_params=jnp.array(
                [7.857992398977854, 21.572701026008765, 0.9000668764548863]
            )
        )
        log.info(f"Optimized parameters for H: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "CZ" or gate == "all":
        log.info("Optimizing CZ gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_CZ(
            init_pulse_params=jnp.array([0.962596375687258]), log_interval=5
        )
        log.info(f"Optimized parameters for CZ: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "CY" or gate == "all":
        log.info("Optimizing CY gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_CY(
            init_pulse_params=jnp.array(
                [
                    0.5,
                    13.679990291069169,
                    6.86497650976022,
                    1.05475551194351,
                    14.96056469588421,
                    13.040583781891456,
                    0.33844677502596704,
                    0.8709563476069772,
                    0.5,
                ]
            ),
        )
        log.info(f"Optimized parameters for CY: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "CX" or gate == "all":
        log.info("Optimizing CX gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_CX(
            init_pulse_params=jnp.array(
                [
                    7.951920934692106,
                    21.655479574101687,
                    0.8929524493211076,
                    0.9548359253748596,
                    7.94488020182026,
                    21.61729834699293,
                    0.9067943033364354,
                ]
            )
        )
        log.info(f"Optimized parameters for CX: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "CRX" or gate == "all":
        log.info("Optimizing CRX gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_CRX(
            w=jnp.pi,
            init_pulse_params=jnp.array(
                [
                    0.0869898677627019,
                    6.317621899953034,
                    20.912822341258522,
                    1.9221452238507606,
                    8.238376987907568,
                    20.613174536740054,
                    0.8737746698069836,
                    1.3952917467811454,
                    7.486811620275413,
                    21.98771445546622,
                    0.610849203728594,
                    7.276467778265519,
                    20.089001022776028,
                    1.0953393377060914,
                    7.260820896041849,
                    19.914716438316788,
                    0.3965192075282136,
                    0.9260646756057936,
                    8.559496874753883,
                    21.502993748426707,
                    1.1125869787900438,
                    0.5326923136891571,
                ]
            ),
        )
        log.info(f"Optimized parameters for CRX: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "CRY" or gate == "all":
        log.info("Optimizing CRY gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_CRY(
            w=jnp.pi,
            init_pulse_params=jnp.array(
                [
                    6.338251784241546,
                    22.218883551618084,
                    0.9884981881682461,
                    8.660014835148415,
                    20.952862650089553,
                    0.6642884429132376,
                    1.2732148054025483,
                    6.552248449325667,
                    20.533588403615788,
                    1.1057700409147055,
                    5.480401029932529,
                    19.84968803100232,
                    1.2999867767029427,
                    8.402279591821387,
                    20.623906763446577,
                    1.3050181945303863,
                    0.9549534431854169,
                    8.904512396279161,
                    19.793377558836266,
                    1.1223124670193627,
                ]
            ),
        )
        log.info(f"Optimized parameters for CRY: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    if gate == "CRZ" or gate == "all":
        log.info("Optimizing CRZ gate...")
        optimized_pulse_params, best_loss, loss_values = qoc.optimize_CRZ(
            init_pulse_params=jnp.array(
                [
                    0.5226296692020662,
                    6.495223824775022,
                    21.789528537935603,
                    0.7877065607968377,
                    1.5600575401823529,
                    8.461005039055976,
                    22.000950235408457,
                    0.6954321175337905,
                    0.4985574282655742,
                    6.734342354364616,
                    20.932677593123522,
                    1.0966905643848288,
                    0.9819214824461874,
                    7.459675656997471,
                    19.558641924041503,
                    1.285349794196732,
                ]
            )
        )
        log.info(f"Optimized parameters for CRZ: {optimized_pulse_params}\n")
        log.info(f"Best achieved fidelity: {1 - best_loss}")

    log.info("Loading back results")
    PulseInformation.update_params()
