# flake8: noqa: E402
# flake8: noqa: E731
import os
os.environ["JAX_ENABLE_X64"] = "1"
import csv
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from ansaetze import PulseGates
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

# TODO: Put figures in docs/figures/qoc_<gate_name>.png

class QOC:
    def __init__(
        self,
        log_dir="tensorboard/qoc",
        make_plots=False,
        fig_dir="qml_essentials/figures",
        fig_points=70
    ):
        """
        Initialize Quantum Optimal Control with Pulse-level Gates.

        Args:
            log_dir (str): Directory for TensorBoard logs.
            make_plots (bool): Whether to generate and save plots.
            fig_dir (str): Directory to save figures.
            fig_points (int): Number of points for plotting rotations.
        """
        ps = PulseGates()
        self.RX = ps.RX
        self.RY = ps.RY
        self.RZ = ps.RZ
        self.H = ps.H
        self.CZ = ps.CZ
        self.CNOT = ps.CNOT

        self.ws = jnp.linspace(0, 2 * jnp.pi, fig_points)

        # self.writer = SummaryWriter(log_dir=log_dir)
        self.writer = None
        self.make_plots = make_plots
        self.fig_dir = fig_dir
        if make_plots:
            os.makedirs(fig_dir, exist_ok=True)

        self.current_gate = None

    def get_circuits(self):
        """
        Return pulse-based and ideal circuits for the current gate.

        Returns:
            tuple: (pulse_circuit, ideal_circuit, operation_str)
        """
        dev = qml.device("default.qubit", wires=1)

        if self.current_gate in ["RX", "RY"]:
            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None, t=None):
                getattr(self, self.current_gate)(w, 0, pulse_params, t)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0))
                ]

            @qml.qnode(dev)
            def ideal_circuit(w):
                getattr(qml, self.current_gate)(w, wires=0)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0))
                ]

            operation = f"{self.current_gate}(w)"

        elif self.current_gate == "RZ":
            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, *_):
                qml.RX(jnp.pi / 2, 0)
                getattr(self, self.current_gate)(w, 0)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0))
                ]

            @qml.qnode(dev)
            def ideal_circuit(w):
                qml.RX(jnp.pi / 2, 0)
                getattr(qml, self.current_gate)(w, wires=0)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0))
                ]

            operation = f"RX(π / 2)·{self.current_gate}(w)"

        elif self.current_gate == "H":
            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None, t=None):
                qml.RX(w, wires=0)
                getattr(self, self.current_gate)(0, pulse_params, t)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0))
                ]

            @qml.qnode(dev)
            def ideal_circuit(w):
                qml.RX(w, wires=0)
                getattr(qml, self.current_gate)(wires=0)
                return [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliY(0)),
                    qml.expval(qml.PauliZ(0))
                ]

            operation = f"RX(w)·{self.current_gate}"

        elif self.current_gate == "CZ":
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None, t=None):
                qml.RX(w, wires=0)
                qml.RX(w, wires=1)
                self.CZ(wires=[0, 1])
                qml.RX(-w, wires=1)
                qml.RX(-w, wires=0)
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1))
                ]

            @qml.qnode(dev)
            def ideal_circuit(w):
                qml.RX(w, wires=0)
                qml.RX(w, wires=1)
                qml.CZ(wires=[0, 1])
                qml.RX(-w, wires=1)
                qml.RX(-w, wires=0)
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1))
                ]

            operation = r"$RX_0(w)$·$RX_1(w)$·$CZ_{0, 1}$·$RX_1(-w)$·$RX_0(-w)$"

        elif self.current_gate == "CNOT":
            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, interface="jax")
            def pulse_circuit(w, pulse_params=None, t=None):
                qml.RX(w, wires=0)  # parametrize control qubit
                self.CNOT(wires=[0, 1], params=pulse_params, t_H=t[0], t_CZ=t[1])
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1))
                ]

            @qml.qnode(dev)
            def ideal_circuit(w):
                qml.RX(w, wires=0)
                qml.CNOT(wires=[0, 1])
                return [
                    qml.expval(qml.PauliX(1)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(1))
                ]

            operation = r"$RX_0(w)$·$CNOT_{0,1}$"

        return pulse_circuit, ideal_circuit, operation

    def plot_rotation(self, params: jnp.ndarray):
        """
        Plot the expectation values of the pulse-based and ideal circuits
        for the current gate as a function of the rotation angle.

        Args:
            params (jnp.ndarray): Pulse parameters for the gate.
        """
        pulse_circuit, ideal_circuit, operation = self.get_circuits()

        if self.current_gate in ["RX", "RY", "H", "CZ"]:
            *pulse_params, t = params
            pulse_args = [[pulse_params], t]

        elif self.current_gate in ["RZ"]:
            pulse_args = []

        elif self.current_gate == "CNOT":
            *pulse_params, t_H, t_CZ = params
            pulse_args = [[pulse_params], [t_H, t_CZ]]

        pulse_expvals = [pulse_circuit(w, *pulse_args) for w in self.ws]
        ideal_expvals = [ideal_circuit(w) for w in self.ws]

        pulse_expvals = jnp.array(pulse_expvals)
        ideal_expvals = jnp.array(ideal_expvals)

        fig, axs = plt.subplots(3, 1, figsize=(6, 12))

        bases = ["X", "Y", "Z"]
        for i, basis in enumerate(bases):
            axs[i].plot(self.ws, pulse_expvals[:, i], label="Pulse-based")
            axs[i].plot(self.ws, ideal_expvals[:, i], '--', label="Unitary-based")
            axs[i].set_xlabel("Rotation angle w (rad)")
            axs[i].set_ylabel(f"⟨{basis}⟩")
            axs[i].set_title(f"{operation} in {basis}-basis")
            axs[i].grid(True)
            axs[i].legend()

        xticks = [0, jnp.pi/2, jnp.pi, 3*jnp.pi/2, 2*jnp.pi]
        xtick_labels = ["0", "π/2", "π", "3π/2", "2π"]
        for ax in axs:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)

        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/{self.current_gate}(w)_qoc.png")
        plt.close()

    def save_results(self, opt_params, filename="qml_essentials/qoc_results.csv"):
        """
        Save optimized pulse parameters to CSV file.

        Args:
            opt_params (list or array): Optimized parameters to save.
            filename (str): Path to CSV file.
        """
        header = ["gate"] + [f"param_{i+1}" for i in range(len(opt_params))]
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([self.current_gate] + list(map(float, opt_params)))

    def loss_fn(self, state, target_state):
        """
        Compute infidelity between two quantum states.

        Args:
            state (array): Output state from pulse circuit.
            target_state (array): Target state from ideal circuit.

        Returns:
            float: Infidelity (1 - fidelity).
        """
        fidelity = jnp.abs(jnp.vdot(target_state, state)) ** 2
        return 1 - fidelity

    def cost_fn(self, params, circuit, w, target_state):
        """
        Compute cost for optimization by evaluating circuit and loss.

        Args:
            params (array): Pulse parameters where the last element is
                time t or t_H, t_CZ when the current gate is CNOT.
            circuit (callable): QNode circuit accepting (pulse_params, t, w).
            w (float): Rotation angle.
            target_state (array): Target quantum state.

        Returns:
            float: Computed loss.
        """
        if self.current_gate == "CNOT":
            *pulse_params, t_H, t_CZ = params
            t = [t_H, t_CZ]
        else:
            *pulse_params, t = params
        pulse_params = [pulse_params]
        state = circuit(pulse_params, t, w)
        return self.loss_fn(state, target_state)

    def run_optimization(
        self,
        cost,
        init_params,
        steps,
        print_every,
        *args,
        patience: int = 20
    ):
        """
        Run gradient-based optimization on given cost function.

        Args:
            cost (callable): Cost function to minimize.
            init_params (array): Initial parameters.
            steps (int): Number of optimization steps.
            print_every (int): Print frequency.
            *args: Extra args for cost.
            patience (int): Early stopping patience (default: 20).

        Returns:
            tuple: (optimized parameters, best loss, list of loss values)
        """
        optimizer = optax.adam(0.1)
        opt_state = optimizer.init(init_params)
        params = init_params
        losses = []

        best_loss = float("inf")
        best_params = params
        no_improve_counter = 0

        @jax.jit
        def opt_step(params, opt_state, *args):
            loss, grads = jax.value_and_grad(cost)(params, *args)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for step in range(steps):
            params, opt_state, loss = opt_step(params, opt_state, *args)
            losses.append(loss)

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(
                    f"Loss/train/{self.current_gate}", loss.item(), step
                )

            if loss < best_loss:
                best_loss = loss
                best_params = params
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if (step + 1) % print_every == 0:
                print(f"Step {step + 1}/{steps}, Loss: {loss:.2e}")

            if no_improve_counter >= patience:
                print(f"Early stopping at step {step + 1} due to no improvement.")
                break

        if self.writer is not None:
            self.writer.flush()

        return best_params, best_loss, losses

    def optimize_RX(
            self,
            steps: int = 1000,
            w: float = jnp.pi,
            init_params: jnp.ndarray = jnp.array([1.0, 15.0, 1.0]),
            print_every: int = 50
    ):
        """
        Optimize pulse parameters for the RX(w) gate to best approximate
        the unitary RX(w) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based RX(w) circuit expectation value and the target gate-based RX(w).

        Args:
            steps (int): Number of optimization steps. Default: 600.
            w (float): Rotation angle in radians with which to run the optimization.
               Default: π.
            init_params (jnp.ndarray): Initial pulse parameters (A, sigma) and time.
               Default: [1.0, 15.0, 1.0].
            print_every (int): Frequency of printing loss during optimization.
               Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values
                during optimization.
        """
        self.current_gate = "RX"

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(pulse_params, t, w):
            self.RX(w, 0, pulse_params, t)
            return qml.state()

        @qml.qnode(dev)
        def ideal_circuit(w):
            qml.RX(w, wires=0)
            return qml.state()

        target = ideal_circuit(w)

        # Optimizing
        cost = lambda params: self.cost_fn(params, circuit, w, target)
        params, loss, losses = self.run_optimization(
            cost, init_params, steps, print_every
        )

        # Saving the optimized parameters
        self.save_results(params)

        # Plotting the RX rotation
        if self.make_plots:
            print("Plotting RX rotation...")
            self.plot_rotation(params)

        return params, loss, losses

    def optimize_RY(
            self,
            steps: int = 1000,
            w: float = jnp.pi,
            init_params: jnp.ndarray = jnp.array([1.0, 15.0, 1.0]),
            print_every: int = 50
    ):
        """
        Optimize pulse parameters for the RY(w) gate to best approximate
        the unitary RY(w) gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based RY(w) circuit expectation value and the target gate-based RY(w).

        Args:
            steps (int): Number of optimization steps. Default: 600.
            w (float): Rotation angle in radians with which to run the optimization.
                Default: π.
            init_params (jnp.ndarray): Initial pulse parameters (A, sigma) and time.
                Default: [1.0, 15.0, 1.0].
            print_every (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values
                during optimization.
        """
        self.current_gate = "RY"

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(pulse_params, t, w):
            self.RY(w, 0, pulse_params, t)
            return qml.state()

        @qml.qnode(dev)
        def ideal_circuit(w):
            qml.RY(w, wires=0)
            return qml.state()

        target = ideal_circuit(w)

        # Optimizing
        cost = lambda params: self.cost_fn(params, circuit, w, target)
        params, loss, losses = self.run_optimization(
            cost, init_params, steps, print_every
        )

        # Saving the optimized parameters
        self.save_results(params)

        # Plotting the RY rotation
        if self.make_plots:
            print("Plotting RY rotation...")
            self.plot_rotation(params)

        return params, loss, losses

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
            print("Plotting RZ rotation...")
            self.plot_rotation([])

        return None, None

    def optimize_H(
        self,
        steps=1000,
        init_params: jnp.ndarray = jnp.array([1.0, 15.0, 1.0]),
        print_every: int = 50
    ):
        """
        Optimize pulse parameters for the Hadamard (H) gate to best approximate
        the unitary H gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based H circuit output state and the target gate-based H state.

        Args:
            steps (int): Number of optimization steps. Default: 1000.
            init_params (jnp.ndarray): Initial pulse parameters (A, sigma, t)
                Default: [1.0, 15.0, 1.0].
            print_every (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray), best loss (float), and
                list of loss values during optimization.
        """
        self.current_gate = "H"

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(pulse_params, t, w):
            self.H(0, pulse_params, t)
            return qml.state()

        @qml.qnode(dev)
        def ideal_circuit():
            qml.H(wires=0)
            return qml.state()

        target = ideal_circuit()

        # Optimizing
        cost = lambda params: self.cost_fn(params, circuit, None, target)
        params, loss, losses = self.run_optimization(
            cost, init_params, steps, print_every
        )

        # Saving the optimized parameters
        self.save_results(params)

        # Plotting the RX rotation
        if self.make_plots:
            print("Plotting H rotation...")
            self.plot_rotation(params)

        return params, loss, losses

    def optimize_CZ(
        self,
        steps=1000,
        init_params: jnp.ndarray = jnp.array([1.0]),
        print_every: int = 50
    ):
        """
        Optimize pulse parameters for the CZ gate to best approximate
        the unitary CZ gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based CZ circuit expectation value and the target gate-based CZ.

        Args:
            steps (int): Number of optimization steps. Default: 600.
            init_params (jnp.ndarray): Initial pulse duration. Default: [1.0].
            print_every (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values
                during optimization.
        """
        self.current_gate = "CZ"

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(pulse_params, t, w):
            qml.H(wires=0)
            qml.H(wires=1)
            self.CZ(wires=[0, 1], t=t)
            return qml.state()

        @qml.qnode(dev)
        def ideal_circuit():
            qml.H(wires=0)
            qml.H(wires=1)
            qml.CZ(wires=[0, 1])
            return qml.state()

        target = ideal_circuit()

        # Optimizing
        cost = lambda params: self.cost_fn(params, circuit, None, target)
        params, loss, losses = self.run_optimization(
            cost, init_params, steps, print_every
        )

        # Saving the optimized parameters
        self.save_results(params)

        # Plotting the CZ rotation
        if self.make_plots:
            print("Plotting CZ rotation...")
            self.plot_rotation(params)

        return params, loss, losses

    def optimize_CNOT(
        self,
        steps=1000,
        init_params: jnp.ndarray = jnp.array([1.0, 15.0, 1.0, 1.0]),
        print_every: int = 50
    ):
        """
        Optimize pulse parameters for the CNOT gate to best approximate the
        unitary CNOT gate.

        Uses gradient-based optimization to minimize the difference between the
        pulse-based CNOT circuit expectation value and the target gate-based CNOT.

        Args:
            steps (int): Number of optimization steps (default: 600).
            init_params (jnp.ndarray): Initial pulse parameters (A, sigma) and
                duration (t_H, t_CZ). Default: [1.0, 15.0, 1.0, 1.0].
            print_every (int): Frequency of printing loss during optimization.
                Default: 50.

        Returns:
            tuple: Optimized parameters (jnp.ndarray) and list of loss values during
                optimization.
        """
        self.current_gate = "CNOT"

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(pulse_params, t, w):
            qml.H(wires=0)
            self.CNOT(wires=[0, 1], params=pulse_params, t_H=t[0], t_CZ=t[1])
            return qml.state()

        @qml.qnode(dev)
        def ideal_circuit():
            qml.H(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        target = ideal_circuit()

        # Optimizing
        # def cost(params):
        #     return self.cost_fn(params, circuit, None, target)
        cost = lambda params: self.cost_fn(params, circuit, None, target)
        params, loss, losses = self.run_optimization(
            cost, init_params, steps, print_every
        )

        # Saving the optimized parameters
        self.save_results(params)

        # Plotting the CNOT rotation
        if self.make_plots:
            print("Plotting CNOT rotation...")
            self.plot_rotation(params)

        return params, loss, losses


if __name__ == "__main__":
    qoc = QOC(make_plots=False, fig_points=70)
    
    # # - Run optimization for RX gate -
    # print("Optimizing RX gate...")
    # optimized_params, best_loss, loss_values = qoc.optimize_RX(
    #     w=jnp.pi, init_params=jnp.array([1.0, 15.0, 1.0])
    # )
    # print(f"Optimized parameters for RX: {optimized_params}\n")
    # print(f"Best achieved fidelity: {1 - best_loss}")
    # print("-" * 20, "\n")

    # # - Run optimization for RY gate -
    # print("Optimizing RY gate...")
    # optimized_params, best_loss, loss_values = qoc.optimize_RY(
    #     w=jnp.pi, init_params=jnp.array([1.0, 15.0, 1.0])
    # )
    # print(f"Optimized parameters for RY: {optimized_params}\n")
    # print(f"Best achieved fidelity: {1 - best_loss:.6f}")
    # print("-" * 20, "\n")

    # # - Run optimization for RZ gate -
    # print("Plotting RZ gate rotation...")
    # qoc.optimize_RZ()
    # print("Plotted RZ gate rotation")
    # print("-" * 20, "\n")

    # # - Run optimization for H gate -
    # print("Optimizing H gate...")
    # optimized_params, best_loss, loss_values = qoc.optimize_H(
    #     init_params=jnp.array([1.0, 15.0, 1.0])
    # )
    # print(f"Optimized parameters for H: {optimized_params}\n")
    # print(f"Best achieved fidelity: {1 - best_loss}")
    # print("-" * 20, "\n")

    # # - Run optimization for CZ gate -
    # print("Optimizing CZ gate...")
    # optimized_params, best_loss, loss_values = qoc.optimize_CZ(
    #     init_params=jnp.array([0.975]), print_every=50
    # )
    # print(f"Optimized parameters for CZ: {optimized_params}\n")
    # print(f"Best achieved fidelity: {1 - best_loss}")
    # print("-" * 20, "\n")

    # # - Run optimization for CNOT gate -
    # print("Optimizing CNOT gate...")
    # optimized_params, best_loss, loss_values = qoc.optimize_CNOT(
    #     init_params=jnp.array([1.0, 15.0, 1.0, 1.0]), print_every=50
    # )
    # print(f"Optimized parameters for CNOT: {optimized_params}\n")
    # print(f"Best achieved fidelity: {1 - best_loss}")
    # print("-" * 20, "\n")
