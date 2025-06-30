import os
import csv
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from ansaetze import PulseGates
import matplotlib.pyplot as plt


def run_optimization(cost_fn, init_params, optimizer, steps, *args):
    opt_state = optimizer.init(init_params)
    params = init_params
    losses = []

    @jax.jit
    def step(params, opt_state, *args):
        loss, grads = jax.value_and_grad(cost_fn)(params, *args)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state, *args)
        losses.append(loss)
    
    return params, losses

def save_results(gate_name, opt_params, filename="qml_essentials/qoc_results.csv"):
    header = ["gate"] + [f"param_{i+1}" for i in range(len(opt_params))]
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([gate_name] + list(map(float, opt_params)))

def plot_results(ws, expvals, expected, gate_name):
    plt.plot(ws, expvals, label=f"Pulse-based {gate_name}")
    plt.plot(ws, expected, '--', label=f"Unitary-based {gate_name}")
    plt.xlabel("Rotation angle w (rad)")
    plt.ylabel("⟨Z⟩")
    plt.legend()
    plt.grid(True)
    plt.title(f"Pulse {gate_name}(w) vs unitary {gate_name}(w)")
    plt.savefig(f"qml_essentials/figs/{gate_name}(w)_qoc.png")
    plt.close()

def optimize_RX():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="jax")
    def circuit(params, t, w):
        PulseGates().RX(w, params, t, wire=0)
        return qml.expval(qml.Z(0))

    target = -1.0
    def cost_fn(params, w):
        A, sigma, t = params
        p = [jnp.array([A, sigma])]
        return (circuit(p, t, w) - target) ** 2

    init_params = jnp.array([1.0, 15.0, 1.0])
    optimizer = optax.adam(0.1)
    w = jnp.pi
    steps = 600

    params, losses = run_optimization(cost_fn, init_params, optimizer, steps, w)
    A_opt, sigma_opt, t_opt = params

    save_results("RX", [A_opt, sigma_opt, t_opt])

    ws = jnp.linspace(0, 2 * jnp.pi, 100)
    expvals = [circuit([jnp.array([A_opt, sigma_opt])], t=t_opt, w=w_) for w_ in ws]
    expected = jnp.cos(ws)
    plot_results(ws, expvals, expected, "RX")


if __name__ == "__main__":
    optimize_RX()
    print("Optimization and plotting completed successfully.")