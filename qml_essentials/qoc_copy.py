import pennylane as qml
from ansaetze import PulseGates
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import os
import csv

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, interface="jax")
def circuit(params, t, w):
    PulseGates().RX(w, params, t, wire=0)
    return qml.expval(qml.Z(0))

# Target state: RX(π)|0⟩ = |1⟩ → ⟨Z⟩ = -1
target = -1.0

def cost_fn(params, w):
    A, sigma, t = params
    p = [jnp.array([A, sigma])]  # omega_c fixed for now
    return (circuit(p, t, w) - target) ** 2

# Initial guess
init_params = jnp.array([1, 15, 1.])  # A, sigma, t
optimizer = optax.adam(0.1)
opt_state = optimizer.init(init_params)

@jax.jit
def step(params, opt_state, w):
    loss, grads = jax.value_and_grad(cost_fn)(params, w)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Run optimization
w = jnp.pi
params = init_params
losses = []

for i in range(400):
    params, opt_state, loss = step(params, opt_state, w)
    losses.append(loss)
    print(loss)

A_opt, sigma_opt, t_opt = params

# Data to save
gate_name = "RX"
opt_params = [float(A_opt), float(sigma_opt), float(t_opt)]  # convert from JAX to native floats

# Create header dynamically
header = ["gate"] + [f"param_{i+1}" for i in range(len(opt_params))]
output_file = "qml_essentials/qoc_results.csv"

# Check if file exists to decide on writing header
file_exists = os.path.isfile(output_file)

with open(output_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerow([gate_name] + opt_params)

ws = jnp.linspace(0, 2 * jnp.pi, 100)
expvals = [circuit([jnp.array([A_opt, sigma_opt])], t=t_opt, w=w) for w in ws]
expected = jnp.cos(ws)

plt.plot(ws, expvals, label="Pulse-based RX")
plt.plot(ws, expected, '--', label="Ideal RX (cos(θ))")
plt.xlabel("Rotation angle w (rad)")
plt.ylabel("⟨Z⟩")
plt.legend()
plt.grid(True)
plt.title("RX(w) via pulse vs ideal RX(w)")
plt.savefig("qml_essentials/figs/RX(w)_qoc.png")
plt.close()
