import pennylane as qml
from jax import numpy as jnp
from ansaetze import PulseGates
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, interface="jax")
def circuit(params, t):
    PulseGates().RX(jnp.pi, params, t, wire=0)
    return qml.expval(qml.Z(0))

dt = 0.002
times = jnp.arange(0, dt * 80, dt)
# times = jnp.linspace(0, jnp.pi / 2, 60)

for A in [1, 10, 30]:
    for wc in [1, 10, 30]:
        params = jnp.array([A, 10, wc * jnp.pi])  # A, sigma, omega_c
        results = [circuit([params], t=float(t)) for t in times]

        plt.scatter(times, results, s=10)
        plt.xlabel('Time (t)')
        plt.ylabel('Circuit Output')
        plt.title('Circuit Output vs Time')
        plt.grid(True)
        plt.savefig(f"qml_essentials/figs/RX_{params[0]}_{params[1]}_{params[2]:.2f}.png")
        plt.close()
