import pennylane as qml
from jax import numpy as jnp
from ansaetze import PulseGates


dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, interface="jax")
def circuit(params, t):
    PulseGates().RX(jnp.pi, params, t, wire=0)
    # PulseGates().RY(jnp.pi, params, t, wire=0)
    return qml.expval(qml.Z(0))

# Example
params = jnp.array([1, 15, 10 * jnp.pi])  # A, sigma, omega_c
print(circuit([params], t=0.002))
print(circuit([params], t=0.004))
print(circuit([params], t=0.006))
print(circuit([params], t=0.008))
print(circuit([params], t=0.01))
print(circuit([params], t=0.012))
print(circuit([params], t=0.014))
print(circuit([params], t=0.016))
print(circuit([params], t=0.018))
print(circuit([params], t=0.02))

# result = circuit([params], t=[2, 5])
# print(result)