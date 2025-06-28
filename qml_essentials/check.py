import pennylane as qml
from jax import numpy as jnp
from ansaetze import PulseGates


from scipy.integrate import quad
import numpy as np

def Sx(t, A, sigma, T):
    return A * np.exp(-0.5 * ((t - T/2) / sigma) ** 2)

T = 1.0 
sigma = 15 
A = 1.0
alpha = 0.3

integral, _ = quad(Sx, 0, T, args=(A, sigma, T))

theta = 2 * integral
print("Rotation angle theta:", theta)
print("Half pi rotation:", np.pi / 2)

# dev = qml.device("default.qubit", wires=1)

# @qml.qnode(dev, interface="jax")
# def circuit(params, t):
#     gate = PulseGates().RX(jnp.pi / 2, wires=0)
#     gate(params, t=t)  # t is a list like [0, 4]
#     return qml.expval(qml.Z(0))

# # Example parameters and time interval
# T = 4.0  # Duration of the pulse
# t0 = 0.0  # Start time
# params = jnp.array([1.0, 1., T, 0.3])  # A, sigma, T, alpha
# time_interval = [t0, t0 + T]

# result = circuit(params=[params, params], t=time_interval)
# print(result)
