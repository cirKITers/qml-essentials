import hashlib
import sys, os
import subprocess
import importlib

devnull = open(os.devnull, "w")

print("Installing old version of pennylane...")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "pennylane==0.40",
    ],
    stdout=devnull,
    stderr=subprocess.STDOUT,
)


import pennylane as qml
import pennylane.numpy as np

print(qml.__version__)


@qml.qnode(device=qml.device("default.mixed", wires=2))
def circuit(theta):
    qml.RZ(theta, wires=0)
    qml.CRZ(theta, wires=[0, 1])
    return qml.density_matrix(wires=[0, 1])


rho = circuit(np.array([1, 2]))

hs_new = hashlib.md5(repr(rho).encode("utf-8"))
print(hs_new.hexdigest())


print("Installing new version of pennylane...")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "git+https://github.com/PennyLaneAI/pennylane.git@fix-default-mixed-bug",
    ],
    stdout=devnull,
    stderr=subprocess.STDOUT,
)

importlib.reload(qml)
importlib.reload(np)

print(qml.__version__)


@qml.qnode(device=qml.device("default.mixed", wires=2))
def circuit(theta):
    qml.RZ(theta, wires=0)
    qml.CRZ(theta, wires=[0, 1])
    return qml.density_matrix(wires=[0, 1])


rho = circuit(np.array([1, 2]))

hs_old = hashlib.md5(repr(rho).encode("utf-8"))
print(hs_old.hexdigest())

assert hs_old.hexdigest() == hs_new.hexdigest(), "Hashes do not match"
print("Hashes match")
