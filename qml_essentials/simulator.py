# import jax
import pennylane as qml


class QNode:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        mat = qml.matrix(self.func(*args, **kwargs))
        return self.func(*args, **kwargs)

    @staticmethod
    def matrix(op):
        return qml.matrix(op)

    @staticmethod
    def BitFlip(p, wires):
        return qml.BitFlip(p, wires=wires)

    @staticmethod
    def PhaseFlip(p, wires):
        return qml.PhaseFlip(p, wires=wires)

    @staticmethod
    def DepolarizingChannel(p, wires):
        return qml.DepolarizingChannel(p, wires=wires)

    @staticmethod
    def NQubitDepolarizingChannel(p, wires):
        return qml.NQubitDepolarizingChannel(p, wires=wires)

    @staticmethod
    def QubitChannel(k_list, wires):
        return qml.QubitChannel(k_list, wires=wires)

    @staticmethod
    def PauliX(wires):
        return qml.PauliX(wires=wires)

    @staticmethod
    def PauliY(wires):
        return qml.PauliY(wires=wires)

    @staticmethod
    def PauliZ(wires):
        return qml.PauliZ(wires=wires)

    @staticmethod
    def RX(theta, wires):
        return qml.RX(theta, wires=wires)

    @staticmethod
    def RY(theta, wires):
        return qml.RY(theta, wires=wires)

    @staticmethod
    def RZ(theta, wires):
        return qml.RZ(theta, wires=wires)

    @staticmethod
    def Rot(phi, theta, omega, wires):
        return qml.Rot(phi, theta, omega, wires=wires)

    @staticmethod
    def CX(wires):
        return qml.CNOT(wires=wires)

    @staticmethod
    def CRX(theta, wires):
        return qml.CRX(theta, wires=wires)

    @staticmethod
    def CY(wires):
        return qml.CY(wires=wires)

    @staticmethod
    def CRY(theta, wires):
        return qml.CRY(theta, wires=wires)

    @staticmethod
    def CZ(wires):
        return qml.CZ(wires=wires)

    @staticmethod
    def CRZ(theta, wires):
        return qml.CRZ(theta, wires=wires)

    @staticmethod
    def CNOT(wires):
        return qml.CNOT(wires=wires)

    @staticmethod
    def SWAP(wires):
        return qml.SWAP(wires=wires)

    @staticmethod
    def I(wires):
        return qml.Identity(wires=wires)

    @staticmethod
    def Hadamard(wires):
        return qml.Hadamard(wires=wires)
