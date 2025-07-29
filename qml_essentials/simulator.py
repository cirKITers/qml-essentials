import jax
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
    def RX(theta, wires):
        return qml.RX(theta, wires=wires)

    @staticmethod
    def RY(theta, wires):
        return qml.RY(theta, wires=wires)

    @staticmethod
    def RZ(theta, wires):
        return qml.RZ(theta, wires=wires)

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
    def SWAP(wires):
        return qml.SWAP(wires=wires)

    @staticmethod
    def I(wires):
        return qml.Identity(wires=wires)

    @staticmethod
    def H(wires):
        return qml.Hadamard(wires=wires)
