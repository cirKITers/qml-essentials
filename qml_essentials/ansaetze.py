from abc import ABC, abstractmethod
from typing import Any, Optional
import pennylane.numpy as np
import pennylane as qml

from typing import List

import logging

log = logging.getLogger(__name__)


class Circuit(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def n_params_per_layer(n_qubits: int) -> int:
        return

    @abstractmethod
    def get_control_indices(self, n_qubits: int) -> List[int]:
        """
        Returns the indices for the controlled rotation gates for one layer.
        Indices should slice the list of all parameters for one layer as follows:
        [indices[0]:indices[1]:indices[2]]

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit

        Returns
        -------
        Optional[np.ndarray]
            List of all controlled indices, or None if the circuit does not
            contain controlled rotation gates.
        """
        return

    def get_control_angles(self, w: np.ndarray, n_qubits: int) -> Optional[np.ndarray]:
        """
        Returns the angles for the controlled rotation gates from the list of
        all parameters for one layer.

        Parameters
        ----------
        w : np.ndarray
            List of parameters for one layer
        n_qubits : int
            Number of qubits in the circuit

        Returns
        -------
        Optional[np.ndarray]
            List of all controlled parameters, or None if the circuit does not
            contain controlled rotation gates.
        """
        indices = self.get_control_indices(n_qubits)
        if indices is None:
            return None

        return w[indices[0] : indices[1] : indices[2]]

    @abstractmethod
    def build(self, n_qubits: int, n_layers: int):
        return

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.build(*args, **kwds)


class Ansaetze:
    def get_available():
        return [
            Ansaetze.No_Ansatz,
            Ansaetze.Circuit_1,
            Ansaetze.Circuit_9,
            Ansaetze.Circuit_15,
            Ansaetze.Circuit_18,
            Ansaetze.Circuit_19,
            Ansaetze.No_Entangling,
            Ansaetze.Strongly_Entangling,
            Ansaetze.Hardware_Efficient,
        ]

    class No_Ansatz(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return 0

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            pass

    class Hardware_Efficient(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RY(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1
                qml.RY(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    qml.CZ(wires=[(2 * q), (2 * q + 1)])
                for q in range((n_qubits - 1) // 2):
                    qml.CZ(wires=[(2 * q + 1), (2 * q + 2)])

    class Bansatz(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RY(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    qml.CRX(w[w_idx], wires=[(2 * q), (2 * q + 1)])
                    w_idx += 1

                for q in range((n_qubits - 1) // 2):
                    qml.CRX(w[w_idx], wires=[(2 * q + 1), (2 * q + 2)])
                    w_idx += 1

    class Circuit_19(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CRX(
                        w[w_idx],
                        wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits],
                    )
                    w_idx += 1

    class Circuit_18(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit18 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CRZ(
                        w[w_idx],
                        wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits],
                    )
                    w_idx += 1

    class Circuit_15(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit15 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            raise NotImplementedError  # Did not figured out the entangling sequence yet

            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits])

            for q in range(n_qubits):
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

    class Circuit_9(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3-1
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.Hadamard(wires=q)

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CZ(wires=[n_qubits - q - 2, n_qubits - q - 1])

            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1

    class Circuit_1(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a Circuit1 ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.RX(w[w_idx], wires=q)
                w_idx += 1
                qml.RZ(w[w_idx], wires=q)
                w_idx += 1

    class Strongly_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 6
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int) -> None:
            """
            Creates a StronglyEntanglingLayers ansatz.

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])

            for q in range(n_qubits):
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + n_qubits // 2) % n_qubits])

    class No_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int):
            """
            Creates a circuit without entangling, but with U3 gates on all qubits

            Length of flattened vector must be n_qubits*3

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3
