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


class Gates:
    def noise_gate(wires, noise_params=None):
        if noise_params is not None:
            if isinstance(wires, int):
                wires = [wires]  # single qubit gate
            # iterate for multi qubit gates
            for wire in wires:
                qml.BitFlip(noise_params.get("BitFlip", 0.0), wires=wire)
                qml.PhaseFlip(noise_params.get("PhaseFlip", 0.0), wires=wire)
                qml.DepolarizingChannel(
                    noise_params.get("DepolarizingChannel", 0.0), wires=wire
                )

    def Rot(phi, theta, omega, wires, noise_params=None):
        qml.Rot(phi, theta, omega, wires=wires)
        Gates.noise_gate(wires, noise_params)

    def RX(w, wires, noise_params=None):
        qml.RX(w, wires=wires)
        Gates.noise_gate(wires, noise_params)

    def RY(w, wires, noise_params=None):
        qml.RY(w, wires=wires)
        Gates.noise_gate(wires, noise_params)

    def RZ(w, wires, noise_params=None):
        qml.RZ(w, wires=wires)
        Gates.noise_gate(wires, noise_params)

    def CRX(w, wires, noise_params=None):
        qml.CRX(w, wires=wires)
        Gates.noise_gate(wires, noise_params)

    def CRY(w, wires, noise_params=None):
        qml.CRY(w, wires=wires)
        Gates.noise_gate(wires, noise_params)

    def CRZ(w, wires, noise_params=None):
        qml.CRZ(w, wires=wires)
        Gates.noise_gate(wires, noise_params)

    def CX(wires, noise_params=None):
        qml.CNOT(wires=wires)
        Gates.noise_gate(wires, noise_params)

    def CY(wires, noise_params=None):
        qml.CY(wires=wires)
        Gates.noise_gate(wires, noise_params)

    def CZ(wires, noise_params=None):
        qml.CZ(wires=wires)
        Gates.noise_gate(wires, noise_params)

    def H(wires, noise_params=None):
        qml.Hadamard(wires=wires)
        Gates.noise_gate(wires, noise_params)


class Ansaetze:

    def get_available():
        return [
            Ansaetze.No_Ansatz,
            Ansaetze.Circuit_1,
            Ansaetze.Circuit_2,
            Ansaetze.Circuit_3,
            Ansaetze.Circuit_4,
            Ansaetze.Circuit_6,
            Ansaetze.Circuit_9,
            Ansaetze.Circuit_10,
            Ansaetze.Circuit_15,
            Ansaetze.Circuit_16,
            Ansaetze.Circuit_17,
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
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
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
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Hardware-Efficient ansatz, as proposed in
            https://arxiv.org/pdf/2309.03279

            Length of flattened vector must be n_qubits*3

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    Gates.CX(wires=[(2 * q), (2 * q + 1)], noise_params=noise_params)
                for q in range((n_qubits - 1) // 2):
                    Gates.CX(
                        wires=[(2 * q + 1), (2 * q + 2)], noise_params=noise_params
                    )
                if n_qubits > 2:
                    Gates.CX(wires=[(n_qubits - 1), 0], noise_params=noise_params)

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
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
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
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CRX(
                        w[w_idx],
                        wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits],
                        noise_params=noise_params,
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
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit18 ansatz.

            Length of flattened vector must be n_qubits*3

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CRZ(
                        w[w_idx],
                        wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits],
                        noise_params=noise_params,
                    )
                    w_idx += 1

    class Circuit_15(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 2
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit15 ansatz.

            Length of flattened vector must be n_qubits*2
            because for >1 qubits there are three gates

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits], noise_params=noise_params)

            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(wires=[(q - 1) % n_qubits, (q - 2) % n_qubits], noise_params=noise_params)

    class Circuit_9(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit9 ansatz.

            Length of flattened vector must be n_qubits

            Args:
                w (np.ndarray): weight vector of size n_layers*n_qubits
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.H(wires=q, noise_params=noise_params)

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    Gates.CZ(
                        wires=[n_qubits - q - 2, n_qubits - q - 1],
                        noise_params=noise_params,
                    )

            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

    class Circuit_6(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            if n_qubits > 1:
                return n_qubits * 3 + n_qubits**2
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 4

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit6 ansatz.

            Length of flattened vector must be
                n_qubits * 4 + n_qubits * (n_qubits - 1) =
                n_qubits * 3 + n_qubits**2

            Args:
                w (np.ndarray): weight vector of size
                    n_layers * (n_qubits * 3 + n_qubits**2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for ql in range(n_qubits):
                    for q in range(n_qubits):
                        if q == ql:
                            continue
                        Gates.CRX(
                            w[w_idx],
                            wires=[n_qubits - ql - 1, (n_qubits - q - 1) % n_qubits],
                            noise_params=noise_params,
                        )
                        w_idx += 1

            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

    class Circuit_1(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit1 ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

    class Circuit_2(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 2

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit2 ansatz.

            Length of flattened vector must be n_qubits*2

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
            
            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    Gates.CX(wires=[n_qubits - q - 1, n_qubits - q - 2], noise_params=noise_params)

    class Circuit_3(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 - 1

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit3 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
            
            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    Gates.CRZ(
                        w[w_idx],
                        wires=[n_qubits - q - 1, n_qubits - q - 2],
                        noise_params=noise_params
                    )
                    w_idx += 1

    class Circuit_4(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 - 1

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit4 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
            
            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    Gates.CRX(
                        w[w_idx],
                        wires=[n_qubits - q - 1, n_qubits - q - 2],
                        noise_params=noise_params
                    )
                    w_idx += 1

    class Circuit_10(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 2 # constant gates not considered yet. has to be fixed

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit10 ansatz.

            Length of flattened vector must be n_qubits

            Args:
                w (np.ndarray): weight vector of size n_layers*n_qubits
                n_qubits (int): number of qubits
            """
            w_idx = 0
            # constant gates, independent of layers. has to be fixed
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    Gates.CZ(wires=[(n_qubits - q - 2) % n_qubits, (n_qubits - q - 1) % n_qubits], noise_params=noise_params)
                if n_qubits > 2:
                    Gates.CZ(wires=[n_qubits - 1, 0], noise_params=noise_params)

            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

    class Circuit_16(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 - 1

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit16 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Args:
                w (np.ndarray): weight vector of size n_layers*n_qubits*3-1
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    Gates.CRZ(
                            w[w_idx],
                            wires=[(2 * q + 1), (2 * q)],
                            noise_params=noise_params
                        )
                    w_idx += 1

                for q in range((n_qubits - 1) // 2):
                    Gates.CRZ(
                            w[w_idx],
                            wires=[(2 * q + 2), (2 * q + 1)],
                            noise_params=noise_params
                        )
                    w_idx += 1

    class Circuit_17(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3 - 1

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a Circuit17 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Args:
                w (np.ndarray): weight vector of size n_layers*n_qubits*3-1
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    Gates.CRX(
                            w[w_idx],
                            wires=[(2 * q + 1), (2 * q)],
                            noise_params=noise_params
                        )
                    w_idx += 1

                for q in range((n_qubits - 1) // 2):
                    Gates.CRX(
                            w[w_idx],
                            wires=[(2 * q + 2), (2 * q + 1)],
                            noise_params=noise_params
                        )
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
        def build(w: np.ndarray, n_qubits: int, noise_params=None) -> None:
            """
            Creates a StronglyEntanglingLayers ansatz.

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*6)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.Rot(
                    w[w_idx],
                    w[w_idx + 1],
                    w[w_idx + 2],
                    wires=q,
                    noise_params=noise_params,
                )
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(wires=[q, (q + 1) % n_qubits], noise_params=noise_params)

            for q in range(n_qubits):
                Gates.Rot(
                    w[w_idx],
                    w[w_idx + 1],
                    w[w_idx + 2],
                    wires=q,
                    noise_params=noise_params,
                )
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(
                        wires=[q, (q + n_qubits // 2) % n_qubits],
                        noise_params=noise_params,
                    )

    class No_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            """
            Creates a circuit without entangling, but with U3 gates on all qubits

            Length of flattened vector must be n_qubits*3

            Args:
                w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
                n_qubits (int): number of qubits
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.Rot(
                    w[w_idx],
                    w[w_idx + 1],
                    w[w_idx + 2],
                    wires=q,
                    noise_params=noise_params,
                )
                w_idx += 3
