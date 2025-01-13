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
    def noise_gate(noise_params=None):
        if noise_params is not None:
            qml.BitFlip(noise_params.get("BitFlip", 0.0), wires=0)
            qml.PhaseFlip(noise_params.get("PhaseFlip", 0.0), wires=0)
            qml.DepolarizingChannel(
                noise_params.get("DepolarizingChannel", 0.0), wires=0
            )

    def NRot(w, wires, noise_params=None):
        qml.Rot(w[0], w[1], w[2], wires=wires)
        Gates.noise_gate(noise_params)

    def NRX(w, wires, noise_params=None):
        qml.RX(w, wires=wires)
        Gates.noise_gate(noise_params)

    def NRY(w, wires, noise_params=None):
        qml.RY(w, wires=wires)
        Gates.noise_gate(noise_params)

    def NRZ(w, wires, noise_params=None):
        qml.RZ(w, wires=wires)
        Gates.noise_gate(noise_params)

    def NCRX(w, wires, noise_params=None):
        qml.CRX(w, wires=wires)
        Gates.noise_gate(noise_params)

    def NRY(w, wires, noise_params=None):
        qml.CRY(w, wires=wires)
        Gates.noise_gate(noise_params)

    def NCRZ(w, wires, noise_params=None):
        qml.CRZ(w, wires=wires)
        Gates.noise_gate(noise_params)

    def NCX(wires, noise_params=None):
        qml.CNOT(wires=wires)
        Gates.noise_gate(noise_params)

    def NCY(wires, noise_params=None):
        qml.CY(wires=wires)
        Gates.noise_gate(noise_params)

    def NCZ(wires, noise_params=None):
        Gates.NCZ(wires=wires)
        Gates.noise_gate(noise_params)

    def NH(wires, noise_params=None):
        qml.Hadamard(wires=wires)
        Gates.noise_gate(noise_params)


class Ansaetze:

    def get_available():
        return [
            Ansaetze.No_Ansatz,
            Ansaetze.Circuit_1,
            Ansaetze.Circuit_6,
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
                Gates.NRY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.NRZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.NRY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    Gates.NCX(wires=[(2 * q), (2 * q + 1)], noise_params=noise_params)
                for q in range((n_qubits - 1) // 2):
                    Gates.NCX(
                        wires=[(2 * q + 1), (2 * q + 2)], noise_params=noise_params
                    )
                if n_qubits > 2:
                    Gates.NCX(wires=[(n_qubits - 1), 0], noise_params=noise_params)

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
                Gates.NRX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.NRZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.NCRX(
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
                Gates.NRX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.NRZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.NCRZ(
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
            raise NotImplementedError  # Did not figured out the entangling sequence yet

            w_idx = 0
            for q in range(n_qubits):
                Gates.NRX(w[w_idx], wires=q)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.NCX(wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits])

            for q in range(n_qubits):
                Gates.NRZ(w[w_idx], wires=q)
                w_idx += 1

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
                qml.Hadamard(wires=q)

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CZ(
                        wires=[n_qubits - q - 2, n_qubits - q - 1],
                        noise_params=noise_params,
                    )

            for q in range(n_qubits):
                Gates.NRX(w[w_idx], wires=q, noise_params=noise_params)
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
                Gates.NRX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.NRZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for ql in range(n_qubits):
                    for q in range(n_qubits):
                        if q == ql:
                            continue
                        Gates.NCRX(
                            w[w_idx],
                            wires=[n_qubits - ql - 1, (n_qubits - q - 1) % n_qubits],
                            noise_params=noise_params,
                        )
                        w_idx += 1

            for q in range(n_qubits):
                Gates.NRX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.NRZ(w[w_idx], wires=q, noise_params=noise_params)
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
                Gates.NRX(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.NRZ(w[w_idx], wires=q, noise_params=noise_params)
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
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.NCX(wires=[q, (q + 1) % n_qubits])

            for q in range(n_qubits):
                qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.NCX(wires=[q, (q + n_qubits // 2) % n_qubits])

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
                Gates.NRot(
                    w[w_idx],
                    w[w_idx + 1],
                    w[w_idx + 2],
                    wires=q,
                    noise_params=noise_params,
                )
                w_idx += 3
