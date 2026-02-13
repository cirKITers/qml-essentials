import os
from typing import Optional, List, Union, Dict, Callable, Tuple
import numbers
import csv
import jax.numpy as np
import pennylane as qml
import jax
import itertools
from contextlib import contextmanager
import logging

from qml_essentials.utils import safe_random_split

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


class UnitaryGates:
    """Collection of unitary quantum gates with optional noise simulation."""

    batch_gate_error = True

    @staticmethod
    def NQubitDepolarizingChannel(p: float, wires: List[int]) -> qml.QubitChannel:
        """
        Generate Kraus operators for n-qubit depolarizing channel.

        The n-qubit depolarizing channel models uniform depolarizing noise
        acting on n qubits simultaneously, useful for simulating realistic
        multi-qubit noise affecting entangling gates.

        Args:
            p (float): Total probability of depolarizing error (0 ≤ p ≤ 1).
            wires (List[int]): Qubit indices on which the channel acts.
                Must contain at least 2 qubits.

        Returns:
            qml.QubitChannel: PennyLane QubitChannel with Kraus operators
                representing the depolarizing noise channel.

        Raises:
            ValueError: If p is not in [0, 1] or if fewer than 2 qubits provided.
        """

        def n_qubit_depolarizing_kraus(p: float, n: int) -> List[np.ndarray]:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability p must be between 0 and 1, got {p}")
            if n < 2:
                raise ValueError(f"Number of qubits must be >= 2, got {n}")

            Id = np.eye(2)
            X = qml.matrix(qml.PauliX(0))
            Y = qml.matrix(qml.PauliY(0))
            Z = qml.matrix(qml.PauliZ(0))
            paulis = [Id, X, Y, Z]

            dim = 2**n
            all_ops = []

            # Generate all n-qubit Pauli tensor products:
            for indices in itertools.product(range(4), repeat=n):
                P = np.eye(1)
                for idx in indices:
                    P = np.kron(P, paulis[idx])
                all_ops.append(P)

            # Identity operator corresponds to all zeros indices (Id^n)
            K0 = np.sqrt(1 - p * (4**n - 1) / (4**n)) * np.eye(dim)

            kraus_ops = []
            for i, P in enumerate(all_ops):
                if i == 0:
                    # Skip the identity, already handled as K0
                    continue
                kraus_ops.append(np.sqrt(p / (4**n)) * P)

            return [K0] + kraus_ops

        return qml.QubitChannel(n_qubit_depolarizing_kraus(p, len(wires)), wires=wires)

    @staticmethod
    def Noise(
        wires: Union[int, List[int]], noise_params: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Apply noise channels to specified qubits.

        Applies various single-qubit and multi-qubit noise channels based on
        the provided noise parameters dictionary.

        Args:
            wires (Union[int, List[int]]): Qubit index or list of qubit indices
                to apply noise to.
            noise_params (Optional[Dict[str, float]]): Dictionary of noise
                parameters. Supported keys:
                - "BitFlip" (float): Bit flip error probability
                - "PhaseFlip" (float): Phase flip error probability
                - "Depolarizing" (float): Single-qubit depolarizing probability
                - "MultiQubitDepolarizing" (float): Multi-qubit depolarizing
                  probability (applies if len(wires) > 1)
                All parameters default to 0.0 if not provided.

        Returns:
            None: Noise channels are applied in-place to the circuit.
        """
        if noise_params is not None:
            if isinstance(wires, int):
                wires = [wires]  # single qubit gate

            # noise on single qubits
            for wire in wires:
                bf = noise_params.get("BitFlip", 0.0)
                if bf > 0:
                    qml.BitFlip(bf, wires=wire)

                pf = noise_params.get("PhaseFlip", 0.0)
                if pf > 0:
                    qml.PhaseFlip(pf, wires=wire)

                dp = noise_params.get("Depolarizing", 0.0)
                if dp > 0:
                    qml.DepolarizingChannel(dp, wires=wire)

            # noise on two-qubits
            if len(wires) > 1:
                p = noise_params.get("MultiQubitDepolarizing", 0.0)
                if p > 0:
                    UnitaryGates.NQubitDepolarizingChannel(p, wires)

    @staticmethod
    def GateError(
        w: Union[float, np.ndarray, List[float]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[np.ndarray, jax.random.PRNGKey]:
        """
        Apply gate error noise to rotation angle(s).

        Adds Gaussian noise to gate rotation angles to simulate imperfect
        gate implementations.

        Args:
            w (Union[float, np.ndarray, List[float]]): Rotation angle(s) in radians.
            noise_params (Optional[Dict[str, float]]): Dictionary with optional
                "GateError" key specifying standard deviation of Gaussian noise.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for
                stochastic noise generation.

        Returns:
            Tuple[np.ndarray, jax.random.PRNGKey]: Tuple containing:
                - Modified rotation angle(s) with applied noise
                - Updated JAX random key

        Raises:
            AssertionError: If noise_params contains "GateError" but random_key is None.
        """
        if noise_params is not None and noise_params.get("GateError", None) is not None:
            assert (
                random_key is not None
            ), "A random_key must be provided when using GateError"

            random_key, sub_key = safe_random_split(random_key)
            w += noise_params["GateError"] * jax.random.normal(
                sub_key,
                (
                    w.shape
                    if isinstance(w, np.ndarray) and UnitaryGates.batch_gate_error
                    else (1,)
                ),
            )
        return w, random_key

    @staticmethod
    def Rot(
        phi: Union[float, np.ndarray, List[float]],
        theta: Union[float, np.ndarray, List[float]],
        omega: Union[float, np.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply general rotation gate with optional noise.

        Applies a three-angle rotation Rot(phi, theta, omega) with optional
        gate errors and noise channels.

        Args:
            phi (Union[float, np.ndarray, List[float]]): First rotation angle.
            theta (Union[float, np.ndarray, List[float]]): Second rotation angle.
            omega (Union[float, np.ndarray, List[float]]): Third rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
                Supports BitFlip, PhaseFlip, Depolarizing, and GateError.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        if noise_params is not None and "GateError" in noise_params:
            phi, random_key = UnitaryGates.GateError(phi, noise_params, random_key)
            theta, random_key = UnitaryGates.GateError(theta, noise_params, random_key)
            omega, random_key = UnitaryGates.GateError(omega, noise_params, random_key)
        qml.Rot(phi, theta, omega, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RX(
        w: Union[float, np.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply X-axis rotation with optional noise.

        Args:
            w (Union[float, np.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        qml.RX(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RY(
        w: Union[float, np.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply Y-axis rotation with optional noise.

        Args:
            w (Union[float, np.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        qml.RY(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RZ(
        w: Union[float, np.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply Z-axis rotation with optional noise.

        Args:
            w (Union[float, np.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        qml.RZ(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRX(
        w: Union[float, np.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply controlled X-rotation with optional noise.

        Args:
            w (Union[float, np.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        qml.CRX(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRY(
        w: Union[float, np.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply controlled Y-rotation with optional noise.

        Args:
            w (Union[float, np.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        qml.CRY(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRZ(
        w: Union[float, np.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply controlled Z-rotation with optional noise.

        Args:
            w (Union[float, np.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        qml.CRZ(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CX(
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply controlled-NOT (CNOT) gate with optional noise.

        Args:
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        qml.CNOT(wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CY(
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply controlled-Y gate with optional noise.

        Args:
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        qml.CY(wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CZ(
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply controlled-Z gate with optional noise.

        Args:
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        qml.CZ(wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def H(
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply Hadamard gate with optional noise.

        Args:
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        qml.Hadamard(wires=wires)
        UnitaryGates.Noise(wires, noise_params)


class PulseParams:
    """
    Container for hierarchical pulse parameters.

    Manages pulse parameters for quantum gates, supporting both leaf nodes
    (gates with direct parameters) and composite nodes (gates decomposed
    into simpler gates). Enables hierarchical parameter access and
    manipulation.

    Attributes:
        name (str): Name identifier for the gate.
        _params (np.ndarray): Direct pulse parameters (leaf nodes only).
        _pulse_obj (List): Child PulseParams objects (composite nodes only).
    """

    def __init__(
        self,
        name: str = "",
        params: Optional[np.ndarray] = None,
        pulse_obj: Optional[List] = None,
    ) -> None:
        """
        Initialize pulse parameters container.

        Args:
            name (str): Name identifier for the gate. Defaults to empty string.
            params (Optional[np.ndarray]): Direct pulse parameters for leaf gates.
                Mutually exclusive with pulse_obj.
            pulse_obj (Optional[List]): List of child PulseParams for composite
                gates. Mutually exclusive with params.

        Raises:
            AssertionError: If both or neither of params and pulse_obj are provided.
        """
        assert (params is None and pulse_obj is not None) or (
            params is not None and pulse_obj is None
        ), "Exactly one of `params` or `pulse_params` must be provided."

        self._pulse_obj = pulse_obj

        if params is not None:
            self._params = params

        self.name = name

    def __len__(self) -> int:
        """
        Get the total number of pulse parameters.

        For composite gates, returns the accumulated count from all children.

        Returns:
            int: Total number of pulse parameters.
        """
        return len(self.params)

    def __getitem__(self, idx: int) -> Union[float, np.ndarray]:
        """
        Access pulse parameter(s) by index.

        For leaf gates, returns the parameter at the given index.
        For composite gates, returns parameters of the child at the given index.

        Args:
            idx (int): Index to access.

        Returns:
            Union[float, np.ndarray]: Parameter value or child parameters.
        """
        if self.is_leaf:
            return self.params[idx]
        else:
            return self.childs[idx].params

    def __str__(self) -> str:
        """Return string representation (gate name)."""
        return self.name

    def __repr__(self) -> str:
        """Return repr string (gate name)."""
        return self.name

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (direct parameters, no children)."""
        return self._pulse_obj is None

    @property
    def size(self) -> int:
        """Get the total parameter count (alias for __len__)."""
        return len(self)

    @property
    def leafs(self) -> List["PulseParams"]:
        """
        Get all leaf nodes in the hierarchy.

        Recursively collects all leaf PulseParams objects in the tree.

        Returns:
            List[PulseParams]: List of unique leaf nodes.
        """
        if self.is_leaf:
            return [self]

        leafs = []
        for obj in self._pulse_obj:
            leafs.extend(obj.leafs)

        return list(set(leafs))

    @property
    def childs(self) -> List["PulseParams"]:
        """
        Get direct children of this node.

        Returns:
            List[PulseParams]: List of child PulseParams objects, or empty list
                if this is a leaf node.
        """
        if self.is_leaf:
            return []

        return self._pulse_obj

    @property
    def shape(self) -> List[int]:
        """
        Get the shape of pulse parameters.

        For leaf nodes, returns list with parameter count.
        For composite nodes, returns nested list of child shapes.

        Returns:
            List[int]: Parameter shape specification.
        """
        if self.is_leaf:
            return [len(self.params)]

        shape = []
        for obj in self.childs:
            shape.append(*obj.shape())

        return shape

    @property
    def params(self) -> np.ndarray:
        """
        Get or compute pulse parameters.

        For leaf nodes, returns internal pulse parameters.
        For composite nodes, returns concatenated parameters from all children.

        Returns:
            np.ndarray: Pulse parameters array.
        """
        if self.is_leaf:
            return self._params

        params = self.split_params(params=None, leafs=False)

        return np.concatenate(params)

    @params.setter
    def params(self, value: np.ndarray) -> None:
        """
        Set pulse parameters.

        For leaf nodes, sets internal parameters directly.
        For composite nodes, distributes values across children.

        Args:
            value (np.ndarray): Pulse parameters to set.

        Raises:
            AssertionError: If value is not np.ndarray for leaf nodes.
        """
        if self.is_leaf:
            assert isinstance(value, np.ndarray), "params must be a np.ndarray"
            self._params = value
            return

        idx = 0
        for obj in self.childs:
            nidx = idx + obj.size
            obj.params = value[idx:nidx]
            idx = nidx

    @property
    def leaf_params(self) -> np.ndarray:
        """
        Get parameters from all leaf nodes.

        Returns:
            np.ndarray: Concatenated parameters from all leaf nodes.
        """
        if self.is_leaf:
            return self._params

        params = self.split_params(None, leafs=True)

        return np.concatenate(params)

    @leaf_params.setter
    def leaf_params(self, value: np.ndarray) -> None:
        """
        Set parameters for all leaf nodes.

        Args:
            value (np.ndarray): Parameters to distribute across leaf nodes.
        """
        if self.is_leaf:
            self._params = value
            return

        idx = 0
        for obj in self.leafs:
            nidx = idx + obj.size
            obj.params = value[idx:nidx]
            idx = nidx

    def split_params(
        self,
        params: Optional[np.ndarray] = None,
        leafs: bool = False,
    ) -> List[np.ndarray]:
        """
        Split parameters into sub-arrays for children or leaves.

        Args:
            params (Optional[np.ndarray]): Parameters to split. If None,
                uses internal parameters.
            leafs (bool): If True, splits across leaf nodes; if False,
                splits across direct children. Defaults to False.

        Returns:
            List[np.ndarray]: List of parameter arrays for children or leaves.
        """
        if params is None:
            if self.is_leaf:
                return self._params

            objs = self.leafs if leafs else self.childs
            s_params = []
            for obj in objs:
                s_params.append(obj.params)

            return s_params
        else:
            if self.is_leaf:
                return params

            objs = self.leafs if leafs else self.childs
            s_params = []
            idx = 0
            for obj in objs:
                nidx = idx + obj.size
                s_params.append(params[idx:nidx])
                idx = nidx

            return s_params


class PulseInformation:
    """
    Stores pulse parameter counts and optimized pulse parameters for quantum Gates.
    """

    RX = PulseParams(
        name="RX",
        params=np.array([15.863171563255692, 29.66617464185762, 0.7544382603281181]),
    )
    RY = PulseParams(
        name="RY",
        params=np.array([7.921864297441735, 22.038129802391797, 1.0940923114464387]),
    )
    RZ = PulseParams(name="RZ", params=np.array([0.5]))
    CZ = PulseParams(name="CZ", params=np.array([0.3183095268754836]))
    H = PulseParams(
        name="H",
        pulse_obj=[RZ, RY],
    )

    # Rot = PulseParams(name=Gates.Rot, pulse_obj=[RZ, RY, RZ])
    CX = PulseParams(name="CX", pulse_obj=[H, CZ, H])
    CY = PulseParams(name="CY", pulse_obj=[RZ, CX, RZ])

    CRX = PulseParams(name="CRX", pulse_obj=[RZ, RY, CX, RY, CX, RZ])
    CRY = PulseParams(name="CRY", pulse_obj=[RY, CX, RY, CX])
    CRZ = PulseParams(name="CRZ", pulse_obj=[RZ, CX, RZ, CX])

    Rot = PulseParams(name="Rot", pulse_obj=[RZ, RY, RZ])

    unique_gate_set = [
        RX,
        RY,
        RZ,
        CZ,
    ]

    @staticmethod
    def gate_by_name(gate):
        if isinstance(gate, str):
            return getattr(PulseInformation, gate, None)
        else:
            return getattr(PulseInformation, gate.__name__, None)

    @staticmethod
    def num_params(gate):
        return len(PulseInformation.gate_by_name(gate))

    @staticmethod
    def update_params(path=f"{os.getcwd()}/qml_essentials/qoc_results.csv"):
        if os.path.isfile(path):
            log.info(f"Loading optimized pulses from {path}")
            with open(path, "r") as f:
                reader = csv.reader(f)

                for row in reader:
                    log.debug(
                        f"Loading optimized pulses for {row[0]}\
                            (Fidelity: {float(row[1]):.5f}): {row[2:]}"
                    )
                    PulseInformation.OPTIMIZED_PULSES[row[0]] = np.array(
                        [float(x) for x in row[2:]]
                    )
        else:
            log.error(f"No optimized pulses found at {path}")

    @staticmethod
    def shuffle_params(random_key):
        log.info(
            f"Shuffling optimized pulses with random key {random_key}\
              of gates {PulseInformation.unique_gate_set}"
        )
        for gate in PulseInformation.unique_gate_set:
            random_key, sub_key = safe_random_split(random_key)
            gate.params = jax.random.uniform(sub_key, (len(gate),))


class PulseGates:
    """
    Pulse-level implementations of quantum gates.

    Implements quantum gates using time-dependent Hamiltonians and pulse
    sequences, following the approach from https://doi.org/10.5445/IR/1000184129.
    Gates are decomposed using shaped Gaussian pulses with carrier modulation.

    Attributes:
        omega_q (float): Qubit frequency (10π).
        omega_c (float): Carrier frequency (10π).
        H_static (np.ndarray): Static Hamiltonian in qubit rotating frame.
        Id, X, Y, Z (np.ndarray): Pauli matrices for gate construction.
    """

    # NOTE: Implementation of S, RX, RY, RZ, CZ, CNOT/CX and H pulse level
    #   gates closely follow https://doi.org/10.5445/IR/1000184129
    # TODO: Mention deviations from the above?
    omega_q = 10 * np.pi
    omega_c = 10 * np.pi

    H_static = np.array([[np.exp(1j * omega_q / 2), 0], [0, np.exp(-1j * omega_q / 2)]])

    Id = np.eye(2, dtype=np.complex64)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    @staticmethod
    def _S(
        p: Union[List[float], np.ndarray],
        t: Union[float, List[float], np.ndarray],
        phi_c: float,
    ) -> np.ndarray:
        """
        Generate shaped Gaussian pulse envelope with carrier modulation.

        Internal helper function for creating time-dependent pulse shapes
        used in rotation gates. Not intended for direct circuit use.

        Args:
            p (Union[List[float], np.ndarray]): Pulse parameters [A, sigma]:
                - A (float): Amplitude of the Gaussian envelope
                - sigma (float): Width (standard deviation) of the Gaussian
            t (Union[float, List[float], np.ndarray]): Time or time interval
                for pulse application. If sequence, center is computed as midpoint.
            phi_c (float): Phase offset for the cosine carrier.

        Returns:
            np.ndarray: Shaped pulse amplitude at time(s) t.
        """
        A, sigma = p
        t_c = (t[0] + t[1]) / 2 if isinstance(t, (list, tuple)) else t / 2

        f = A * np.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        x = np.cos(PulseGates.omega_c * t + phi_c)

        return f * x

    @staticmethod
    def Rot(
        phi: float,
        theta: float,
        omega: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[np.ndarray] = None,
    ) -> None:
        """
        Apply general single-qubit rotation using pulse decomposition.

        Decomposes a general rotation into RZ(phi) · RY(theta) · RZ(omega)
        and applies each component using pulse-level implementations.

        Args:
            phi (float): First rotation angle.
            theta (float): Second rotation angle.
            omega (float): Third rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[np.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gates are applied in-place to the circuit.
        """
        params_RZ_1, params_RY, params_RZ_2 = PulseInformation.Rot.split_params(
            pulse_params
        )

        PulseGates.RZ(phi, wires=wires, pulse_params=params_RZ_1)
        PulseGates.RY(theta, wires=wires, pulse_params=params_RY)
        PulseGates.RZ(omega, wires=wires, pulse_params=params_RZ_2)

    @staticmethod
    def RX(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[np.ndarray] = None,
    ) -> None:
        """
        Apply X-axis rotation using pulse-level implementation.

        Implements RX rotation using a shaped Gaussian pulse with optimized
        envelope parameters.

        Args:
            w (float): Rotation angle in radians.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[np.ndarray]): Array containing pulse parameters
                [A, sigma, t] for the Gaussian envelope. If None, uses optimized
                parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        pulse_params = PulseInformation.RX.split_params(pulse_params)

        def Sx(p, t):
            return PulseGates._S(p, t, phi_c=np.pi) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.X @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sx * _H

        qml.evolve(H_eff)([pulse_params[0:2]], pulse_params[2])

    @staticmethod
    def RY(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[np.ndarray] = None,
    ) -> None:
        """
        Apply Y-axis rotation using pulse-level implementation.

        Implements RY rotation using a shaped Gaussian pulse with optimized
        envelope parameters.

        Args:
            w (float): Rotation angle in radians.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[np.ndarray]): Array containing pulse parameters
                [A, sigma, t] for the Gaussian envelope. If None, uses optimized
                parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        pulse_params = PulseInformation.RY.split_params(pulse_params)

        def Sy(p, t):
            return PulseGates._S(p, t, phi_c=-np.pi / 2) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.Y @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sy * _H

        qml.evolve(H_eff)([pulse_params[0:2]], pulse_params[2])

    @staticmethod
    def RZ(
        w: float, wires: Union[int, List[int]], pulse_params: Optional[float] = None
    ) -> None:
        """
        Apply Z-axis rotation using pulse-level implementation.

        Implements RZ rotation using virtual Z rotations (phase tracking)
        without physical pulse application.

        Args:
            w (float): Rotation angle in radians.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[float]): Duration parameter for the pulse.
                Rotation angle = w * 2 * pulse_params. Defaults to 0.5 if None.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        pulse_params = PulseInformation.RZ.split_params(pulse_params)

        _H = qml.Hermitian(PulseGates.Z, wires=wires)

        def Sz(p, t):
            return p * w

        H_eff = Sz * _H

        qml.evolve(H_eff)([pulse_params], 1)

    @staticmethod
    def H(
        wires: Union[int, List[int]], pulse_params: Optional[np.ndarray] = None
    ) -> None:
        """
        Apply Hadamard gate using pulse decomposition.

        Implements Hadamard as RZ(π) · RY(π/2) with a correction phase,
        using pulse-level implementations for each component.

        Args:
            wires (Union[int, List[int]]): Qubit index or indices to apply gate to.
            pulse_params (Optional[np.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        pulse_params_RZ, pulse_params_RY = PulseInformation.H.split_params(pulse_params)

        # qml.GlobalPhase(-np.pi / 2)  # this could act as substitute to Sc
        PulseGates.RZ(np.pi, wires=wires, pulse_params=pulse_params_RZ)
        PulseGates.RY(np.pi / 2, wires=wires, pulse_params=pulse_params_RY)

        def Sc(p, t):
            return -1.0

        _H = np.pi / 2 * np.eye(2, dtype=np.complex64)
        _H = qml.Hermitian(_H, wires=wires)
        H_corr = Sc * _H

        qml.evolve(H_corr)([0], 1)

    @staticmethod
    def CX(wires: List[int], pulse_params: Optional[np.ndarray] = None) -> None:
        """
        Apply CNOT gate using pulse decomposition.

        Implements CNOT as H_target · CZ · H_target, where H and CZ are
        applied using their respective pulse-level implementations.

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[np.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        params_H_1, params_CZ, params_H_2 = PulseInformation.CX.split_params(
            pulse_params
        )

        target = wires[1]

        PulseGates.H(wires=target, pulse_params=params_H_1)
        PulseGates.CZ(wires=wires, pulse_params=params_CZ)
        PulseGates.H(wires=target, pulse_params=params_H_2)

    @staticmethod
    def CY(wires: List[int], pulse_params: Optional[np.ndarray] = None) -> None:
        """
        Apply controlled-Y gate using pulse decomposition.

        Implements CY as RZ(-π/2)_target · CX · RZ(π/2)_target using
        pulse-level implementations.

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[np.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        params_RZ_1, params_CX, params_RZ_2 = PulseInformation.CY.split_params(
            pulse_params
        )

        target = wires[1]

        PulseGates.RZ(-np.pi / 2, wires=target, pulse_params=params_RZ_1)
        PulseGates.CX(wires=wires, pulse_params=params_CX)
        PulseGates.RZ(np.pi / 2, wires=target, pulse_params=params_RZ_2)

    @staticmethod
    def CZ(wires: List[int], pulse_params: Optional[float] = None) -> None:
        """
        Apply controlled-Z gate using pulse-level implementation.

        Implements CZ using a two-qubit interaction Hamiltonian based on
        ZZ coupling.

        Args:
            wires (List[int]): Control and target qubit indices.
            pulse_params (Optional[float]): Time or duration parameter for
                the pulse evolution. If None, uses optimized value.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        if pulse_params is None:
            pulse_params = PulseInformation.CZ.params
        else:
            pulse_params = pulse_params

        I_I = np.kron(PulseGates.Id, PulseGates.Id)
        Z_I = np.kron(PulseGates.Z, PulseGates.Id)
        I_Z = np.kron(PulseGates.Id, PulseGates.Z)
        Z_Z = np.kron(PulseGates.Z, PulseGates.Z)

        def Scz(p, t):
            return p * np.pi

        _H = (np.pi / 4) * (I_I - Z_I - I_Z + Z_Z)
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Scz * _H

        qml.evolve(H_eff)([pulse_params], 1)

    @staticmethod
    def CRX(
        w: float, wires: List[int], pulse_params: Optional[np.ndarray] = None
    ) -> None:
        """
        Apply controlled-RX gate using pulse decomposition.

        Implements CRX(w) as RZ(π/2) · RY(w/2) · CX · RY(-w/2) · CX · RZ(-π/2)
        applied to the target qubit, following arXiv:2408.01036.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[np.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        params_RZ_1, params_RY, params_CX_1, params_RY_2, params_CX_2, params_RZ_2 = (
            PulseInformation.CRX.split_params(pulse_params)
        )

        target = wires[1]

        PulseGates.RZ(np.pi / 2, wires=target, pulse_params=params_RZ_1)
        PulseGates.RY(w / 2, wires=target, pulse_params=params_RY)
        PulseGates.CX(wires=wires, pulse_params=params_CX_1)
        PulseGates.RY(-w / 2, wires=target, pulse_params=params_RY_2)
        PulseGates.CX(wires=wires, pulse_params=params_CX_2)
        PulseGates.RZ(-np.pi / 2, wires=target, pulse_params=params_RZ_2)

    @staticmethod
    def CRY(
        w: float, wires: List[int], pulse_params: Optional[np.ndarray] = None
    ) -> None:
        """
        Apply controlled-RY gate using pulse decomposition.

        Implements CRY(w) as RY(w/2) · CX · RY(-w/2) · CX applied to the
        target qubit, following arXiv:2408.01036.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[np.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        params_RY_1, params_CX_1, params_RY_2, params_CX_2 = (
            PulseInformation.CRY.split_params(pulse_params)
        )

        target = wires[1]

        PulseGates.RY(w / 2, wires=target, pulse_params=params_RY_1)
        PulseGates.CX(wires=wires, pulse_params=params_CX_1)
        PulseGates.RY(-w / 2, wires=target, pulse_params=params_RY_2)
        PulseGates.CX(wires=wires, pulse_params=params_CX_2)

    @staticmethod
    def CRZ(
        w: float, wires: List[int], pulse_params: Optional[np.ndarray] = None
    ) -> None:
        """
        Apply controlled-RZ gate using pulse decomposition.

        Implements CRZ(w) as RZ(w/2) · CX · RZ(-w/2) · CX applied to the
        target qubit, following arXiv:2408.01036.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[np.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        params_RZ_1, params_CX_1, params_RZ_2, params_CX_2 = (
            PulseInformation.CRZ.split_params(pulse_params)
        )

        target = wires[1]

        PulseGates.RZ(w / 2, wires=target, pulse_params=params_RZ_1)
        PulseGates.CX(wires=wires, pulse_params=params_CX_1)
        PulseGates.RZ(-w / 2, wires=target, pulse_params=params_RZ_2)
        PulseGates.CX(wires=wires, pulse_params=params_CX_2)


# Meta class to avoid instantiating the Gates class
class GatesMeta(type):
    def __getattr__(cls, gate_name):
        def handler(*args, **kwargs):
            return Gates._inner_getattr(gate_name, *args, **kwargs)

        # Dirty way to preserve information about the gate name
        handler.__name__ = gate_name
        return handler


class Gates(metaclass=GatesMeta):
    """
    Dynamic accessor for quantum Gates.

    Routes calls like `Gates.RX(...)` to either `UnitaryGates` or `PulseGates`
    depending on the `gate_mode` keyword (defaults to 'unitary').

    During circuit building, the pulse manager can be activated via
    `pulse_manager_context`, which slices the global model pulse parameters
    and passes them to each gate. Model pulse parameters act as element-wise
    scalers on the gate's optimized pulse parameters.

    Parameters
    ----------
    gate_mode : str, optional
        Determines the backend. 'unitary' for UnitaryGates, 'pulse' for PulseGates.
        Defaults to 'unitary'.

    Examples
    --------
    >>> Gates.RX(w, wires)
    >>> Gates.RX(w, wires, gate_mode="unitary")
    >>> Gates.RX(w, wires, gate_mode="pulse")
    >>> Gates.RX(w, wires, pulse_params, gate_mode="pulse")
    """

    def __getattr__(self, gate_name):
        def handler(**kwargs):
            return self._inner_getattr(gate_name, **kwargs)

        return handler

    @staticmethod
    def _inner_getattr(gate_name, *args, **kwargs):
        gate_mode = kwargs.pop("gate_mode", "unitary")

        # Backend selection and kwargs filtering
        allowed_args = ["w", "wires", "phi", "theta", "omega"]
        if gate_mode == "unitary":
            gate_backend = UnitaryGates
            allowed_args += ["noise_params", "random_key"]
        elif gate_mode == "pulse":
            gate_backend = PulseGates
            allowed_args += ["pulse_params"]
        else:
            raise ValueError(
                f"Unknown gate mode: {gate_mode}. Use 'unitary' or 'pulse'."
            )

        if len(kwargs.keys() - allowed_args) > 0:
            # TODO: pulse params are always provided?
            log.debug(
                f"Unsupported keyword arguments: {list(kwargs.keys() - allowed_args)}"
            )

        kwargs = {k: v for k, v in kwargs.items() if k in allowed_args}
        pulse_params = kwargs.get("pulse_params")
        pulse_mgr = getattr(Gates, "_pulse_mgr", None)

        # TODO: rework this part to convert to valid PulseParams earlier
        # Type check on pulse parameters
        if pulse_params is not None:
            # flatten pulse parameters
            if isinstance(pulse_params, (list, tuple)):
                flat_params = pulse_params

            elif isinstance(pulse_params, jax.core.Tracer):
                flat_params = np.ravel(pulse_params)

            elif isinstance(pulse_params, (np.ndarray, np.ndarray)):
                flat_params = pulse_params.flatten().tolist()
            elif isinstance(pulse_params, PulseParams):
                # extract the params in case a full object is given
                kwargs["pulse_params"] = pulse_params.params
                flat_params = pulse_params.params.flatten().tolist()

            else:
                raise TypeError(f"Unsupported pulse_params type: {type(pulse_params)}")

            # checks elements in flat parameters are real numbers or jax Tracer
            if not all(
                isinstance(x, (numbers.Real, jax.core.Tracer)) for x in flat_params
            ):
                raise TypeError(
                    "All elements in pulse_params must be int or float, "
                    f"got {pulse_params}, type {type(pulse_params)}. "
                )

        # Len check on pulse parameters
        if pulse_params is not None and not isinstance(pulse_mgr, PulseParamManager):
            n_params = PulseInformation.gate_by_name(gate_name).size
            if len(flat_params) != n_params:
                raise ValueError(
                    f"Gate '{gate_name}' expects {n_params} pulse parameters, "
                    f"got {len(flat_params)}"
                )

        # Pulse slicing + scaling
        if gate_mode == "pulse" and isinstance(pulse_mgr, PulseParamManager):
            n_params = PulseInformation.gate_by_name(gate_name).size
            scalers = pulse_mgr.get(n_params)
            base = PulseInformation.gate_by_name(gate_name).params
            kwargs["pulse_params"] = base * scalers

        # Call the selected gate backend
        gate = getattr(gate_backend, gate_name, None)
        if gate is None:
            raise AttributeError(
                f"'{gate_backend.__class__.__name__}' object "
                f"has no attribute '{gate_name}'"
            )

        return gate(*args, **kwargs)

    @staticmethod
    @contextmanager
    def pulse_manager_context(pulse_params: np.ndarray):
        """Temporarily set the global pulse manager for circuit building."""
        Gates._pulse_mgr = PulseParamManager(pulse_params)
        try:
            yield
        finally:
            Gates._pulse_mgr = None

    @staticmethod
    def parse_gates(
        gates: Union[str, Callable, List[Union[str, Callable]]],
        set_of_gates=None,
    ):
        set_of_gates = set_of_gates or Gates

        if isinstance(gates, str):
            # if str, use the pennylane fct
            parsed_gates = [getattr(set_of_gates, f"{gates}")]
        elif isinstance(gates, list):
            parsed_gates = []
            for enc in gates:
                # if list, check if str or callable
                if isinstance(enc, str):
                    parsed_Gates.append(getattr(set_of_gates, f"{enc}"))
                # check if callable
                elif callable(enc):
                    parsed_Gates.append(enc)
                else:
                    raise ValueError(
                        f"Operation {enc} is not a valid gate or callable.\
                        Got {type(enc)}"
                    )
        elif callable(gates):
            # default to callable
            parsed_gates = [gates]
        elif gates is None:
            parsed_gates = [lambda *args, **kwargs: None]
        else:
            raise ValueError(
                f"Operation {gates} is not a valid gate or callable or list of both."
            )
        return parsed_gates

    @staticmethod
    def is_rotational(gate):
        return gate.__name__ in [
            "RX",
            "RY",
            "RZ",
            "Rot",
            "CRX",
            "CRY",
            "CRZ",
        ]

    @staticmethod
    def is_entangling(gate):
        return gate.__name__ in ["CX", "CY", "CZ", "CRX", "CRY", "CRZ"]


class PulseParamManager:
    def __init__(self, pulse_params: np.ndarray):
        self.pulse_params = pulse_params
        self.idx = 0

    def get(self, n: int):
        """Return the next n parameters and advance the cursor."""
        if self.idx + n > len(self.pulse_params):
            raise ValueError("Not enough pulse parameters left for this gate")
        # TODO: we squeeze here to get rid of any extra hidden dimension
        params = self.pulse_params[self.idx : self.idx + n].squeeze()
        self.idx += n
        return params
