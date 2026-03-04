import os
from typing import Optional, List, Union, Dict, Callable, Tuple
import numbers
import csv
import jax.numpy as jnp

from qml_essentials import operations as op
from qml_essentials import yaqsi as ys
import jax
import itertools
from contextlib import contextmanager
import logging

from qml_essentials.utils import safe_random_split

log = logging.getLogger(__name__)


class UnitaryGates:
    """Collection of unitary quantum gates with optional noise simulation."""

    batch_gate_error = True

    @staticmethod
    def NQubitDepolarizingChannel(p: float, wires: List[int]) -> op.QubitChannel:
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
            op.QubitChannel: QubitChannel with Kraus operators
                representing the depolarizing noise channel.

        Raises:
            ValueError: If p is not in [0, 1] or if fewer than 2 qubits provided.
        """

        def n_qubit_depolarizing_kraus(p: float, n: int) -> List[jnp.ndarray]:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability p must be between 0 and 1, got {p}")
            if n < 2:
                raise ValueError(f"Number of qubits must be >= 2, got {n}")

            Id = jnp.eye(2)
            X = op.PauliX._matrix
            Y = op.PauliY._matrix
            Z = op.PauliZ._matrix
            paulis = [Id, X, Y, Z]

            dim = 2**n
            all_ops = []

            # Generate all n-qubit Pauli tensor products:
            for indices in itertools.product(range(4), repeat=n):
                P = jnp.eye(1)
                for idx in indices:
                    P = jnp.kron(P, paulis[idx])
                all_ops.append(P)

            # Identity operator corresponds to all zeros indices (Id^n)
            K0 = jnp.sqrt(1 - p * (4**n - 1) / (4**n)) * jnp.eye(dim)

            kraus_ops = []
            for i, P in enumerate(all_ops):
                if i == 0:
                    # Skip the identity, already handled as K0
                    continue
                kraus_ops.append(jnp.sqrt(p / (4**n)) * P)

            return [K0] + kraus_ops

        return op.QubitChannel(n_qubit_depolarizing_kraus(p, len(wires)), wires=wires)

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
                    op.BitFlip(bf, wires=wire)

                pf = noise_params.get("PhaseFlip", 0.0)
                if pf > 0:
                    op.PhaseFlip(pf, wires=wire)

                dp = noise_params.get("Depolarizing", 0.0)
                if dp > 0:
                    op.DepolarizingChannel(dp, wires=wire)

            # noise on two-qubits
            if len(wires) > 1:
                p = noise_params.get("MultiQubitDepolarizing", 0.0)
                if p > 0:
                    UnitaryGates.NQubitDepolarizingChannel(p, wires)

    @staticmethod
    def GateError(
        w: Union[float, jnp.ndarray, List[float]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
        """
        Apply gate error noise to rotation angle(s).

        Adds Gaussian noise to gate rotation angles to simulate imperfect
        gate implementations.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Rotation angle(s) in radians.
            noise_params (Optional[Dict[str, float]]): Dictionary with optional
                "GateError" key specifying standard deviation of Gaussian noise.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for
                stochastic noise generation.

        Returns:
            Tuple[jnp.ndarray, jax.random.PRNGKey]: Tuple containing:
                - Modified rotation angle(s) with applied noise
                - Updated JAX random key

        Raises:
            AssertionError: If noise_params contains "GateError" but random_key is None.
        """
        if noise_params is not None and noise_params.get("GateError", None) is not None:
            assert (
                random_key is not None
            ), "A random_key must be provided when using GateError"

            if UnitaryGates.batch_gate_error:
                random_key, sub_key = safe_random_split(random_key)
            else:
                # Use a fixed key so that every batch element (under vmap)
                # draws the same noise value, effectively broadcasting.
                sub_key = jax.random.key(0)

            w += noise_params["GateError"] * jax.random.normal(
                sub_key,
                (
                    w.shape
                    if isinstance(w, jnp.ndarray) and UnitaryGates.batch_gate_error
                    else ()
                ),
            )
        return w, random_key

    @staticmethod
    def Rot(
        phi: Union[float, jnp.ndarray, List[float]],
        theta: Union[float, jnp.ndarray, List[float]],
        omega: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply general rotation gate with optional noise.

        Applies a three-angle rotation Rot(phi, theta, omega) with optional
        gate errors and noise channels.

        Args:
            phi (Union[float, jnp.ndarray, List[float]]): First rotation angle.
            theta (Union[float, jnp.ndarray, List[float]]): Second rotation angle.
            omega (Union[float, jnp.ndarray, List[float]]): Third rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
                Supports BitFlip, PhaseFlip, Depolarizing, and GateError.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        if noise_params is not None and "GateError" in noise_params:
            phi, random_key = UnitaryGates.GateError(phi, noise_params, random_key)
            theta, random_key = UnitaryGates.GateError(theta, noise_params, random_key)
            omega, random_key = UnitaryGates.GateError(omega, noise_params, random_key)
        op.Rot(phi, theta, omega, wires=wires, input_idx=False)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def PauliRot(
        theta: float,
        pauli: str,
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply general rotation gate with optional noise.

        Applies a three-angle rotation Rot(phi, theta, omega) with optional
        gate errors and noise channels.

        Args:
            theta (Union[float, jnp.ndarray, List[float]]): Second rotation angle.
            pauli (str): Pauli operator to apply. Must be "X", "Y", or "Z".
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
                Supports BitFlip, PhaseFlip, Depolarizing, and GateError.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        if noise_params is not None and "GateError" in noise_params:
            theta, random_key = UnitaryGates.GateError(theta, noise_params, random_key)
        op.PauliRot(theta, pauli, wires=wires, input_idx=input_idx)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RX(
        w: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply X-axis rotation with optional noise.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        op.RX(w, wires=wires, input_idx=input_idx)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RY(
        w: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply Y-axis rotation with optional noise.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        op.RY(w, wires=wires, input_idx=input_idx)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RZ(
        w: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply Z-axis rotation with optional noise.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        op.RZ(w, wires=wires, input_idx=input_idx)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRX(
        w: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply controlled X-rotation with optional noise.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        op.CRX(w, wires=wires, input_idx=input_idx)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRY(
        w: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply controlled Y-rotation with optional noise.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        op.CRY(w, wires=wires, input_idx=input_idx)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRZ(
        w: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply controlled Z-rotation with optional noise.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Rotation angle.
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        op.CRZ(w, wires=wires, input_idx=input_idx)
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
        op.CX(wires=wires)
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
        op.CY(wires=wires)
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
        op.CZ(wires=wires)
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
        op.H(wires=wires)
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
        _params (jnp.ndarray): Direct pulse parameters (leaf nodes only).
        _pulse_obj (List): Child PulseParams objects (composite nodes only).
    """

    def __init__(
        self,
        name: str = "",
        params: Optional[jnp.ndarray] = None,
        pulse_obj: Optional[List] = None,
    ) -> None:
        """
        Initialize pulse parameters container.

        Args:
            name (str): Name identifier for the gate. Defaults to empty string.
            params (Optional[jnp.ndarray]): Direct pulse parameters for leaf gates.
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

    def __getitem__(self, idx: int) -> Union[float, jnp.ndarray]:
        """
        Access pulse parameter(s) by index.

        For leaf gates, returns the parameter at the given index.
        For composite gates, returns parameters of the child at the given index.

        Args:
            idx (int): Index to access.

        Returns:
            Union[float, jnp.ndarray]: Parameter value or child parameters.
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
    def params(self) -> jnp.ndarray:
        """
        Get or compute pulse parameters.

        For leaf nodes, returns internal pulse parameters.
        For composite nodes, returns concatenated parameters from all children.

        Returns:
            jnp.ndarray: Pulse parameters array.
        """
        if self.is_leaf:
            return self._params

        params = self.split_params(params=None, leafs=False)

        return jnp.concatenate(params)

    @params.setter
    def params(self, value: jnp.ndarray) -> None:
        """
        Set pulse parameters.

        For leaf nodes, sets internal parameters directly.
        For composite nodes, distributes values across children.

        Args:
            value (jnp.ndarray): Pulse parameters to set.

        Raises:
            AssertionError: If value is not jnp.ndarray for leaf nodes.
        """
        if self.is_leaf:
            assert isinstance(value, jnp.ndarray), "params must be a jnp.ndarray"
            self._params = value
            return

        idx = 0
        for obj in self.childs:
            nidx = idx + obj.size
            obj.params = value[idx:nidx]
            idx = nidx

    @property
    def leaf_params(self) -> jnp.ndarray:
        """
        Get parameters from all leaf nodes.

        Returns:
            jnp.ndarray: Concatenated parameters from all leaf nodes.
        """
        if self.is_leaf:
            return self._params

        params = self.split_params(None, leafs=True)

        return jnp.concatenate(params)

    @leaf_params.setter
    def leaf_params(self, value: jnp.ndarray) -> None:
        """
        Set parameters for all leaf nodes.

        Args:
            value (jnp.ndarray): Parameters to distribute across leaf nodes.
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
        params: Optional[jnp.ndarray] = None,
        leafs: bool = False,
    ) -> List[jnp.ndarray]:
        """
        Split parameters into sub-arrays for children or leaves.

        Args:
            params (Optional[jnp.ndarray]): Parameters to split. If None,
                uses internal parameters.
            leafs (bool): If True, splits across leaf nodes; if False,
                splits across direct children. Defaults to False.

        Returns:
            List[jnp.ndarray]: List of parameter arrays for children or leaves.
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


class PulseEnvelope:
    """Registry of pulse envelope shapes.

    Each envelope is a pure function ``(p, t, t_c) -> amplitude`` that
    computes the pulse envelope *without* carrier modulation.  The carrier
    ``cos(omega_c * t + phi_c)`` is applied separately in the coefficient
    functions built by :meth:`build_coeff_fns`.

    Attributes:
        REGISTRY: Mapping from envelope name to metadata dict containing
            ``fn`` (callable), ``n_envelope_params`` (int), and per-gate
            default parameter arrays.
    """

    @staticmethod
    def gaussian(p, t, t_c):
        """Gaussian envelope. ``p = [A, sigma]``."""
        A, sigma = p[0], p[1]
        return A * jnp.exp(-0.5 * ((t - t_c) / sigma) ** 2)

    @staticmethod
    def square(p, t, t_c):
        """Rectangular envelope. ``p = [A, width]``."""
        A, width = p[0], p[1]
        return A * (jnp.abs(t - t_c) <= width / 2)

    @staticmethod
    def cosine(p, t, t_c):
        """Raised cosine envelope. ``p = [A, width]``."""
        A, width = p[0], p[1]
        x = jnp.clip((t - t_c) / width, -0.5, 0.5)
        return A * jnp.cos(jnp.pi * x)

    @staticmethod
    def drag(p, t, t_c):
        """DRAG (Derivative Removal by Adiabatic Gate). ``p = [A, sigma, beta]``."""
        A, sigma, beta = p[0], p[1], p[2]
        g = A * jnp.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        dg = g * (-(t - t_c) / sigma**2)
        return g + beta * dg

    @staticmethod
    def sech(p, t, t_c):
        """Hyperbolic secant envelope. ``p = [A, sigma]``."""
        A, sigma = p[0], p[1]
        return A / jnp.cosh((t - t_c) / sigma)

    # ``n_envelope_params`` counts only the envelope parameters (excluding
    # the evolution time ``t`` which is always the last element of the full
    # pulse parameter vector).
    REGISTRY = {
        "gaussian": {
            "fn": gaussian.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array(
                    [15.917705975121452, 29.72253135127928, 0.7551576189610215]
                ),
                "RY": jnp.array(
                    [7.855362198639627, 21.96253607741858, 1.100281557726808]
                ),
                "RZ": jnp.array([0.49999999899901876]),
                "CZ": jnp.array([0.3182752540123598]),
            },
        },
        "square": {
            "fn": square.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
                "RZ": jnp.array([0.5]),
                "CZ": jnp.array([0.318]),
            },
        },
        "cosine": {
            "fn": cosine.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
                "RZ": jnp.array([0.5]),
                "CZ": jnp.array([0.318]),
            },
        },
        "drag": {
            "fn": drag.__func__,
            "n_envelope_params": 3,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 0.1, 1.0]),
                "RY": jnp.array([1.0, 1.0, 0.1, 1.0]),
                "RZ": jnp.array([0.5]),
                "CZ": jnp.array([0.318]),
            },
        },
        "sech": {
            "fn": sech.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
                "RZ": jnp.array([0.5]),
                "CZ": jnp.array([0.318]),
            },
        },
    }

    @staticmethod
    def available() -> List[str]:
        """Return list of registered envelope names."""
        return list(PulseEnvelope.REGISTRY.keys())

    @staticmethod
    def get(name: str) -> dict:
        """Look up envelope metadata by name.

        Raises:
            ValueError: If *name* is not registered.
        """
        if name not in PulseEnvelope.REGISTRY:
            raise ValueError(
                f"Unknown pulse envelope '{name}'. "
                f"Available: {PulseEnvelope.available()}"
            )
        return PulseEnvelope.REGISTRY[name]

    @staticmethod
    def build_coeff_fns(envelope_fn, omega_c):
        """Build ``(coeff_Sx, coeff_Sy)`` for a given envelope function.

        Each returned function has a unique ``__code__`` object so that
        the yaqsi JIT solver cache (keyed on ``id(coeff_fn.__code__)``)
        assigns a separate compiled XLA program per envelope shape.

        The rotation angle ``w`` is expected as the **last** element of the
        parameter array ``p`` (i.e. ``p[-1]``).  Envelope parameters occupy
        ``p[:-1]`` (excluding the evolution-time element that is passed
        separately to ``ys.evolve``).

        Args:
            envelope_fn: Pure envelope function ``(p, t, t_c) -> scalar``.
            omega_c: Carrier frequency.

        Returns:
            Tuple of ``(coeff_Sx, coeff_Sy)``.
        """

        def _coeff_Sx(p, t):
            t_c = t / 2
            env = envelope_fn(p, t, t_c)
            carrier = jnp.cos(omega_c * t + jnp.pi)
            return env * carrier * p[-1]

        def _coeff_Sy(p, t):
            t_c = t / 2
            env = envelope_fn(p, t, t_c)
            carrier = jnp.cos(omega_c * t - jnp.pi / 2)
            return env * carrier * p[-1]

        return _coeff_Sx, _coeff_Sy


class PulseInformation:
    """Stores pulse parameter counts and optimized pulse parameters.

    Call :meth:`set_envelope` to switch the active pulse shape.  This
    rebuilds all :class:`PulseParams` trees so that parameter counts
    and defaults match the selected envelope.
    """

    _envelope: str = "gaussian"

    @classmethod
    def _build_leaf_gates(cls):
        """(Re-)create leaf PulseParams from the active envelope defaults."""
        defaults = PulseEnvelope.get(cls._envelope)["defaults"]
        cls.RX = PulseParams(name="RX", params=defaults["RX"])
        cls.RY = PulseParams(name="RY", params=defaults["RY"])
        cls.RZ = PulseParams(name="RZ", params=defaults["RZ"])
        cls.CZ = PulseParams(name="CZ", params=defaults["CZ"])

    @classmethod
    def _build_composite_gates(cls):
        """(Re-)create composite PulseParams trees from current leaves."""
        cls.H = PulseParams(name="H", pulse_obj=[cls.RZ, cls.RY])
        cls.CX = PulseParams(name="CX", pulse_obj=[cls.H, cls.CZ, cls.H])
        cls.CY = PulseParams(name="CY", pulse_obj=[cls.RZ, cls.CX, cls.RZ])
        cls.CRX = PulseParams(
            name="CRX", pulse_obj=[cls.RZ, cls.RY, cls.CX, cls.RY, cls.CX, cls.RZ]
        )
        cls.CRY = PulseParams(name="CRY", pulse_obj=[cls.RY, cls.CX, cls.RY, cls.CX])
        cls.CRZ = PulseParams(name="CRZ", pulse_obj=[cls.RZ, cls.CX, cls.RZ, cls.CX])
        cls.Rot = PulseParams(name="Rot", pulse_obj=[cls.RZ, cls.RY, cls.RZ])
        cls.unique_gate_set = [cls.RX, cls.RY, cls.RZ, cls.CZ]

    @classmethod
    def set_envelope(cls, name: str) -> None:
        """Switch pulse envelope and rebuild all PulseParams trees.

        Also updates the coefficient functions used by :class:`PulseGates`.

        Args:
            name: One of :meth:`PulseEnvelope.available`.
        """
        info = PulseEnvelope.get(name)  # validates name
        cls._envelope = name
        cls._build_leaf_gates()
        cls._build_composite_gates()

        # Rebuild coefficient functions on PulseGates
        coeff_Sx, coeff_Sy = PulseEnvelope.build_coeff_fns(
            info["fn"], PulseGates.omega_c
        )
        PulseGates._coeff_Sx = staticmethod(coeff_Sx)
        PulseGates._coeff_Sy = staticmethod(coeff_Sy)
        PulseGates._active_envelope = name

        log.info(f"Pulse envelope set to '{name}'")

    @classmethod
    def get_envelope(cls) -> str:
        """Return the name of the active pulse envelope."""
        return cls._envelope

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
                    PulseInformation.OPTIMIZED_PULSES[row[0]] = jnp.array(
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


# Initialise PulseInformation with default (gaussian) envelope
PulseInformation._build_leaf_gates()
PulseInformation._build_composite_gates()


class PulseGates:
    """Pulse-level implementations of quantum gates.

    Implements quantum gates using time-dependent Hamiltonians and pulse
    sequences, following the approach from https://doi.org/10.5445/IR/1000184129.
    The active pulse envelope is selected via
    :meth:`PulseInformation.set_envelope`.

    Attributes:
        omega_q: Qubit frequency (10π).
        omega_c: Carrier frequency (10π).
        _active_envelope: Name of the currently active envelope shape.
    """

    # NOTE: Implementation of S, RX, RY, RZ, CZ, CNOT/CX and H pulse level
    #   gates closely follow https://doi.org/10.5445/IR/1000184129
    omega_q = 10 * jnp.pi
    omega_c = 10 * jnp.pi

    H_static = jnp.array(
        [[jnp.exp(1j * omega_q / 2), 0], [0, jnp.exp(-1j * omega_q / 2)]]
    )

    Id = jnp.eye(2, dtype=jnp.complex64)
    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])

    # Pre-computed interaction-picture Hamiltonians (H_static† @ P @ H_static).
    _H_X = H_static.conj().T @ X @ H_static
    _H_Y = H_static.conj().T @ Y @ H_static

    # Pre-computed CZ Hamiltonian: (π/4)(I⊗I - Z⊗I - I⊗Z + Z⊗Z)
    _H_CZ = (jnp.pi / 4) * (
        jnp.kron(Id, Id) - jnp.kron(Z, Id) - jnp.kron(Id, Z) + jnp.kron(Z, Z)
    )

    # Pre-computed H correction Hamiltonian: (π/2) I
    _H_corr = jnp.pi / 2 * jnp.eye(2, dtype=jnp.complex64)

    _active_envelope: str = "gaussian"

    @staticmethod
    def _coeff_Sx(p, t):
        """Coefficient function for RX pulse (active envelope)."""
        t_c = t / 2
        env = PulseEnvelope.gaussian(p, t, t_c)
        carrier = jnp.cos(PulseGates.omega_c * t + jnp.pi)
        return env * carrier * p[-1]

    @staticmethod
    def _coeff_Sy(p, t):
        """Coefficient function for RY pulse (active envelope)."""
        t_c = t / 2
        env = PulseEnvelope.gaussian(p, t, t_c)
        carrier = jnp.cos(PulseGates.omega_c * t - jnp.pi / 2)
        return env * carrier * p[-1]

    @staticmethod
    def _coeff_Sz(p, t):
        """Coefficient function for RZ pulse: p * w."""
        return p[0] * p[1]

    @staticmethod
    def _coeff_Sc(p, t):
        """Constant coefficient for H correction phase."""
        return -1.0

    @staticmethod
    def _coeff_Scz(p, t):
        """Coefficient function for CZ pulse."""
        return p * jnp.pi

    @staticmethod
    def Rot(
        phi: float,
        theta: float,
        omega: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
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
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
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
    def PauliRot(
        pauli: str,
        theta: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
    ) -> None:
        """
        Apply Pauli rotation using pulse-level implementation.

        Implements Pauli rotation using a shaped Gaussian pulse with optimized
        envelope parameters.

        Args:
            pauli (str): Pauli string (X, Y, Z).
            theta (float): Rotation angle in radians.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[jnp.ndarray]): Array containing pulse parameters
                [A, sigma, t] for the Gaussian envelope. If None, uses optimized
                parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        raise NotImplementedError("PauliRot gate is not implemented as PulseGate")

    @staticmethod
    def RX(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
    ) -> None:
        """Apply X-axis rotation using the active pulse envelope.

        Args:
            w: Rotation angle in radians.
            wires: Qubit index or indices.
            pulse_params: Envelope parameters ``[env_0, ..., env_n, t]``.
                If ``None``, uses optimized defaults.
        """
        pulse_params = PulseInformation.RX.split_params(pulse_params)

        _H = op.Hermitian(PulseGates._H_X, wires=wires, record=False)
        H_eff = PulseGates._coeff_Sx * _H

        # Pack: [envelope_params..., w] — evolution time is the last element
        # of pulse_params (pulse_params[-1]).
        env_params = jnp.array([*pulse_params[:-1], w])
        ys.evolve(H_eff)([env_params], pulse_params[-1])

    @staticmethod
    def RY(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
    ) -> None:
        """Apply Y-axis rotation using the active pulse envelope.

        Args:
            w: Rotation angle in radians.
            wires: Qubit index or indices.
            pulse_params: Envelope parameters ``[env_0, ..., env_n, t]``.
                If ``None``, uses optimized defaults.
        """
        pulse_params = PulseInformation.RY.split_params(pulse_params)

        _H = op.Hermitian(PulseGates._H_Y, wires=wires, record=False)
        H_eff = PulseGates._coeff_Sy * _H

        # Pack w into the params so the coefficient function doesn't need
        # a closure — this enables JIT solver cache sharing across all RY calls.
        env_params = jnp.array([*pulse_params[:-1], w])
        ys.evolve(H_eff)([env_params], pulse_params[-1])

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

        _H = op.Hermitian(PulseGates.Z, wires=wires, record=False)
        H_eff = PulseGates._coeff_Sz * _H

        # Pack w into the params so the coefficient function doesn't need
        # a closure — [pulse_param_scalar, w] enables JIT solver cache sharing.
        # pulse_params may be a 1-element array or scalar; ravel + index to
        # ensure a scalar for concatenation.
        pp_scalar = jnp.ravel(jnp.asarray(pulse_params))[0]
        ys.evolve(H_eff)([jnp.array([pp_scalar, w])], 1)

    @staticmethod
    def H(
        wires: Union[int, List[int]], pulse_params: Optional[jnp.ndarray] = None
    ) -> None:
        """
        Apply Hadamard gate using pulse decomposition.

        Implements Hadamard as RZ(π) · RY(π/2) with a correction phase,
        using pulse-level implementations for each component.

        Args:
            wires (Union[int, List[int]]): Qubit index or indices to apply gate to.
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        pulse_params_RZ, pulse_params_RY = PulseInformation.H.split_params(pulse_params)

        # qml.GlobalPhase(-jnp.pi / 2)  # this could act as substitute to Sc
        PulseGates.RZ(jnp.pi, wires=wires, pulse_params=pulse_params_RZ)
        PulseGates.RY(jnp.pi / 2, wires=wires, pulse_params=pulse_params_RY)

        _H = op.Hermitian(PulseGates._H_corr, wires=wires, record=False)
        H_corr = PulseGates._coeff_Sc * _H

        ys.evolve(H_corr)([0], 1)

    @staticmethod
    def CX(wires: List[int], pulse_params: Optional[jnp.ndarray] = None) -> None:
        """
        Apply CNOT gate using pulse decomposition.

        Implements CNOT as H_target · CZ · H_target, where H and CZ are
        applied using their respective pulse-level implementations.

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
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
    def CY(wires: List[int], pulse_params: Optional[jnp.ndarray] = None) -> None:
        """
        Apply controlled-Y gate using pulse decomposition.

        Implements CY as RZ(-π/2)_target · CX · RZ(π/2)_target using
        pulse-level implementations.

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        params_RZ_1, params_CX, params_RZ_2 = PulseInformation.CY.split_params(
            pulse_params
        )

        target = wires[1]

        PulseGates.RZ(-jnp.pi / 2, wires=target, pulse_params=params_RZ_1)
        PulseGates.CX(wires=wires, pulse_params=params_CX)
        PulseGates.RZ(jnp.pi / 2, wires=target, pulse_params=params_RZ_2)

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

        _H = op.Hermitian(PulseGates._H_CZ, wires=wires, record=False)
        H_eff = PulseGates._coeff_Scz * _H

        ys.evolve(H_eff)([pulse_params], 1)

    @staticmethod
    def CRX(
        w: float, wires: List[int], pulse_params: Optional[jnp.ndarray] = None
    ) -> None:
        """
        Apply controlled-RX gate using pulse decomposition.

        Implements CRX(w) as RZ(π/2) · RY(w/2) · CX · RY(-w/2) · CX · RZ(-π/2)
        applied to the target qubit, following arXiv:2408.01036.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        params_RZ_1, params_RY, params_CX_1, params_RY_2, params_CX_2, params_RZ_2 = (
            PulseInformation.CRX.split_params(pulse_params)
        )

        target = wires[1]

        PulseGates.RZ(jnp.pi / 2, wires=target, pulse_params=params_RZ_1)
        PulseGates.RY(w / 2, wires=target, pulse_params=params_RY)
        PulseGates.CX(wires=wires, pulse_params=params_CX_1)
        PulseGates.RY(-w / 2, wires=target, pulse_params=params_RY_2)
        PulseGates.CX(wires=wires, pulse_params=params_CX_2)
        PulseGates.RZ(-jnp.pi / 2, wires=target, pulse_params=params_RZ_2)

    @staticmethod
    def CRY(
        w: float, wires: List[int], pulse_params: Optional[jnp.ndarray] = None
    ) -> None:
        """
        Apply controlled-RY gate using pulse decomposition.

        Implements CRY(w) as RY(w/2) · CX · RY(-w/2) · CX applied to the
        target qubit, following arXiv:2408.01036.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
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
        w: float, wires: List[int], pulse_params: Optional[jnp.ndarray] = None
    ) -> None:
        """
        Apply controlled-RZ gate using pulse decomposition.

        Implements CRZ(w) as RZ(w/2) · CX · RZ(-w/2) · CX applied to the
        target qubit, following arXiv:2408.01036.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
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
        allowed_args = ["w", "wires", "phi", "theta", "omega", "input_idx"]
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
                flat_params = jnp.ravel(pulse_params)

            elif isinstance(pulse_params, (jnp.ndarray, jnp.ndarray)):
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
    def pulse_manager_context(pulse_params: jnp.ndarray):
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
                    parsed_gates.append(getattr(set_of_gates, f"{enc}"))
                # check if callable
                elif callable(enc):
                    parsed_gates.append(enc)
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
    def __init__(self, pulse_params: jnp.ndarray):
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
