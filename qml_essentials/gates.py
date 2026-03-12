import os
from dataclasses import dataclass
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
from qml_essentials.tape import active_pulse_tape

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


@dataclass
class DecompositionStep:
    """One step in a composite pulse gate decomposition.

    Attributes:
        gate: Child PulseParams object for this step.
        wire_fn: Wire selection — ``"all"``, ``"target"``, or ``"control"``.
        angle_fn: Maps parent angle(s) ``w`` to child angle.
            ``None`` means pass ``w`` through unchanged.
    """

    gate: "PulseParams"
    wire_fn: str = "all"
    angle_fn: Optional[Callable] = None


class PulseParams:
    """Container for hierarchical pulse parameters.

    Leaf nodes hold direct parameters; composite nodes hold a list of
    :class:`DecompositionStep` objects that describe how the gate is
    built from simpler gates.

    Attributes:
        name: Gate identifier (e.g. ``"RX"``, ``"H"``).
        decomposition: List of :class:`DecompositionStep` (composite only).
    """

    def __init__(
        self,
        name: str = "",
        params: Optional[jnp.ndarray] = None,
        decomposition: Optional[List[DecompositionStep]] = None,
    ) -> None:
        """
        Args:
            name: Gate name.
            params: Direct pulse parameters (leaf gates).
                Mutually exclusive with *decomposition*.
            decomposition: List of :class:`DecompositionStep` (composite gates).
                Mutually exclusive with *params*.
        """
        assert (params is None) != (
            decomposition is None
        ), "Exactly one of `params` or `decomposition` must be provided."

        self.decomposition = decomposition
        # Derive _pulse_obj for backward compat with childs/leafs/split_params
        self._pulse_obj = (
            [step.gate for step in decomposition] if decomposition else None
        )

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
        """DRAG (Derivative Removal by Adiabatic Gate). ``p = [A, beta, sigma]``."""
        A, beta, sigma = p[0], p[1], p[2]
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
                    [17.13676044364824, 0.8979891072359009, 0.7494982236190029]
                    # 16.09481118093026, 1.6973130876690965, 0.7499052005559208
                ),
                "RY": jnp.array(
                    [8.509443822118781, 0.9565890425279769, 1.0977437900953921]
                    # 8.012252123237978, 1.94827419290548, 1.0989055608209761
                ),
            },
        },
        "square": {
            "fn": square.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
            },
        },
        "cosine": {
            "fn": cosine.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
            },
        },
        "drag": {
            "fn": drag.__func__,
            "n_envelope_params": 3,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 0.1, 1.0]),
                "RY": jnp.array([1.0, 1.0, 0.1, 1.0]),
            },
        },
        "sech": {
            "fn": sech.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
            },
        },
        "general": {
            "fn": None,
            "n_envelope_params": 0,
            "defaults": {
                "RZ": jnp.array([0.4999999689207045]),
                # 0.49999999899901876
                "CZ": jnp.array([0.9550217449349658]),
                # 0.3182752540123598
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
        general = PulseEnvelope.get("general")["defaults"]

        cls.RX = PulseParams(name="RX", params=defaults["RX"])
        cls.RY = PulseParams(name="RY", params=defaults["RY"])

        cls.RZ = PulseParams(name="RZ", params=general["RZ"])
        cls.CZ = PulseParams(name="CZ", params=general["CZ"])

    @classmethod
    def _build_composite_gates(cls):
        """(Re-)create composite PulseParams trees from current leaves."""
        cls.H = PulseParams(
            name="H",
            decomposition=[
                DecompositionStep(cls.RZ, "all", lambda w: jnp.pi),
                DecompositionStep(cls.RY, "all", lambda w: jnp.pi / 2),
            ],
        )
        cls.CX = PulseParams(
            name="CX",
            decomposition=[
                DecompositionStep(cls.H, "target", lambda w: 0.0),
                DecompositionStep(cls.CZ, "all", lambda w: 0.0),
                DecompositionStep(cls.H, "target", lambda w: 0.0),
            ],
        )
        cls.CY = PulseParams(
            name="CY",
            decomposition=[
                DecompositionStep(cls.RZ, "target", lambda w: -jnp.pi / 2),
                DecompositionStep(cls.CX, "all"),
                DecompositionStep(cls.RZ, "target", lambda w: jnp.pi / 2),
            ],
        )
        cls.CRX = PulseParams(
            name="CRX",
            decomposition=[
                DecompositionStep(cls.RZ, "target", lambda w: jnp.pi / 2),
                DecompositionStep(cls.RY, "target", lambda w: w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RY, "target", lambda w: -w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RZ, "target", lambda w: -jnp.pi / 2),
            ],
        )
        cls.CRY = PulseParams(
            name="CRY",
            decomposition=[
                DecompositionStep(cls.RY, "target", lambda w: w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RY, "target", lambda w: -w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
            ],
        )
        cls.CRZ = PulseParams(
            name="CRZ",
            decomposition=[
                DecompositionStep(cls.RZ, "target", lambda w: w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RZ, "target", lambda w: -w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
            ],
        )
        cls.Rot = PulseParams(
            name="Rot",
            decomposition=[
                DecompositionStep(cls.RZ, "all", lambda w: w[0]),
                DecompositionStep(cls.RY, "all", lambda w: w[1]),
                DecompositionStep(cls.RZ, "all", lambda w: w[2]),
            ],
        )
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
    def _record_pulse_event(gate_name, w, wires, pulse_params, parent=None):
        """Append a PulseEvent to the active pulse tape if recording.

        This is called from leaf gate methods (RX, RY, RZ, CZ) so that
        :func:`~qml_essentials.tape.pulse_recording` can collect events
        without the caller needing to know about the tape.
        """
        ptape = active_pulse_tape()
        if ptape is None:
            return

        from qml_essentials.drawing import PulseEvent, LEAF_META

        meta = LEAF_META.get(gate_name, {})
        wires_list = [wires] if isinstance(wires, int) else list(wires)

        if meta.get("physical", False):
            info = PulseEnvelope.get(PulseInformation.get_envelope())
            pp = PulseInformation.gate_by_name(gate_name).split_params(pulse_params)
            env_p = pp[:-1]
            dur = float(pp[-1])
            ptape.append(
                PulseEvent(
                    gate=gate_name,
                    wires=wires_list,
                    envelope_fn=info["fn"],
                    envelope_params=jnp.array(env_p),
                    w=float(w),
                    duration=dur,
                    carrier_phase=meta["carrier_phase"],
                    parent=parent,
                )
            )
        else:
            pp = PulseInformation.gate_by_name(gate_name).split_params(pulse_params)
            ptape.append(
                PulseEvent(
                    gate=gate_name,
                    wires=wires_list,
                    envelope_fn=None,
                    envelope_params=jnp.ravel(jnp.asarray(pp)),
                    w=float(w) if not isinstance(w, list) else 0.0,
                    duration=1.0,
                    carrier_phase=0.0,
                    parent=parent,
                )
            )

    @staticmethod
    def Rot(
        phi: float,
        theta: float,
        omega: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply general rotation via decomposition: RZ(phi) · RY(theta) · RZ(omega).

        Args:
            phi (float): First rotation angle.
            theta (float): Second rotation angle.
            omega (float): Third rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility

        Returns:
            None: Gates are applied in-place to the circuit.
        """
        if noise_params is not None and "GateError" in noise_params:
            phi, random_key = UnitaryGates.GateError(phi, noise_params, random_key)
            theta, random_key = UnitaryGates.GateError(theta, noise_params, random_key)
            omega, random_key = UnitaryGates.GateError(omega, noise_params, random_key)
        PulseGates._execute_composite("Rot", [phi, theta, omega], wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def PauliRot(
        pauli: str,
        theta: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Not implemented as a PulseGate."""
        raise NotImplementedError("PauliRot gate is not implemented as PulseGate")

    @staticmethod
    def RX(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply X-axis rotation using the active pulse envelope.

        Args:
            w: Rotation angle in radians.
            wires: Qubit index or indices.
            pulse_params: Envelope parameters ``[env_0, ..., env_n, t]``.
                If ``None``, uses optimized defaults.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        pulse_params = PulseInformation.RX.split_params(pulse_params)

        PulseGates._record_pulse_event("RX", w, wires, pulse_params)

        _H = op.Hermitian(PulseGates._H_X, wires=wires, record=False)
        H_eff = PulseGates._coeff_Sx * _H

        # Pack: [envelope_params..., w] — evolution time is the last element
        # of pulse_params (pulse_params[-1]).
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        env_params = jnp.array([*pulse_params[:-1], w])
        ys.evolve(H_eff)([env_params], pulse_params[-1])
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RY(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply Y-axis rotation using the active pulse envelope.

        Args:
            w: Rotation angle in radians.
            wires: Qubit index or indices.
            pulse_params: Envelope parameters ``[env_0, ..., env_n, t]``.
                If ``None``, uses optimized defaults.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        pulse_params = PulseInformation.RY.split_params(pulse_params)

        PulseGates._record_pulse_event("RY", w, wires, pulse_params)

        _H = op.Hermitian(PulseGates._H_Y, wires=wires, record=False)
        H_eff = PulseGates._coeff_Sy * _H

        # Pack w into the params so the coefficient function doesn't need
        # a closure — this enables JIT solver cache sharing across all RY calls.
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        env_params = jnp.array([*pulse_params[:-1], w])
        ys.evolve(H_eff)([env_params], pulse_params[-1])
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RZ(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[float] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
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
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        pulse_params = PulseInformation.RZ.split_params(pulse_params)

        PulseGates._record_pulse_event("RZ", w, wires, pulse_params)

        _H = op.Hermitian(PulseGates.Z, wires=wires, record=False)
        H_eff = PulseGates._coeff_Sz * _H

        # Pack w into the params so the coefficient function doesn't need
        # a closure — [pulse_param_scalar, w] enables JIT solver cache sharing.
        # pulse_params may be a 1-element array or scalar; ravel + index to
        # ensure a scalar for concatenation.
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        pp_scalar = jnp.ravel(jnp.asarray(pulse_params))[0]
        ys.evolve(H_eff)([jnp.array([pp_scalar, w])], 1)

        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def _resolve_wires(wire_fn, wires):
        """Resolve a wire selector string to actual wire(s).

        Args:
            wire_fn: ``"all"``, ``"target"``, or ``"control"``.
            wires: Parent gate's wire(s) (int or list).

        Returns:
            Wire(s) for the child gate.
        """
        wires_list = [wires] if isinstance(wires, int) else list(wires)
        if wire_fn == "all":
            return wires if len(wires_list) > 1 else wires_list[0]
        if wire_fn == "target":
            return wires_list[-1] if len(wires_list) > 1 else wires_list[0]
        if wire_fn == "control":
            return wires_list[0]
        raise ValueError(f"Unknown wire_fn: {wire_fn!r}")

    @staticmethod
    def _execute_composite(gate_name, w, wires, pulse_params=None):
        """Execute a composite gate by walking its decomposition.

        Reads the :class:`DecompositionStep` list from
        :class:`PulseInformation` and dispatches each step to the
        appropriate ``PulseGates`` method.

        Args:
            gate_name: Gate name (e.g. ``"H"``, ``"CX"``).
            w: Rotation angle(s) passed to the parent gate.
            wires: Wire(s) of the parent gate.
            pulse_params: Optional pulse parameters (split across children).
        """
        pp_obj = PulseInformation.gate_by_name(gate_name)
        parts = pp_obj.split_params(pulse_params)

        for step, child_params in zip(pp_obj.decomposition, parts):
            child_wires = PulseGates._resolve_wires(step.wire_fn, wires)
            child_w = step.angle_fn(w) if step.angle_fn is not None else w
            child_gate = getattr(PulseGates, step.gate.name)

            # Leaf gates that take a rotation angle
            if step.gate.name in ("RX", "RY", "RZ"):
                child_gate(child_w, wires=child_wires, pulse_params=child_params)
            # Leaf gates without a rotation angle
            elif step.gate.name in ("CZ",):
                child_gate(wires=child_wires, pulse_params=child_params)
            # Composite gates with a rotation angle (CRX, CRY, CRZ, Rot, ...)
            elif step.gate.name in ("Rot",):
                # Rot expects (phi, theta, omega, wires, ...)
                child_gate(*child_w, wires=child_wires, pulse_params=child_params)
            elif step.gate.decomposition is not None and step.gate.name in (
                "CRX",
                "CRY",
                "CRZ",
            ):
                child_gate(child_w, wires=child_wires, pulse_params=child_params)
            # Other composite gates (H, CX, CY, ...)
            else:
                child_gate(wires=child_wires, pulse_params=child_params)

    @staticmethod
    def H(
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply Hadamard gate using pulse decomposition.

        Decomposes as RZ(π) · RY(π/2) followed by a correction phase.

        Args:
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).
        """
        PulseGates._execute_composite("H", 0.0, wires, pulse_params)

        # Correction phase unique to the H gate
        _H = op.Hermitian(PulseGates._H_corr, wires=wires, record=False)
        H_corr = PulseGates._coeff_Sc * _H
        ys.evolve(H_corr)([0], 1)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CX(
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply CNOT gate via decomposition: H(target) · CZ · H(target).

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        PulseGates._execute_composite("CX", 0.0, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CY(
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-Y via decomposition.

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        """
        PulseGates._execute_composite("CY", 0.0, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CZ(
        wires: List[int],
        pulse_params: Optional[float] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-Z using ZZ coupling Hamiltonian.

        Args:
            wires (List[int]): Control and target qubit indices.
            pulse_params (Optional[float]): Time or duration parameter for
                the pulse evolution. If None, uses optimized value.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        """
        if pulse_params is None:
            pulse_params = PulseInformation.CZ.params

        PulseGates._record_pulse_event("CZ", 0.0, wires, pulse_params)

        _H = op.Hermitian(PulseGates._H_CZ, wires=wires, record=False)
        H_eff = PulseGates._coeff_Scz * _H
        ys.evolve(H_eff)([pulse_params], 1)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRX(
        w: float,
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-RX via decomposition.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).
        """
        PulseGates._execute_composite("CRX", w, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRY(
        w: float,
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-RY via decomposition.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        PulseGates._execute_composite("CRY", w, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRZ(
        w: float,
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-RZ via decomposition.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        PulseGates._execute_composite("CRZ", w, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)


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
        allowed_args = [
            "w",
            "wires",
            "phi",
            "theta",
            "omega",
            "input_idx",
            "noise_params",
            "random_key",
        ]
        if gate_mode == "unitary":
            gate_backend = UnitaryGates
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
