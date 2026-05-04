from typing import Optional, List, Union, Dict, Tuple
import itertools
import jax.numpy as jnp
import jax

from qml_essentials import operations as op
import logging

from qml_essentials.utils import safe_random_split

log = logging.getLogger(__name__)


# Cache for computed rulers
_GOLOMB_RULER_CACHE: Dict[int, Tuple[int, ...]] = {}


def _greedy_golomb(d: int) -> Tuple[int, ...]:
    """Construct a valid Golomb ruler of order *d* using a greedy algorithm.

    Starting from mark 0, each subsequent mark is the smallest integer
    whose pairwise differences with all existing marks are distinct.
    This always succeeds and produces a valid ruler, though it may not
    be optimal (i.e. the max mark may not be minimal).

    Args:
        d: Order of the ruler (number of marks).

    Returns:
        Tuple of *d* non-negative integers forming a valid Golomb ruler.
    """
    if d <= 0:
        return ()
    marks = [0]
    diffs: set = set()
    candidate = 1
    while len(marks) < d:
        new_diffs: set = set()
        valid = True
        for existing in marks:
            diff = candidate - existing
            if diff in diffs or diff in new_diffs:
                valid = False
                break
            new_diffs.add(diff)
        if valid:
            marks.append(candidate)
            diffs |= new_diffs
        candidate += 1
    return tuple(marks)


def golomb_ruler(d: int) -> Tuple[int, ...]:
    """Return a valid Golomb ruler of order *d*.

    A Golomb ruler is a set of *d* non-negative integers such that all
    pairwise differences are distinct.  When used as the diagonal of a
    data-encoding Hamiltonian ``H = diag(marks)``, the resulting Fourier
    spectrum ``\\Omega`` has ``|\\Omega| = d(d-1) + 1`` distinct frequencies
    with ``|R(k)| = 1`` for all ``k ≠ 0`` — the minimal possible degeneracy
    for any *d*-dimensional Hamiltonian.

    Uses a greedy construction that always produces a valid ruler.
    Results are cached for efficiency.

    Args:
        d: Order of the ruler (number of marks, equal to the Hilbert
            space dimension ``2^n_qubits``).

    Returns:
        Tuple of *d* non-negative integers forming a Golomb ruler.

    Raises:
        ValueError: If ``d <= 0``.

    References:
        Peters et al., "Generalization despite overfitting in quantum
        machine learning models", arXiv:2209.05523, Appendix C.4.
    """
    if d <= 0:
        raise ValueError(f"Golomb ruler order must be positive, got {d}")
    if d not in _GOLOMB_RULER_CACHE:
        _GOLOMB_RULER_CACHE[d] = _greedy_golomb(d)
    return _GOLOMB_RULER_CACHE[d]


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
    def CPhase(
        w: Union[float, jnp.ndarray, List[float]],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """
        Apply controlled phase shift gate with optional noise.

        This is a generalization of the CZ gate, applying a phase shift of
        exp(i*w) to the |11⟩ state. When w=π, this reduces to CZ.

        Args:
            w (Union[float, jnp.ndarray, List[float]]): Phase shift angle.
            wires (Union[int, List[int]]): Control and target qubit indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for noise.
            input_idx (int): Flag for the tape to track inputs

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        op.ControlledPhaseShift(w, wires=wires, input_idx=input_idx)
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

    @staticmethod
    def GolombEncoding(
        w: Union[float, jnp.ndarray],
        wires: Union[int, List[int]],
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
        input_idx: int = -1,
    ) -> None:
        """Apply Golomb encoding as a diagonal unitary on all given wires.

        Implements ``S(x) = exp(-i H x)`` where
        ``H = diag(g_0, g_1, ..., g_{d-1})`` and the ``g_j`` are the marks
        of a Golomb ruler of order ``d = 2^len(wires)``.  This produces a
        maximally non-degenerate Fourier spectrum with
        ``|\\Omega| = d(d-1) + 1`` distinct frequencies, each with degeneracy
        ``|R(k)| = 1``.

        See Peters et al., arXiv:2209.05523, Sec. 3.1 and Appendix C.4.

        Args:
            w: Scalar input value (the data point *x* to encode).
            wires: Qubit indices this encoding acts on.  All qubits are
                acted upon simultaneously via a single multi-qubit diagonal
                gate.
            noise_params: Optional noise parameters dictionary.
            random_key: JAX random key for stochastic noise.
            input_idx: Flag for the tape to track inputs.

        Returns:
            None: Gate and noise are applied in-place to the circuit.
        """
        wires_list = list(wires) if isinstance(wires, (list, tuple)) else [wires]
        d = 2 ** len(wires_list)
        marks = jnp.array(golomb_ruler(d), dtype=float)

        # Apply gate error to the input angle
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)

        # Build diagonal: exp(-i * mark_j * x)
        diag = jnp.exp(-1j * marks * w)

        op.DiagonalQubitUnitary(diag, wires=wires_list, input_idx=input_idx)
        UnitaryGates.Noise(wires_list, noise_params)
