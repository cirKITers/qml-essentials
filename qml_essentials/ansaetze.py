from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Callable, Tuple
import jax.numpy as np
import jax
import logging
import warnings

from qml_essentials.gates import Gates, PulseInformation
from qml_essentials.topologies import Topology

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


class Circuit(ABC):
    """Abstract base class for quantum circuit ansätze."""

    def __init__(self) -> None:
        """Initialize the circuit."""
        pass

    @abstractmethod
    def n_params_per_layer(self, n_qubits: int) -> int:
        """
        Get the number of parameters per circuit layer.

        Args:
            n_qubits (int): Number of qubits in the circuit.

        Returns:
            int: Number of parameters required per layer.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("n_params_per_layer method is not implemented")

    def n_pulse_params_per_layer(self, n_qubits: int) -> int:
        """
        Get the number of pulse parameters per circuit layer.

        Subclasses that do not use pulse-level simulation do not need to
        override this method.

        Args:
            n_qubits (int): Number of qubits in the circuit.

        Returns:
            int: Number of pulse parameters required per layer.

        Raises:
            NotImplementedError: If called but not overridden by subclass.
        """
        raise NotImplementedError("n_pulse_params_per_layer method is not implemented")

    @abstractmethod
    def get_control_indices(self, n_qubits: int) -> Optional[List[int]]:
        """
        Get indices for controlled rotation gates in one layer.

        Returns slice indices [start:stop:step] for extracting controlled
        gate parameters from a full parameter array for one layer.

        Args:
            n_qubits (int): Number of qubits in the circuit.

        Returns:
            Optional[List[int]]: List of three integers [start, stop, step]
                for slicing, or None if the circuit contains no controlled
                rotation gates.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("get_control_indices method is not implemented")

    def get_control_angles(self, w: np.ndarray, n_qubits: int) -> Optional[np.ndarray]:
        """
        Extract angles for controlled rotation gates from parameter array.

        Args:
            w (np.ndarray): Parameter array for one layer.
            n_qubits (int): Number of qubits in the circuit.

        Returns:
            Optional[np.ndarray]: Array of controlled gate parameters,
                or empty array if circuit contains no controlled gates.
        """
        indices = self.get_control_indices(n_qubits)
        if indices is None:
            return np.array([])

        return w[indices[0] : indices[1] : indices[2]]

    def _build(self, w: np.ndarray, n_qubits: int, **kwargs) -> Any:
        """
        Build one layer of the circuit using unitary or pulse-level parameters.

        Internal method that handles pulse parameter validation and context
        management before delegating to the build() method.

        Args:
            w (np.ndarray): Parameter array for the current layer.
            n_qubits (int): Number of qubits in the circuit.
            **kwargs: Additional keyword arguments:
                - gate_mode (str): "unitary" (default) or "pulse" for
                  pulse-level simulation.
                - pulse_params (np.ndarray): Pulse parameters if gate_mode="pulse".
                - noise_params (Dict): Noise parameters dictionary.

        Returns:
            Any: Result from the build() method.

        Raises:
            ValueError: If pulse_params length doesn't match expected count.
        """
        gate_mode = kwargs.get("gate_mode", "unitary")

        if gate_mode == "pulse" and "pulse_params" in kwargs:
            pulse_params_per_layer = self.n_pulse_params_per_layer(n_qubits)

            if len(kwargs["pulse_params"]) != pulse_params_per_layer:
                raise ValueError(
                    f"Pulse params length {len(kwargs['pulse_params'])} "
                    f"does not match expected {pulse_params_per_layer} "
                    f"for {n_qubits} qubits"
                )

            with Gates.pulse_manager_context(kwargs["pulse_params"]):
                return self.build(w, n_qubits, **kwargs)
        else:
            return self.build(w, n_qubits, **kwargs)

    @abstractmethod
    def build(self, w: np.ndarray, n_qubits: int, **kwargs) -> Any:
        """
        Build one layer of the quantum circuit.

        Args:
            w (np.ndarray): Parameter array for the current layer.
            n_qubits (int): Number of qubits in the circuit.
            **kwargs: Additional keyword arguments passed from _build.

        Returns:
            Any: Circuit construction result.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("build method is not implemented")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call the _build method with provided arguments."""
        self._build(*args, **kwds)


class DeclarativeCircuit(Circuit):
    """
    A circuit defined entirely by a sequence of Block descriptors.

    Subclasses only need to set the class attribute `structure` — a tuple of

    All of `n_params_per_layer`, `n_pulse_params_per_layer`,
    `get_control_indices`, and `build` are derived automatically.
    """

    @staticmethod
    def structure() -> Tuple[Any, ...]:
        """Override in subclass to return the structure tuple."""
        raise NotImplementedError

    @classmethod
    def n_params_per_layer(cls, n_qubits: int) -> int:
        structure = cls.structure()
        n_params = 0
        for block in structure:
            # we can rely on n_params only returning a valid number
            _n_params = block.n_params(n_qubits)

            n_params += _n_params

        return n_params

    @classmethod
    def n_pulse_params_per_layer(cls, n_qubits: int) -> int:
        structure = cls.structure()
        return sum(block.n_pulse_params(n_qubits) for block in structure)

    @classmethod
    def get_control_indices(cls, n_qubits: int) -> Optional[List]:
        """
        Computes parameter indices for controlled rotation Gates.
        Scans the structure for Block with
        [start, stop, step] into the flat parameter vector, or None.
        """
        structure = cls.structure()
        total_params = sum(block.n_params(n_qubits) for block in structure)

        # Collect which parameter indices correspond to controlled rotations
        controlled_indices = []
        offset = 0
        for block in structure:
            n = block.n_params(n_qubits)
            if block.is_controlled_rotation:
                controlled_indices.extend(range(offset, offset + n))
            offset += n

        # FIXME: this last part should be reworked

        if not controlled_indices:
            return None

        # Check if indices form a contiguous tail (the common case)
        # This preserves backwards compatibility with the [start, None, None] format
        if controlled_indices == list(
            range(total_params - len(controlled_indices), total_params)
        ):
            return [-len(controlled_indices), None, None]

        # Fallback: return raw indices (future-proof)
        return controlled_indices

    @classmethod
    def build(cls, w: np.ndarray, n_qubits: int, **kwargs) -> None:
        structure = cls.structure()
        w_idx = 0
        for block in structure:
            w_idx = block.apply(w, w_idx, n_qubits, **kwargs)


class Block:
    def __init__(
        self,
        gate: str,
        topology: Any = None,
        **kwargs,
    ):
        """
        Initialize a Block object; the atoms of Ansatzes.

        Args:
            gate (str): Name of the Gate class to use.
            topology (Any, optional): Topology of the gate for entangling gates.
                Defaults to None.
            kwargs: Additional keyword arguments passed to the topology function.
        """
        if isinstance(gate, str):
            self.gate = getattr(Gates, gate)
        else:
            self.gate = gate

        if Gates.is_entangling(self.gate):
            assert (
                topology is not None
            ), "Topology must be specified for entangling gates"

        self.topology = topology
        self.kwargs = kwargs

    @property
    def is_entangling(self):
        return Gates.is_entangling(self.gate)

    @property
    def is_rotational(self):
        return Gates.is_rotational(self.gate)

    @property
    def is_controlled_rotation(self):
        return self.is_entangling and self.is_rotational

    def n_params(self, n_qubits: int) -> int:
        assert n_qubits > 0, "Number of qubits must be positive"

        if Gates.is_rotational(self.gate):
            if Gates.is_entangling(self.gate):
                if n_qubits > 1:
                    return len(self.topology(n_qubits=n_qubits, **self.kwargs))
                else:
                    warnings.warn(
                        f"Skipping {self.topology.__name__} with n_qubits={n_qubits} "
                        f"as there are not enough qubits for this topology"
                    )
                    return 0
            else:
                return n_qubits if self.gate.__name__ != "Rot" else 3 * n_qubits

        return 0

    def n_pulse_params(self, n_qubits: int) -> int:
        return PulseInformation.num_params(self.gate) * n_qubits

    def apply(self, w: np.ndarray, w_idx: int, n_qubits: int, **kwargs) -> int:
        assert n_qubits > 0, "Number of qubits must be positive"

        iterator = (
            self.topology(n_qubits=n_qubits, **self.kwargs)
            if Gates.is_entangling(self.gate)
            else range(n_qubits)
        )

        for wires in iterator:
            if Gates.is_rotational(self.gate):
                if self.gate.__name__ == "Rot":
                    self.gate(
                        w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=wires, **kwargs
                    )
                    w_idx += 3
                else:
                    self.gate(w[w_idx], wires=wires, **kwargs)
                    w_idx += 1
            else:
                self.gate(wires=wires, **kwargs)
        return w_idx


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
            Ansaetze.GHZ,
        ]

    class No_Ansatz(DeclarativeCircuit):
        @staticmethod
        def structure():
            return ()

    class GHZ(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.H),
                Block(gate=Gates.CX, topology=Topology.stairs, reverse=True),
            )

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            Gates.H(wires=0, **kwargs)
            for q in range(n_qubits - 1):
                Gates.CX(wires=[q, q + 1], **kwargs)

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            n_params = PulseInformation.num_params("H")  # only 1 H
            n_params += (n_qubits - 1) * PulseInformation.num_params(Gates.CX)
            return n_params

    class Circuit_1(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
            )

    class Circuit_2(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(
                    gate=Gates.CX,
                    topology=Topology.stairs,
                ),
            )

    class Circuit_3(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(gate=Gates.CRZ, topology=Topology.stairs),
            )

    class Circuit_4(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(gate=Gates.CRX, topology=Topology.stairs),
            )

    class Circuit_6(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(gate=Gates.CRX, topology=Topology.all_to_all),
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
            )

    class Circuit_9(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.H),
                Block(gate="CZ", topology=Topology.stairs),
                Block(gate=Gates.RX),
            )

    class Circuit_10(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RY),
                Block(gate="CZ", topology=Topology.stairs, wrap=True, offset=1),
                Block(gate=Gates.RY),
            )

    class Circuit_15(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RY),
                Block(
                    gate=Gates.CX,
                    topology=Topology.stairs,
                    wrap=True,
                    reverse=True,
                    mirror=False,
                ),
                Block(gate=Gates.RY),
                Block(
                    gate=Gates.CX,
                    topology=Topology.stairs,
                    reverse=False,
                    offset=lambda n: n // 2,
                    wrap=True,
                ),
            )

    class Circuit_16(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(
                    gate=Gates.CRZ,
                    topology=Topology.stairs,
                    stride=2,
                ),
                Block(
                    gate=Gates.CRZ,
                    topology=Topology.stairs,
                    stride=2,
                    reverse=False,
                    offset=1,
                    modulo=False,
                ),
            )

    class Circuit_17(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(
                    gate=Gates.CRX,
                    topology=Topology.stairs,
                    stride=2,
                ),
                Block(
                    gate=Gates.CRZ,
                    topology=Topology.stairs,
                    stride=2,
                    reverse=False,
                    offset=1,
                    modulo=False,
                ),
            )

    class Circuit_18(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(
                    gate=Gates.CRZ,
                    topology=Topology.stairs,
                    wrap=True,
                    mirror=False,
                ),
            )

    class Circuit_19(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RX),
                Block(gate=Gates.RZ),
                Block(
                    gate=Gates.CRX,
                    topology=Topology.stairs,
                    wrap=True,
                    mirror=False,
                ),
            )

    class No_Entangling(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (Block(gate=Gates.Rot),)

    class Hardware_Efficient(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.RY),
                Block(gate=Gates.RZ),
                Block(gate=Gates.RY),
                Block(
                    gate=Gates.CX,
                    topology=Topology.stairs,
                    stride=2,
                    mirror=False,
                ),
                Block(
                    gate=Gates.CX,
                    topology=Topology.stairs,
                    stride=2,
                    # reverse=False,
                    offset=-1,
                    wrap=True,
                    mirror=False,
                ),
            )

    class Strongly_Entangling(DeclarativeCircuit):
        @staticmethod
        def structure():
            return (
                Block(gate=Gates.Rot),
                Block(
                    gate=Gates.CX,
                    topology=Topology.stairs,
                    wrap=True,
                    reverse=False,
                    mirror=False,
                ),
                Block(gate=Gates.Rot),
                Block(
                    gate=Gates.CX,
                    topology=Topology.stairs,
                    reverse=False,
                    span=lambda n: n // 2,
                    wrap=True,
                    mirror=False,
                ),
            )


class Encoding:
    def __init__(
        self, strategy: str, gates: Union[str, Callable, List[Union[str, Callable]]]
    ):
        """
        Initializes an Encoding object.

        Implementations closely follow https://doi.org/10.22331/q-2023-12-20-1210

        Parameters
        ----------
        strategy : str
            The encoding strategy to use. Available options:
            ['hamming', 'binary', 'ternary']
        gates : Union[str, Callable, List[Union[str, Callable]]]
            The gates to use for encoding. Can be a string, a callable or a list
            of strings or callables.

        Returns
        -------
        None

        Raises
        -------
        ValueError
            If the encoding strategy is not implemented.
        ValueError
            If there is an error parsing the Gates.
        """
        if strategy not in ["hamming", "binary", "ternary"]:
            raise ValueError(
                f"Encoding strategy {strategy} not implemented. "
                "Available options: ['hamming', 'binary', 'ternary']"
            )
        self._strategy = strategy
        strategy = getattr(self, strategy)

        log.debug(f"Using encoding strategy: '{strategy.__name__}'")

        try:
            self._gates = Gates.parse_gates(gates, Gates)
        except ValueError as e:
            raise ValueError(f"Error parsing encodings: {e}")

        self.callable = [strategy(g) for g in self._gates]

    def __len__(self):
        return len(self.callable)

    def __getitem__(self, idx):
        return self.callable[idx]

    def get_n_freqs(self, omegas):
        """
        Returns the number of frequencies required for the encoding strategy.

        Parameters
        ----------
        omegas : int
            The number of frequencies to encode.

        Returns
        -------
        int
            The number of frequencies required for the encoding strategy.
        """
        if self._strategy == "hamming":
            return int(2 * omegas + 1)
        elif self._strategy == "binary":
            return int(2 ** (omegas + 1) - 1)
        elif self._strategy == "ternary":
            return int(3 ** (omegas))
        else:
            raise NotImplementedError

    def get_spectrum(self, omegas):
        """
        Spectrum for one of the following encoding strategies:

        Hamming: {-n_q -(n_q-1), ..., n_q}
        Binary: {-2^{n_q}+1, ..., 2^{n_q}-1}
        Ternary: {-floor(3^{n_q}/2), ..., floor(3^(n_q)/2)}

        See https://doi.org/10.22331/q-2023-12-20-1210 for more details.

        Parameters
        ----------
        omegas : int
            The number of frequencies to encode.

        Returns
        -------
        np.ndarray
            The spectrum of the encoding strategy.
        """
        if self._strategy == "hamming":
            return np.arange(-omegas, omegas + 1)
        elif self._strategy == "binary":
            return np.arange(-(2**omegas) + 1, 2**omegas)
        elif self._strategy == "ternary":
            limit = int(np.floor(3**omegas / 2))
            return np.arange(-limit, limit + 1)
        else:
            raise NotImplementedError

    def hamming(self, enc):
        """
        Hamming encoding strategy.

        Returns an encoding function that uses the Hamming encoding strategy
        which uses 2 * omegas + 1 frequencies for the encoding.
        See https://doi.org/10.22331/q-2023-12-20-1210 for more details.

        Parameters
        ----------
        enc : Callable
            The encoding function to be wrapped.

        Returns
        -------
        Callable
            The wrapped encoding function.
        """
        return enc

    def binary(self, enc):
        """
        Binary encoding strategy.

        Returns an encoding function that scales the input by a factor of 2^wires.

        Binary encoding uses 2^(omegas + 1) - 1 frequencies for the encoding.
        See https://doi.org/10.22331/q-2023-12-20-1210 for more details.

        Parameters
        ----------
        enc : Callable
            The encoding function to be wrapped.

        Returns
        -------
        Callable
            The wrapped encoding function.
        """

        def _enc(inputs, wires, **kwargs):
            return enc(inputs * (2**wires), wires, **kwargs)

        return _enc

    def ternary(self, enc):
        """
        Ternary encoding strategy.

        Returns an encoding function that scales the input by a factor of 3^wires.

        Ternary encoding uses 3^(omegas + 1) - 1 frequencies for the encoding.
        See https://doi.org/10.22331/q-2023-12-20-1210 for more details.

        Parameters
        ----------
        enc : Callable
            The encoding function to be wrapped.

        Returns
        -------
        Callable
            The wrapped encoding function.
        """

        def _enc(inputs, wires, **kwargs):
            return enc(inputs * (3**wires), wires, **kwargs)

        return _enc

    def golomb(self, enc):
        raise NotImplementedError
