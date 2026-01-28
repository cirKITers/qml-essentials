import os
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Dict, Callable
import numbers
import csv
import pennylane.numpy as np
import pennylane as qml
import jax
from jax import numpy as jnp
import itertools
from contextlib import contextmanager
import logging

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


class Circuit(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def n_params_per_layer(n_qubits: int) -> int:
        raise NotImplementedError("n_params_per_layer method is not implemented")

    def n_pulse_params_per_layer(n_qubits: int) -> int:
        """
        Return the number of pulse parameters per layer.

        Subclasses that do not use pulse-level simulation do not need to override this.
        If called and not overridden, this will raise NotImplementedError.
        """
        raise NotImplementedError("n_pulse_params_per_layer method is not implemented")

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
        raise NotImplementedError("get_control_indices method is not implemented")

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
            return np.array([])

        return w[indices[0] : indices[1] : indices[2]]

    def _build(self, w: np.ndarray, n_qubits: int, **kwargs):
        """
        Builds one layer of the circuit using either unitary or pulse-level parameters.

        Parameters
        ----------
        w : np.ndarray
            Array of parameters for the current layer.
        n_qubits : int
            Number of qubits in the circuit.
        **kwargs
            Additional keyword arguments. Supports:
            - gate_mode : str, optional
                "unitary" (default) or "pulse" to use pulse-level simulation.
            - pulse_params : jnp.ndarray, optional
                Array of pulse parameters to use if gate_mode="pulse".
            - noise_params : dict, optional
                Dictionary of noise parameters.

        Raises
        ------
        ValueError
            If the number of provided pulse parameters does not match the expected
            number per layer.
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
    def build(self, n_qubits: int, n_layers: int):
        raise NotImplementedError("build method is not implemented")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._build(*args, **kwds)


class UnitaryGates:
    rng = np.random.default_rng()
    batch_gate_error = True

    @staticmethod
    def init_rng(seed: int):
        """
        Initializes the random number generator with the given seed.

        Parameters
        ----------
        seed : int
            The seed for the random number generator.
        """
        UnitaryGates.rng = np.random.default_rng(seed)

    @staticmethod
    def NQubitDepolarizingChannel(p, wires):
        """
        Generates the Kraus operators for an n-qubit depolarizing channel.

        The n-qubit depolarizing channel is defined as:
            E(rho) = sqrt(1 - p * (4^n - 1) / 4^n) * rho
                + sqrt(p / 4^n) * ∑_{P ≠ I^{⊗n}} P rho P†
        where the sum is over all non-identity n-qubit Pauli operators
        (i.e., tensor products of {I, X, Y, Z} excluding the identity operator I^{⊗n}).
        Each Pauli error operator is weighted equally by p / 4^n.

        This operator-sum (Kraus) representation models uniform depolarizing noise
        acting on n qubits simultaneously. It is useful for simulating realistic
        multi-qubit noise affecting entangling gates in noisy quantum circuits.

        Parameters
        ----------
        p : float
            The total probability of an n-qubit depolarizing error occurring.
            Must satisfy 0 ≤ p ≤ 1.

        wires : Sequence[int]
            The list of qubit indices (wires) on which the channel acts.
            Must contain at least 2 qubits.

        Returns
        -------
        qml.QubitChannel
            A PennyLane QubitChannel constructed from the Kraus operators representing
            the n-qubit depolarizing noise channel acting on the specified wires.
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
        Applies noise to the given wires.

        Parameters
        ----------
        wires : Union[int, List[int]]
            The wire(s) to apply the noise to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
            -BitFlip: Applies a bit flip error to the given wires.
            -PhaseFlip: Applies a phase flip error to the given wires.
            -Depolarizing: Applies a depolarizing channel error to the
                given wires.
            -MultiQubitDepolarizing: Applies a two-qubit depolarizing channel
                error to the given wires.

            All parameters are optional and default to 0.0 if not provided.
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
        w: float, noise_params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Applies a gate error to the given rotation angle(s).

        Parameters
        ----------
        w : Union[float, np.ndarray, List[float]]
            The rotation angle in radians.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -GateError: Applies a normal distribution error to the rotation
            angle. The standard deviation of the noise is specified by
            the "GateError" key in the dictionary.

            All parameters are optional and default to 0.0 if not provided.

        Returns
        -------
        float
            The modified rotation angle after applying the gate error.
        """
        if noise_params is not None and noise_params.get("GateError", None) is not None:
            w += UnitaryGates.rng.normal(
                0,
                noise_params["GateError"],
                (
                    w.shape
                    if isinstance(w, np.ndarray) and UnitaryGates.batch_gate_error
                    else None
                ),
            )
        return w

    @staticmethod
    def Rot(phi, theta, omega, wires, noise_params=None):
        """
        Applies a rotation gate to the given wires and adds `Noise`.

        Parameters
        ----------
        phi : Union[float, np.ndarray, List[float]]
            The first rotation angle in radians.
        theta : Union[float, np.ndarray, List[float]]
            The second rotation angle in radians.
        omega : Union[float, np.ndarray, List[float]]
            The third rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        if noise_params is not None and "GateError" in noise_params:
            phi = UnitaryGates.GateError(phi, noise_params)
            theta = UnitaryGates.GateError(theta, noise_params)
            omega = UnitaryGates.GateError(omega, noise_params)
        qml.Rot(phi, theta, omega, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RX(w, wires, noise_params=None):
        """
        Applies a rotation around the X axis to the given wires and adds `Noise`

        Parameters
        ----------
        w : Union[float, np.ndarray, List[float]]
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        w = UnitaryGates.GateError(w, noise_params)
        qml.RX(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RY(w, wires, noise_params=None):
        """
        Applies a rotation around the Y axis to the given wires and adds `Noise`

        Parameters
        ----------
        w : Union[float, np.ndarray, List[float]]
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
            given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        w = UnitaryGates.GateError(w, noise_params)
        qml.RY(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RZ(w, wires, noise_params=None):
        """
        Applies a rotation around the Z axis to the given wires and adds `Noise`

        Parameters
        ----------
        w : Union[float, np.ndarray, List[float]]
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        w = UnitaryGates.GateError(w, noise_params)
        qml.RZ(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRX(w, wires, noise_params=None):
        """
        Applies a controlled rotation around the X axis to the given wires
        and adds `Noise`

        Parameters
        ----------
        w : Union[float, np.ndarray, List[float]]
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the controlled rotation gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        w = UnitaryGates.GateError(w, noise_params)
        qml.CRX(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRY(w, wires, noise_params=None):
        """
        Applies a controlled rotation around the Y axis to the given wires
        and adds `Noise`

        Parameters
        ----------
        w : Union[float, np.ndarray, List[float]]
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the controlled rotation gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        w = UnitaryGates.GateError(w, noise_params)
        qml.CRY(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRZ(w, wires, noise_params=None):
        """
        Applies a controlled rotation around the Z axis to the given wires
        and adds `Noise`

        Parameters
        ----------
        w : Union[float, np.ndarray, List[float]]
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the controlled rotation gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
            given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        w = UnitaryGates.GateError(w, noise_params)
        qml.CRZ(w, wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CX(wires, noise_params=None):
        """
        Applies a controlled NOT gate to the given wires and adds `Noise`

        Parameters
        ----------
        wires : Union[int, List[int]]
            The wire(s) to apply the controlled NOT gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        qml.CNOT(wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CY(wires, noise_params=None):
        """
        Applies a controlled Y gate to the given wires and adds `Noise`

        Parameters
        ----------
        wires : Union[int, List[int]]
            The wire(s) to apply the controlled Y gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        qml.CY(wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CZ(wires, noise_params=None):
        """
        Applies a controlled Z gate to the given wires and adds `Noise`

        Parameters
        ----------
        wires : Union[int, List[int]]
            The wire(s) to apply the controlled Z gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        qml.CZ(wires=wires)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def H(wires, noise_params=None):
        """
        Applies a Hadamard gate to the given wires and adds `Noise`

        Parameters
        ----------
        wires : Union[int, List[int]]
            The wire(s) to apply the Hadamard gate to.
        noise_params : Optional[Dict[str, float]]
            A dictionary of noise parameters. The following noise gates are
            supported:
           -BitFlip: Applies a bit flip error to the given wires.
           -PhaseFlip: Applies a phase flip error to the given wires.
           -Depolarizing: Applies a depolarizing channel error to the
              given wires.

            All parameters are optional and default to 0.0 if not provided.
        """
        qml.Hadamard(wires=wires)
        UnitaryGates.Noise(wires, noise_params)


class PulseParams:
    def __init__(
        self, name: str = "", params: jnp.ndarray = None, pulse_obj: List = None
    ):
        assert (params is None and pulse_obj is not None) or (
            params is not None and pulse_obj is None
        ), "Exactly one of `params` or `pulse_params` must be provided."

        self._pulse_obj = pulse_obj

        if params is not None:
            self._params = params

        self.name = name

    def __len__(self):
        """
        Returns the number of pulse parameters.
        Note that if this gate consists of childs, the number of parameters
        represents the accumulated number of parameters of the childs.

        Returns
        -------
        int
            The number of pulse parameters.
        """
        return len(self.params)

    def __getitem__(self, idx):
        """
        Returns the pulse parameter at index `idx`.
        If this gate consists of childs, the parameters of the child
        at index `idx` are returned.

        Parameters
        ----------
        idx : int
            The index of the pulse parameter to return.

        Returns
        -------
        float
            The pulse parameter at index `idx`.
        """
        if self.is_leaf:
            return self.params[idx]
        else:
            return self.childs[idx].params

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def is_leaf(self):
        return self._pulse_obj is None

    @property
    def size(self):
        return len(self)

    @property
    def leafs(self):
        if self.is_leaf:
            return [self]

        leafs = []
        for obj in self._pulse_obj:
            leafs.extend(obj.leafs)

        return list(set(leafs))

    @property
    def childs(self):
        """
        A list of PulseParams objects, which are the children
        of this PulseParams object.
        If this object has no children, an empty list is returned.

        Returns
        -------
        list
            A list of PulseParams objects, which are the children
            of this PulseParams object.
        """
        if self.is_leaf:
            return []

        return self._pulse_obj

    @property
    def shape(self):
        """
        The shape of the pulse parameters.

        If the PulseParams object has no children (i.e. self.is_leaf),
        the shape is a list containing the number of pulse parameters.

        If the PulseParams object has children, the shape is a list containing
        the shapes of the children.

        Returns
        -------
        list
            The shape of the pulse parameters.
        """
        if self.is_leaf:
            return [len(self.params)]

        shape = []
        for obj in self.childs:
            shape.append(*obj.shape())

            return shape

    @property
    def params(self):
        """
        The pulse parameters.

        If the PulseParams object has no children (i.e. self.is_leaf),
        returns the internal pulse parameters.

        If the PulseParams object has children, returns the concatenated pulse
        parameters of the children.

        Returns
        -------
        jnp.ndarray
            The pulse parameters.
        """
        if self.is_leaf:
            return self._params

        params = self.split_params(params=None, leafs=False)

        return jnp.concatenate(params)

    @params.setter
    def params(self, value):
        """
        Sets the pulse parameters.

        If the PulseParams object has no children (i.e. self.is_leaf),
        sets the internal pulse parameters.

        If the PulseParams object has children, sets the concatenated pulse
        parameters of the children.

        Parameters
        ----------
        value : jnp.ndarray
            The pulse parameters to set.

        Raises
        -------
        AssertionError
            If the PulseParams object has no children and `value` is not a jnp.ndarray.
        """
        if self.is_leaf:
            assert isinstance(value, jnp.ndarray), "params must be a jnp.ndarray"
            self._params = value

        idx = 0
        for obj in self.childs:
            nidx = idx + obj.size
            obj.params = value[idx:nidx]
            idx = nidx

    @property
    def leaf_params(self):
        if self.is_leaf:
            return self._params

        params = self.split_params(None, leafs=True)

        return jnp.concatenate(params)

    @leaf_params.setter
    def leaf_params(self, value):
        if self.is_leaf:
            self._params = value

        idx = 0
        for obj in self.leafs:
            nidx = idx + obj.size
            obj.params = value[idx:nidx]
            idx = nidx

    def split_params(self, params=None, leafs=False):
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
    Stores pulse parameter counts and optimized pulse parameters for quantum gates.
    """

    RX = PulseParams(
        name="RX",
        params=jnp.array([15.863171563255692, 29.66617464185762, 0.7544382603281181]),
    )
    RY = PulseParams(
        name="RY",
        params=jnp.array([7.921864297441735, 22.038129802391797, 1.0940923114464387]),
    )
    RZ = PulseParams(name="RZ", params=jnp.array([0.5]))
    CZ = PulseParams(name="CZ", params=jnp.array([0.3183095268754836]))
    H = PulseParams(
        name="H",
        pulse_obj=[RZ, RY],
    )

    # Rot = PulseParams(name="Rot", pulse_obj=[RZ, RY, RZ])
    CX = PulseParams(name="CX", pulse_obj=[H, CZ, H])
    CY = PulseParams(name="CY", pulse_obj=[RZ, CX, RZ])

    CRX = PulseParams(name="CRX", pulse_obj=[RZ, RY, CX, RY, CX, RZ])
    CRY = PulseParams(name="CRY", pulse_obj=[RY, CX, RY, CX])
    CRZ = PulseParams(name="CRZ", pulse_obj=[RZ, CX, RZ, CX])

    Rot = PulseParams(name="Rot", pulse_obj=[RZ, RY, RZ])

    @staticmethod
    def gate_by_name(gate_name):
        return getattr(PulseInformation, gate_name, None)

    @staticmethod
    def num_params(gate_name):
        return len(getattr(PulseInformation, gate_name, []))

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
    def shuffle_params(seed=1000):
        rng = np.random.default_rng(seed)
        unique_gate_set = [
            PulseInformation.RX,
            PulseInformation.RY,
            PulseInformation.RZ,
            PulseInformation.CZ,
        ]

        log.info(
            f"Shuffling optimized pulses with seed {seed} of gates {unique_gate_set}"
        )
        for gate in unique_gate_set:
            gate.params = jnp.array(rng.random(len(gate)))


class PulseGates:
    # NOTE: Implementation of S, RX, RY, RZ, CZ, CNOT/CX and H pulse level
    #   gates closely follow https://doi.org/10.5445/IR/1000184129
    # TODO: Mention deviations from the above?
    omega_q = 10 * jnp.pi
    omega_c = 10 * jnp.pi

    H_static = jnp.array(
        [[jnp.exp(1j * omega_q / 2), 0], [0, jnp.exp(-1j * omega_q / 2)]]
    )

    Id = jnp.eye(2, dtype=jnp.complex64)
    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])

    @staticmethod
    def _S(p, t, phi_c):
        """
        Generates a shaped pulse envelope modulated by a carrier.
        Note that this is no actual gate, that can be used in a circuit.

        The pulse is a Gaussian envelope multiplied by a cosine carrier, commonly
        used in implementing rotation gates (e.g., RX, RY).

        Parameters
        ----------
        p : sequence of float
            Pulse parameters `[A, sigma]`:
            - A : float, amplitude of the Gaussian
            - sigma : float, width of the Gaussian
        t : float or sequence of float
            Time or time interval over which the pulse is applied. If a sequence,
            `t_c` is taken as the midpoint `(t[0] + t[1]) / 2`.
        phi_c : float
            Phase of the carrier cosine.

        Returns
        -------
        jnp.ndarray
            The shaped pulse at each time step `t`.
        """
        A, sigma = p
        t_c = (t[0] + t[1]) / 2 if isinstance(t, (list, tuple)) else t / 2

        f = A * jnp.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        x = jnp.cos(PulseGates.omega_c * t + phi_c)

        return f * x

    @staticmethod
    def Rot(phi, theta, omega, wires, pulse_params=None):
        """
        Applies a general single-qubit rotation using a decomposition.

        Decomposition:
            Rot(phi, theta, omega) = RZ(phi) · RY(theta) · RZ(omega)

        Parameters
        ----------
        phi : float
            The first rotation angle.
        theta : float
            The second rotation angle.
        omega : float
            The third rotation angle.
        wires : List[int]
            The wire(s) to apply the rotation to.
        pulse_params : np.ndarray, optional
            Pulse parameters for the composing gates. Defaults
            to optimized parameters if None.
        """
        params_RZ_1, params_RY, params_RZ_2 = PulseInformation.Rot.split_params(
            pulse_params
        )

        PulseGates.RZ(phi, wires=wires, pulse_params=params_RZ_1)
        PulseGates.RY(theta, wires=wires, pulse_params=params_RY)
        PulseGates.RZ(omega, wires=wires, pulse_params=params_RZ_2)

    @staticmethod
    def RX(w, wires, pulse_params=None):
        """
        Applies a rotation around the X axis pulse to the given wires.

        Parameters
        ----------
        w : float
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation to.
        pulse_params : np.ndarray, optional
            Array containing pulse parameters `A`, `sigma` and time `t` for the
            Gaussian envelope. Defaults to optimized parameters and time.
        """
        pulse_params = PulseInformation.RX.split_params(pulse_params)

        def Sx(p, t):
            return PulseGates._S(p, t, phi_c=jnp.pi) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.X @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sx * _H

        qml.evolve(H_eff)([pulse_params[0:2]], pulse_params[2])

    @staticmethod
    def RY(w, wires, pulse_params=None):
        """
        Applies a rotation around the Y axis pulse to the given wires.

        Parameters
        ----------
        w : float
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation to.
        pulse_params : np.ndarray, optional
            Array containing pulse parameters `A`, `sigma` and time `t` for the
            Gaussian envelope. Defaults to optimized parameters and time.
        """
        pulse_params = PulseInformation.RY.split_params(pulse_params)

        def Sy(p, t):
            return PulseGates._S(p, t, phi_c=-jnp.pi / 2) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.Y @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sy * _H

        qml.evolve(H_eff)([pulse_params[0:2]], pulse_params[2])

    @staticmethod
    def RZ(w, wires, pulse_params=None):
        """
        Applies a rotation around the Z axis to the given wires.

        Parameters
        ----------
        w : float
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation to.
        pulse_params : float, optional
            Duration of the pulse. Rotation angle = w * 2 * t.
            Defaults to 0.5 if None.
        """
        pulse_params = PulseInformation.RZ.split_params(pulse_params)

        _H = qml.Hermitian(PulseGates.Z, wires=wires)

        def Sz(p, t):
            return p * w

        H_eff = Sz * _H

        qml.evolve(H_eff)([pulse_params], 1)

    @staticmethod
    def H(wires, pulse_params=None):
        """
        Applies Hadamard gate to the given wires.

        Parameters
        ----------
        wires : Union[int, List[int]]
            The wire(s) to apply the Hadamard gate to.
        pulse_params : np.ndarray, optional
            Pulse parameters for the composing gates. Defaults
            to optimized parameters and time.
        """
        pulse_params_RZ, pulse_params_RY = PulseInformation.H.split_params(pulse_params)

        # qml.GlobalPhase(-jnp.pi / 2)  # this could act as substitute to Sc
        PulseGates.RZ(jnp.pi, wires=wires, pulse_params=pulse_params_RZ)
        PulseGates.RY(jnp.pi / 2, wires=wires, pulse_params=pulse_params_RY)

        def Sc(p, t):
            return -1.0

        _H = jnp.pi / 2 * jnp.eye(2, dtype=jnp.complex64)
        _H = qml.Hermitian(_H, wires=wires)
        H_corr = Sc * _H

        qml.evolve(H_corr)([0], 1)

    @staticmethod
    def CX(wires, pulse_params=None):
        """
        Applies a CNOT gate using a decomposition.

        Decomposition:
            CNOT = H_t · CZ · H_t

        Parameters
        ----------
        wires : List[int]
            The control and target wires for the CNOT gate.
        pulse_params : np.ndarray, optional
            Pulse parameters for the composing gates. Defaults
            to optimized parameters if None.
        """
        params_H_1, params_CZ, params_H_2 = PulseInformation.CX.split_params(
            pulse_params
        )

        target = wires[1]

        PulseGates.H(wires=target, pulse_params=params_H_1)
        PulseGates.CZ(wires=wires, pulse_params=params_CZ)
        PulseGates.H(wires=target, pulse_params=params_H_2)

    @staticmethod
    def CY(wires, pulse_params=None):
        """
        Applies a controlled-Y gate using a decomposition.

        Decomposition:
            CY = RZ(-π/2)_t · CX · RZ(π/2)_t

        Parameters
        ----------
        wires : List[int]
            The control and target wires.
        pulse_params : np.ndarray
            Pulse parameters for the composing gates. Defaults
            to optimized parameters if None.
        """
        params_RZ_1, params_CX, params_RZ_2 = PulseInformation.CY.split_params(
            pulse_params
        )

        target = wires[1]

        PulseGates.RZ(-np.pi / 2, wires=target, pulse_params=params_RZ_1)
        PulseGates.CX(wires=wires, pulse_params=params_CX)
        PulseGates.RZ(np.pi / 2, wires=target, pulse_params=params_RZ_2)

    @staticmethod
    def CZ(wires, pulse_params=None):
        """
        Applies a controlled Z gate to the given wires.

        Parameters
        ----------
        wires : List[int]
            The wire(s) to apply the controlled Z gate to.
        pulse_params : float, optional
            Time or time interval for the evolution.
            Defaults to optimized time if None.
        """
        if pulse_params is None:
            pulse_params = PulseInformation.CZ.params
        else:
            pulse_params = pulse_params

        I_I = jnp.kron(PulseGates.Id, PulseGates.Id)
        Z_I = jnp.kron(PulseGates.Z, PulseGates.Id)
        I_Z = jnp.kron(PulseGates.Id, PulseGates.Z)
        Z_Z = jnp.kron(PulseGates.Z, PulseGates.Z)

        def Scz(p, t):
            return p * jnp.pi

        _H = (jnp.pi / 4) * (I_I - Z_I - I_Z + Z_Z)
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Scz * _H

        qml.evolve(H_eff)([pulse_params], 1)

    @staticmethod
    def CRX(w, wires, pulse_params=None):
        """
        Applies a controlled-RX(w) gate using a decomposition.
        Decomposition based on https://doi.org/10.48550/arXiv.2408.01036

        Decomposition:
            CRX(w) = RZ(-pi/2) · RY(w/2) · CX · RY(-w/2) · CX · RZ(pi/2)

        Parameters
        ----------
        w : float
            Rotation angle.
        wires : List[int]
            The control and target wires.
        pulse_params : np.ndarray
            Pulse parameters for the composing gates. Defaults
            to optimized parameters if None.
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
    def CRY(w, wires, pulse_params=None):
        """
        Applies a controlled-RY(w) gate using a decomposition.
        Decomposition based on https://doi.org/10.48550/arXiv.2408.01036

        Decomposition:
            CRY(w) = RY(w/2) · CX · RX(-w/2) · CX

        Parameters
        ----------
        w : float
            Rotation angle.
        wires : List[int]
            The control and target wires.
        pulse_params : np.ndarray
            Pulse parameters for the composing gates. Defaults
            to optimized parameters if None.
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
    def CRZ(w, wires, pulse_params=None):
        """
        Applies a controlled-RZ(w) gate using a decomposition.
        Decomposition based on https://doi.org/10.48550/arXiv.2408.01036

        Decomposition:
            CRZ(w) = RZ(-w/2)_t · CX · RZ(w/2)_t · CX

        Parameters
        ----------
        w : float
            Rotation angle.
        wires : List[int]
            The control and target wires.
        pulse_params : np.ndarray
            Pulse parameters for the composing gates. Defaults
            to optimized parameters if None.
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

        return handler


class Gates(metaclass=GatesMeta):
    """
    Dynamic accessor for quantum gates.

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
            allowed_args += ["noise_params"]
        elif gate_mode == "pulse":
            gate_backend = PulseGates
            allowed_args += ["pulse_params"]
        else:
            raise ValueError(
                f"Unknown gate mode: {gate_mode}. Use 'unitary' or 'pulse'."
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

            elif isinstance(pulse_params, (np.ndarray, jnp.ndarray)):
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


class Ansaetze:
    def get_available(parameterized_only=False):
        # list of parameterized ansaetze
        ansaetze = [
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

        # extend by the non-parameterized ones
        if not parameterized_only:
            ansaetze += [
                Ansaetze.No_Ansatz,
                Ansaetze.GHZ,
            ]

        return ansaetze

    class No_Ansatz(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return 0

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            return 0

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            pass

    class GHZ(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return 0

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for the GHZ circuit.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            int
                Total number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("H")
            n_params += (n_qubits - 1) * PulseInformation.num_params("CX")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            Gates.H(0, **kwargs)

            for q in range(n_qubits - 1):
                Gates.CX([q, q + 1], **kwargs)

    class Hardware_Efficient(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for the
            Hardware Efficient Ansatz.

            The number of parameters is 3 times the number of qubits when there
            is more than one qubit, as each qubit contributes 3 parameters.
            If the number of qubits is less than 2, a warning is logged since
            no entanglement is possible, and a fixed number of 2 parameters is used.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            int
                Number of parameters required for one layer of the circuit.
            """
            if n_qubits < 2:
                log.warning("Number of Qubits < 2, no entanglement available")
            return n_qubits * 3

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for the
            Hardware Efficient Ansatz.

            This counts all parameters needed if the circuit is used at the
            pulse level. It includes contributions from single-qubit rotations
            (`RY` and `RZ`) and multi-qubit gates (`CX`) if more than one qubit
            is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = 2 * PulseInformation.num_params("RY")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            n_CX = (n_qubits // 2) + ((n_qubits - 1) // 2)
            n_CX += 1 if n_qubits > 2 else 0
            n_params += n_CX * PulseInformation.num_params("CX")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Hardware-Efficient ansatz, as proposed in
            https://arxiv.org/pdf/2309.03279

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    Gates.CX(wires=[(2 * q), (2 * q + 1)], **kwargs)
                for q in range((n_qubits - 1) // 2):
                    Gates.CX(wires=[(2 * q + 1), (2 * q + 2)], **kwargs)
                if n_qubits > 2:
                    Gates.CX(wires=[(n_qubits - 1), 0], **kwargs)

    class Circuit_19(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for Circuit_19.

            The number of parameters is 3 times the number of qubits when there
            is more than one qubit, as each qubit contributes 3 parameters.
            If the number of qubits is less than 2, a warning is logged since
            no entanglement is possible, and a fixed number of 2 parameters is used.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters required for one layer of the circuit
            """
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_19.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CRX`) on each qubit if more than one
            qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += PulseInformation.num_params("CRX") * n_qubits

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
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
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit19 ansatz.

            Length of flattened vector must be n_qubits*3
            because for >1 qubits there are three gates

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CRX(
                        w[w_idx],
                        wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits],
                        **kwargs,
                    )
                    w_idx += 1

    class Circuit_18(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for Circuit_18.

            The number of parameters is 3 times the number of qubits when there
            is more than one qubit, as each qubit contributes 3 parameters.
            If the number of qubits is less than 2, a warning is logged since
            no entanglement is possible, and a fixed number of 2 parameters is used.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters required for one layer of the circuit
            """
            if n_qubits > 1:
                return n_qubits * 3
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_18.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CRZ`) on each qubit if more than one
            qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += PulseInformation.num_params("CRZ") * n_qubits

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
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
            if n_qubits > 1:
                return [-n_qubits, None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit18 ansatz.

            Length of flattened vector must be n_qubits*3

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CRZ(
                        w[w_idx],
                        wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits],
                        **kwargs,
                    )
                    w_idx += 1

    class Circuit_15(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for Circuit_15.

            The number of parameters is 2 times the number of qubits.
            A warning is logged if the number of qubits is less than 2.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters required for one layer of the circuit
            """
            if n_qubits > 1:
                return n_qubits * 2
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 2

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_15.

            This includes contributions from single-qubit rotations (`RY`) on all
            qubits, and controlled rotations (`CX`) on each qubit if more than one
            qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = 2 * PulseInformation.num_params("RY")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += PulseInformation.num_params("CX") * n_qubits

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit15 ansatz.

            Length of flattened vector must be n_qubits*2
            because for >1 qubits there are three gates

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*2
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(
                        wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits],
                        **kwargs,
                    )

            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(
                        wires=[(q - 1) % n_qubits, (q - 2) % n_qubits],
                        **kwargs,
                    )

    class Circuit_9(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for Circuit_9.

            The number of parameters is equal to the number of qubits.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters required for one layer of the circuit
            """
            return n_qubits

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_9.

            This includes contributions from single-qubit rotations (`H`, `RX`) on all
            qubits, and controlled rotations (`CZ`) on each qubit except one if more
            than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("H")
            n_params += PulseInformation.num_params("RX")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * PulseInformation.num_params("CZ")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit9 ansatz.

            Length of flattened vector must be n_qubits

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.H(wires=q, **kwargs)

            for q in range(n_qubits - 1):
                Gates.CZ(
                    wires=[n_qubits - q - 2, n_qubits - q - 1],
                    **kwargs,
                )

            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1

    class Circuit_6(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for Circuit_6.

            The total number of parameters is n_qubits*3+n_qubits**2, which is
            the number of rotations n_qubits*3 plus the number of entangling gates
            n_qubits**2.

            If n_qubits is 1, the number of parameters is 4, and a warning is logged
            since no entanglement is possible.

            Parameters
            ----------
            n_qubits : int
                Number of qubits

            Returns
            -------
            int
                Number of parameters per layer
            """
            if n_qubits > 1:
                return n_qubits * 3 + n_qubits**2
            else:
                log.warning("Number of Qubits < 2, no entanglement available")
                return 4

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_6.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CRX`) on each qubit twice except repeats
            if more than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = 2 * PulseInformation.num_params("RX")
            n_params += 2 * PulseInformation.num_params("RZ")
            n_params *= n_qubits

            n_CRX = n_qubits * (n_qubits - 1)
            n_params += n_CRX * PulseInformation.num_params("CRX")

            return 0

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
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
            # TODO: implement
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit6 ansatz.

            Length of flattened vector must be
                n_qubits*4+n_qubits*(n_qubits-1) =
                n_qubits*3+n_qubits**2

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size
                    n_layers*(n_qubits*3+n_qubits**2)
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for ql in range(n_qubits):
                    for q in range(n_qubits):
                        if q == ql:
                            continue
                        Gates.CRX(
                            w[w_idx],
                            wires=[n_qubits - ql - 1, (n_qubits - q - 1) % n_qubits],
                            **kwargs,
                        )
                        w_idx += 1

            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

    class Circuit_1(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for Circuit_1.

            The total number of parameters is determined by the number of qubits, with
            each qubit contributing 2 parameters.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 2

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_9.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits only.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit1 ansatz.

            Length of flattened vector must be n_qubits*2

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*2
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

    class Circuit_2(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for Circuit_2.

            The total number of parameters is determined by the number of qubits, with
            each qubit contributing 2 parameters.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 2

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_2.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CX`) on each qubit except one if more
            than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += (n_qubits - 1) * PulseInformation.num_params("CX")

            return 0

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit2 ansatz.

            Length of flattened vector must be n_qubits*2

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*2
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            for q in range(n_qubits - 1):
                Gates.CX(
                    wires=[n_qubits - q - 1, n_qubits - q - 2],
                    **kwargs,
                )

    class Circuit_3(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Calculates the number of parameters per layer for Circuit3.

            The number of parameters per layer is given by the number of qubits, with
            each qubit contributing 3 parameters. The last qubit only contributes 2
            parameters because it is the target qubit for the controlled gates.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 3 - 1

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_3.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CRZ`) on each qubit except one if more
            than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params = PulseInformation.num_params("RZ")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * PulseInformation.num_params("CRZ")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            if n_qubits > 1:
                return [-(n_qubits - 1), None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit3 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3-1
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            for q in range(n_qubits - 1):
                Gates.CRZ(
                    w[w_idx],
                    wires=[n_qubits - q - 1, n_qubits - q - 2],
                    **kwargs,
                )
                w_idx += 1

    class Circuit_4(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for the Circuit_4 ansatz.

            The number of parameters is calculated as n_qubits*3-1.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 3 - 1

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_4.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CRX`) on each qubit except one if more
            than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * PulseInformation.num_params("CRX")

            return 0

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            if n_qubits > 1:
                return [-(n_qubits - 1), None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit4 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3-1
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            for q in range(n_qubits - 1):
                Gates.CRX(
                    w[w_idx],
                    wires=[n_qubits - q - 1, n_qubits - q - 2],
                    **kwargs,
                )
                w_idx += 1

    class Circuit_10(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for the Circuit_10 ansatz.

            The number of parameters is calculated as n_qubits*2.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 2  # constant gates not considered yet. has to be fixed

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_10.

            This includes contributions from single-qubit rotations (`RY`) on all
            qubits, controlled rotations (`CZ`) on each qubit except one if more
            than one qubit is present and a final controlled rotation (`CZ`) if
            more than two qubits are present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = 2 * PulseInformation.num_params("RY")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * PulseInformation.num_params("CZ")

            n_params += PulseInformation.num_params("CZ") if n_qubits > 2 else 0

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            Optional[np.ndarray]
                List of all controlled indices, or None if the circuit does not
                contain controlled rotation gates.
            """
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit10 ansatz.

            Length of flattened vector must be n_qubits*2

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*2
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            # constant gates, independent of layers. has to be fixed
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            for q in range(n_qubits - 1):
                Gates.CZ(
                    wires=[
                        (n_qubits - q - 2) % n_qubits,
                        (n_qubits - q - 1) % n_qubits,
                    ],
                    **kwargs,
                )
            if n_qubits > 2:
                Gates.CZ(wires=[n_qubits - 1, 0], **kwargs)

            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1

    class Circuit_16(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for the Circuit_16 ansatz.

            The number of parameters is calculated as n_qubits*3-1.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 3 - 1

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_16.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CRZ`) if more than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            n_CRZ = n_qubits * (n_qubits - 1) // 2
            n_params += n_CRZ * PulseInformation.num_params("CRZ")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            if n_qubits > 1:
                return [-(n_qubits - 1), None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit16 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3-1
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    Gates.CRZ(
                        w[w_idx],
                        wires=[(2 * q + 1), (2 * q)],
                        **kwargs,
                    )
                    w_idx += 1

                for q in range((n_qubits - 1) // 2):
                    Gates.CRZ(
                        w[w_idx],
                        wires=[(2 * q + 2), (2 * q + 1)],
                        **kwargs,
                    )
                    w_idx += 1

    class Circuit_17(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for the Circuit_17 ansatz.

            The number of parameters is calculated as n_qubits*3-1.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 3 - 1

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Circuit_17.

            This includes contributions from single-qubit rotations (`RX`, `RZ`) on all
            qubits, and controlled rotations (`CRX`) if more than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("RX")
            n_params += PulseInformation.num_params("RZ")
            n_params *= n_qubits

            n_CRZ = n_qubits * (n_qubits - 1) // 2
            n_params += n_CRZ * PulseInformation.num_params("CRX")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            if n_qubits > 1:
                return [-(n_qubits - 1), None, None]
            else:
                return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a Circuit17 ansatz.

            Length of flattened vector must be n_qubits*3-1

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3-1
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.RX(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits // 2):
                    Gates.CRX(
                        w[w_idx],
                        wires=[(2 * q + 1), (2 * q)],
                        **kwargs,
                    )
                    w_idx += 1

                for q in range((n_qubits - 1) // 2):
                    Gates.CRX(
                        w[w_idx],
                        wires=[(2 * q + 2), (2 * q + 1)],
                        **kwargs,
                    )
                    w_idx += 1

    class Strongly_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for the
            Strongly Entangling ansatz.

            The number of parameters is calculated as n_qubits*6.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            if n_qubits < 2:
                log.warning("Number of Qubits < 2, no entanglement available")
            return n_qubits * 6

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for Strongly_Entangling
            circuit.

            This includes contributions from single-qubit rotations (`Rot`) on all
            qubits, and controlled rotations (`CX`) if more than one qubit is present.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = 2 * PulseInformation.num_params("Rot")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += n_qubits * 2 * PulseInformation.num_params("CX")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs) -> None:
            """
            Creates a Strongly Entangling ansatz.

            Length of flattened vector must be n_qubits*6

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*6
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.Rot(
                    w[w_idx],
                    w[w_idx + 1],
                    w[w_idx + 2],
                    wires=q,
                    **kwargs,
                )
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(wires=[q, (q + 1) % n_qubits], **kwargs)

            for q in range(n_qubits):
                Gates.Rot(
                    w[w_idx],
                    w[w_idx + 1],
                    w[w_idx + 2],
                    wires=q,
                    **kwargs,
                )
                w_idx += 3

            if n_qubits > 1:
                for q in range(n_qubits):
                    Gates.CX(
                        wires=[q, (q + n_qubits // 2) % n_qubits],
                        **kwargs,
                    )

    class No_Entangling(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of parameters per layer for the NoEntangling ansatz.

            The number of parameters is calculated as n_qubits*3.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters per layer
            """
            return n_qubits * 3

        @staticmethod
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            """
            Returns the number of pulse parameters per layer for No_Entangling circuit.

            This includes contributions from single-qubit rotations (`Rot`) on all
            qubits only.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.

            Returns
            -------
            int
                Number of pulse parameters required for one layer of the circuit.
            """
            n_params = PulseInformation.num_params("Rot")
            n_params *= n_qubits

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            """
            No controlled rotation gates available. Always None.

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
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            """
            Creates a circuit without entangling, but with U3 gates on all qubits

            Length of flattened vector must be n_qubits*3

            Parameters
            ----------
            w : np.ndarray
                Weight vector of size n_qubits*3
            n_qubits : int
                Number of qubits
            noise_params : Optional[Dict[str, float]], optional
                Dictionary of noise parameters to apply to the gates
            """
            w_idx = 0
            for q in range(n_qubits):
                Gates.Rot(
                    w[w_idx],
                    w[w_idx + 1],
                    w[w_idx + 2],
                    wires=q,
                    **kwargs,
                )
                w_idx += 3


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
            If there is an error parsing the gates.
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
