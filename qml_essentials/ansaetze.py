from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Dict
import numbers

import pennylane.numpy as np
import pennylane as qml
import jax
from jax import numpy as jnp
import itertools
from contextlib import contextmanager

import logging

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

    # CHECK
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

    # CHECK
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


class PulseInformation:
    """
    Stores pulse parameter counts and optimized pulse parameters for quantum gates.
    """
    PULSE_PARAM_COUNTS: Dict[str, int] = {
        "RX": 3,
        "RY": 3,
        "RZ": 1,
        "CZ": 1,
        "H": 3
    }
    PULSE_PARAM_COUNTS["Rot"] = 2 * PULSE_PARAM_COUNTS["RZ"] + PULSE_PARAM_COUNTS["RY"]
    PULSE_PARAM_COUNTS["CX"] = 2 * PULSE_PARAM_COUNTS["H"] + PULSE_PARAM_COUNTS["CZ"]
    PULSE_PARAM_COUNTS["CY"] = 2 * PULSE_PARAM_COUNTS["RZ"] + PULSE_PARAM_COUNTS["CX"]
    PULSE_PARAM_COUNTS["CRZ"] = 2 * PULSE_PARAM_COUNTS["RZ"] + PULSE_PARAM_COUNTS["CZ"]
    PULSE_PARAM_COUNTS["CRY"] = 2 * PULSE_PARAM_COUNTS["RX"] + PULSE_PARAM_COUNTS["CRZ"]
    PULSE_PARAM_COUNTS["CRX"] = 2 * PULSE_PARAM_COUNTS["H"] + PULSE_PARAM_COUNTS["CRZ"]

    OPTIMIZED_PULSES: Dict[str, Optional[jnp.ndarray]] = {
        "Rot": jnp.array([
            0.5, 7.857992399021039, 21.57270102638842, 0.9000668764608991, 0.5
        ]),
        "RX": jnp.array([15.70989327341467, 29.5230665326707, 0.7499810441330634]),
        "RY": jnp.array([7.8787724942614235, 22.001319411513432, 1.098524473819202]),
        "RZ": jnp.array([0.5]),
        "CRX": jnp.array([
            9.345887537573672, 12.785220434787014, 0.7109351566377278, 0.5,
            15.102609209445896, 0.5, 2.9162064326095, 0.019005851299126367,
            10.000000000000078
        ]),
        "CRY": jnp.array([
            19.113133239181412, 23.385853735839447, 1.2499994641504941, 0.5,
            1.0796514845999126, 0.5, 12.313295392726795, 17.310360723575805,
            0.8499715424933506
        ]),
        "CRZ": jnp.array([0.5, 1.7037270017441872, 0.5]),
        "CX": jnp.array([
            7.951920934692106, 21.655479574101687, 0.8929524493211076,
            0.9548359253748596, 7.94488020182026, 21.61729834699293, 0.9067943033364354
        ]),
        "CY": jnp.array([
            0.5, 13.679990291069169, 6.86497650976022, 1.0547555119435108,
            14.96056469588421, 13.040583781891456, 0.33844677502596704,
            0.8709563476069772, 0.5
        ]),
        "CZ": jnp.array([0.962596375687258]),
        "H": jnp.array([7.857992398977854, 21.572701026008765, 0.9000668764548863]),
    }

    @classmethod
    def num_params(cls, gate: str) -> int:
        """Return the number of pulse parameters for a given gate."""
        if gate not in cls.PULSE_PARAM_COUNTS:
            raise ValueError(f"Unknown gate '{gate}'")
        return cls.PULSE_PARAM_COUNTS[gate]

    @classmethod
    def optimized_params(cls, gate: str) -> Optional[jnp.ndarray]:
        """Return the optimized pulse parameters for a given gate (or None if TODO)."""
        if gate not in cls.OPTIMIZED_PULSES:
            raise ValueError(f"Unknown gate '{gate}'")
        return cls.OPTIMIZED_PULSES[gate]


pinfo = PulseInformation


class PulseGates:
    # NOTE: Implementation of S, RX, RY, RZ, CZ, CNOT/CX and H pulse level
    #   gates closely follow https://doi.org/10.5445/IR/1000184129
    # TODO: Mention deviations from the above?
    # TODO: Raise error if len of pulse params is shorter/larger than expected
    omega_q = 10 * jnp.pi
    omega_c = 10 * jnp.pi

    H_static = jnp.array([
        [jnp.exp(1j * omega_q / 2), 0],
        [0, jnp.exp(-1j * omega_q / 2)]
    ])

    Id = jnp.eye(2, dtype=jnp.complex64)
    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])

    @staticmethod
    def S(p, t, phi_c):
        # TODO: Docstring
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
        n_RZ = pinfo.num_params("RZ")
        n_RY = pinfo.num_params("RY")

        idx1 = n_RZ
        idx2 = idx1 + n_RY
        idx3 = idx2 + n_RZ

        if pulse_params is None:
            opt = pinfo.optimized_params("Rot")
            params_RZ_1 = opt[:idx1]
            params_RY = opt[idx1:idx2]
            params_RZ_2 = opt[idx2:idx3]
        else:
            params_RZ_1 = pulse_params[:idx1]
            params_RY = pulse_params[idx1:idx2]
            params_RZ_2 = pulse_params[idx2:idx3]

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
        n_RX = pinfo.num_params("RX")
        idx = n_RX - 1
        if pulse_params is None:
            opt = pinfo.optimized_params("RX")
            pulse_params, t = opt[:idx], opt[idx]
        else:
            pulse_params, t = pulse_params[:idx], pulse_params[idx]

        Sx = lambda p, t: PulseGates.S(p, t, phi_c=jnp.pi) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.X @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sx * _H

        return qml.evolve(H_eff)([pulse_params], t)

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
        n_RY = pinfo.num_params("RY")
        idx = n_RY - 1
        if pulse_params is None:
            opt = pinfo.optimized_params("RY")
            pulse_params, t = opt[:idx], opt[idx]
        else:
            pulse_params, t = pulse_params[:idx], pulse_params[idx]

        Sy = lambda p, t: PulseGates.S(p, t, phi_c=-jnp.pi/2) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.Y @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sy * _H

        return qml.evolve(H_eff)([pulse_params], t)

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
        n_RZ = pinfo.num_params("RZ")
        if pulse_params is None:
            t = pinfo.optimized_params("RZ")[n_RZ]
        elif isinstance(pulse_params, (float, int)):
            t = pulse_params
        else:
            t = pulse_params[n_RZ]

        _H = qml.Hermitian(PulseGates.Z, wires=wires)
        # TODO: Put comment why p, t has no effect here
        Sz = lambda p, t: w

        H_eff = Sz * _H

        return qml.evolve(H_eff)([0], t)

    @staticmethod
    def CRX(w, wires, pulse_params=None):
        """
        Applies a controlled-RX(w) gate using a decomposition.

        Decomposition:
            CRX(w) = H_t · CRZ(w) · H_t

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
        n_H = pinfo.num_params("H")
        n_CRZ = pinfo.num_params("CRZ")

        idx1 = n_H
        idx2 = idx1 + n_CRZ
        idx3 = idx2 + n_H

        if pulse_params is None:
            opt = pinfo.optimized_params("CRX")
            params_H_1 = opt[:idx1]
            params_CRZ = opt[idx1:idx2]
            params_H_2 = opt[idx2:idx3]
        else:
            params_H_1 = pulse_params[:idx1]
            params_CRZ = pulse_params[idx1:idx2]
            params_H_2 = pulse_params[idx2:idx3]

        target = wires[1]

        PulseGates.H(wires=target, pulse_params=params_H_1)
        PulseGates.CRZ(w, wires, pulse_params=params_CRZ)
        PulseGates.H(wires=target, pulse_params=params_H_2)

        return

    @staticmethod
    def CRY(w, wires, pulse_params=None):
        """
        Applies a controlled-RY(w) gate using a decomposition.

        Decomposition:
            CRY(w) = RX(-π/2)_t · CRZ(w) · RX(π/2)_t

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
        n_RX = pinfo.num_params("RX")
        n_CRZ = pinfo.num_params("CRZ")

        idx1 = n_RX
        idx2 = idx1 + n_CRZ
        idx3 = idx2 + n_RX

        if pulse_params is None:
            opt = pinfo.optimized_params("CRY")
            params_RX_1 = opt[:idx1]
            params_CRZ = opt[idx1:idx2]
            params_RX_2 = opt[idx2:idx3]
        else:
            params_RX_1 = pulse_params[:idx1]
            params_CRZ = pulse_params[idx1:idx2]
            params_RX_2 = pulse_params[idx2:idx3]

        target = wires[1]

        PulseGates.RX(-np.pi/2, wires=target, pulse_params=params_RX_1)
        PulseGates.CRZ(w, wires=wires, pulse_params=params_CRZ)
        PulseGates.RX(np.pi/2, wires=target, pulse_params=params_RX_2)

        return

    @staticmethod
    def CRZ(w, wires, pulse_params=None):
        """
        Applies a controlled-RZ(w) gate using a decomposition.

        Decomposition:
            CRZ(w) = RZ(w/2)_t · CZ · RZ(-w/2)_t

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
        n_RZ = pinfo.num_params("RZ")
        n_CZ = pinfo.num_params("CZ")

        idx1 = n_RZ
        idx2 = idx1 + n_CZ
        idx3 = idx2 + n_RZ

        if pulse_params is None:
            opt = pinfo.optimized_params("CRZ")
            params_RZ_1 = opt[:idx1]
            params_CZ = opt[idx1:idx2]
            params_RZ_2 = opt[idx2:idx3]
        else:
            params_RZ_1 = pulse_params[:idx1]
            params_CZ = pulse_params[idx1:idx2]
            params_RZ_2 = pulse_params[idx2:idx3]

        target = wires[1]

        PulseGates.RZ(w/2, wires=target, pulse_params=params_RZ_1)
        PulseGates.CZ(wires=wires, pulse_params=params_CZ)
        PulseGates.RZ(-w/2, wires=target, pulse_params=params_RZ_2)

        return

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
        n_H = pinfo.num_params("H")
        n_CZ = pinfo.num_params("CZ")

        idx1 = n_H
        idx2 = idx1 + n_CZ
        idx3 = idx2 + n_H

        if pulse_params is None:
            opt = pinfo.optimized_params("CX")
            params_H_1 = opt[:idx1]
            t_CZ = opt[idx1:idx2]
            params_H_2 = opt[idx2:idx3]

        else:
            params_H_1 = pulse_params[:idx1]
            t_CZ = pulse_params[idx1:idx2]
            params_H_2 = pulse_params[idx2:idx3]

        target = wires[1]

        PulseGates.H(wires=target, pulse_params=params_H_1)
        PulseGates.CZ(wires=wires, pulse_params=t_CZ)
        PulseGates.H(wires=target, pulse_params=params_H_2)

        return

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
        n_RZ = pinfo.num_params("RZ")
        n_CX = pinfo.num_params("CX")

        idx1 = n_RZ
        idx2 = idx1 + n_CX
        idx3 = idx2 + n_RZ

        if pulse_params is None:
            opt = pinfo.optimized_params("CY")
            params_RZ_1 = opt[:idx1]
            params_CX = opt[idx1:idx2]
            params_RZ_2 = opt[idx2:idx3]
        else:
            params_RZ_1 = pulse_params[:idx1]
            params_CX = pulse_params[idx1:idx2]
            params_RZ_2 = pulse_params[idx2:idx3]

        target = wires[1]

        PulseGates.RZ(-np.pi/2, wires=target, pulse_params=params_RZ_1)
        PulseGates.CX(wires=wires, pulse_params=params_CX)
        PulseGates.RZ(np.pi/2, wires=target, pulse_params=params_RZ_2)

        return

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
        idx = pinfo.num_params("CZ") - 1
        if pulse_params is None:
            t = pinfo.optimized_params("CZ")[idx]
        elif isinstance(pulse_params, (float, int)):
            t = pulse_params
        else:
            t = pulse_params[idx]

        I_I = jnp.kron(PulseGates.Id, PulseGates.Id)
        Z_I = jnp.kron(PulseGates.Z, PulseGates.Id)
        I_Z = jnp.kron(PulseGates.Id, PulseGates.Z)
        Z_Z = jnp.kron(PulseGates.Z, PulseGates.Z)

        # TODO: explain why p, t not in signal
        Scz = lambda p, t: jnp.pi

        _H = (jnp.pi / 4) * (I_I - Z_I - I_Z + Z_Z)
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Scz * _H

        return qml.evolve(H_eff)([0], t)

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
        if pulse_params is None:
            pulse_params = pinfo.optimized_params("H")
        else:
            pulse_params = pulse_params

        # qml.GlobalPhase(-jnp.pi / 2)
        # TODO: Explain why p, t not in signal
        Sc = lambda p, t: -1.0

        _H = jnp.pi / 2 * jnp.eye(2, dtype=jnp.complex64)
        _H = qml.Hermitian(_H, wires=wires)
        H_corr = Sc * _H

        qml.evolve(H_corr)([0], 1)

        PulseGates.RZ(jnp.pi, wires=wires)
        PulseGates.RY(jnp.pi / 2, wires=wires, pulse_params=pulse_params)

        return


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

    # TODO: Modularize?
    #   E.g. filter_kwargs(), are_valid_pulse_params(), pulse_params_slice_scale()
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

        # Type check on pulse parameters
        pulse_params = kwargs.get("pulse_params")
        if pulse_params is not None:
            # flatten pulse parameters
            if isinstance(pulse_params, (list, tuple)):
                flat_params = pulse_params

            elif isinstance(pulse_params, jax.core.Tracer):
                flat_params = [pulse_params]

            elif isinstance(pulse_params, (np.ndarray, jnp.ndarray)):
                flat_params = pulse_params.flatten().tolist()

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

        # Pulse slicing + scaling
        try:
            n_params = pinfo.num_params(gate_name)
        except ValueError:
            n_params = None
        pulse_mgr = getattr(Gates, "_pulse_mgr", None)
        if gate_mode == "pulse" and isinstance(pulse_mgr, PulseParamManager):
            scalers = pulse_mgr.get(n_params)
            base = pinfo.optimized_params(gate_name)
            kwargs["pulse_params"] = scalers * base  # element-wise scaling

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


class PulseParamManager:
    def __init__(self, pulse_params: np.ndarray):
        self.pulse_params = pulse_params
        self.idx = 0

    def get(self, n: int):
        """Return the next n parameters and advance the cursor."""
        if self.idx + n > len(self.pulse_params):
            raise ValueError("Not enough pulse parameters left for this gate")
        params = self.pulse_params[self.idx:self.idx + n]
        self.idx += n
        return params


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
            n_params = pinfo.num_params("H")
            n_params += (n_qubits - 1) * pinfo.num_params("CX")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            Gates.H(0, **kwargs)

            for q in range(n_qubits - 1):
                Gates.CX([q, q + 1], **kwargs)

    # CHECK
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

        # CHECK
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
            n_params = 2 * pinfo.num_params("RY")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            n_CX = (n_qubits // 2) + ((n_qubits - 1) // 2)
            n_CX += 1 if n_qubits > 2 else 0
            n_params += n_CX * pinfo.num_params("CX")

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
                    Gates.CX(
                        wires=[(2 * q + 1), (2 * q + 2)], **kwargs
                    )
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
            n_params = pinfo.num_params("RX")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += pinfo.num_params("CRX") * n_qubits

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
            n_params = pinfo.num_params("RX")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += pinfo.num_params("CRZ") * n_qubits

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
            n_params = 2 * pinfo.num_params("RY")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += pinfo.num_params("CX") * n_qubits

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
            n_params = pinfo.num_params("H")
            n_params += pinfo.num_params("RX")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * pinfo.num_params("CZ")

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
            n_params = 2 * pinfo.num_params("RX")
            n_params += 2 * pinfo.num_params("RZ")
            n_params *= n_qubits

            n_CRX = n_qubits * (n_qubits - 1)
            n_params += n_CRX * pinfo.num_params("CRX")

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
            n_params = pinfo.num_params("RX")
            n_params += pinfo.num_params("RZ")
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
            n_params = pinfo.num_params("RX")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += (n_qubits - 1) * pinfo.num_params("CX")

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
            n_params = pinfo.num_params("RX")
            n_params = pinfo.num_params("RZ")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * pinfo.num_params("CRZ")

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
            n_params = pinfo.num_params("RX")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * pinfo.num_params("CRX")

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
            n_params = 2 * pinfo.num_params("RY")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * pinfo.num_params("CZ")

            n_params += pinfo.num_params("CZ") if n_qubits > 2 else 0

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
            n_params = pinfo.num_params("RX")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            n_CRZ = n_qubits * (n_qubits - 1) // 2
            n_params += n_CRZ * pinfo.num_params("CRZ")

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
            n_params = pinfo.num_params("RX")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            n_CRZ = n_qubits * (n_qubits - 1) // 2
            n_params += n_CRZ * pinfo.num_params("CRX")

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
            n_params = 2 * pinfo.num_params("Rot")
            n_params *= n_qubits

            if n_qubits > 1:
                n_params += n_qubits * 2 * pinfo.num_params("CX")

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
            n_params = pinfo.num_params("Rot")
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
