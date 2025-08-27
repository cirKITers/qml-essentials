# flake8: noqa: E731
from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import name
import pennylane.numpy as np
import pennylane as qml
from jax import numpy as jnp
import itertools

from typing import List, Union, Dict

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


class UnitaryGates:
    rng = np.random.default_rng()

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

            dim = 2 ** n
            all_ops = []

            # Generate all n-qubit Pauli tensor products:
            for indices in itertools.product(range(4), repeat=n):
                P = np.eye(1)
                for idx in indices:
                    P = np.kron(P, paulis[idx])
                all_ops.append(P)

            # Identity operator corresponds to all zeros indices (Id^n)
            K0 = np.sqrt(1 - p * (4 ** n - 1) / (4 ** n)) * np.eye(dim)

            kraus_ops = []
            for i, P in enumerate(all_ops):
                if i == 0:
                    # Skip the identity, already handled as K0
                    continue
                kraus_ops.append(np.sqrt(p / (4 ** n)) * P)

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
                qml.BitFlip(noise_params.get("BitFlip", 0.0), wires=wire)
                qml.PhaseFlip(noise_params.get("PhaseFlip", 0.0), wires=wire)
                qml.DepolarizingChannel(
                    noise_params.get("Depolarizing", 0.0), wires=wire
                )

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
        w : float
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
        if (
            noise_params is not None
            and noise_params.get("GateError", None) is not None
        ):
            w += UnitaryGates.rng.normal(0, noise_params["GateError"])
        return w

    @staticmethod
    def Rot(phi, theta, omega, wires, noise_params=None):
        """
        Applies a rotation gate to the given wires and adds `Noise`

        Parameters
        ----------
        phi : float
            The first rotation angle in radians.
        theta : float
            The second rotation angle in radians.
        omega : float
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
        w : float
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
        w : float
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
        w : float
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
        w : float
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
        w : float
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
        w : float
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


class PulseGates:
    # NOTE: Implementation of S, RX, RY, RZ, CZ, CNOT and H pulse level
    # gates closely follow https://doi.org/10.5445/IR/1000184129
    # TODO: Mention deviations from the above?
    # TODO: Which gate decomposition to use for Rot, CRX, CRY, CRZ, CX, CY?
    # Favor CNOT or CZ? Currently, using CZ
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

    opt_params_RX = [15.70989327341467, 29.5230665326707]
    opt_t_RX = 0.7499810441330634

    opt_params_RY = [7.8787724942614235, 22.001319411513432]
    opt_t_RY = 1.098524473819202

    opt_params_H = [7.857992398977854, 21.572701026008765]
    opt_t_H = 0.9000668764548863

    opt_t_CZ = 0.962596375687258

    opt_params_CNOT = [7.944725340235801, 21.639825810701435]
    opt_t_CNOT_H = 0.9072431332410497
    opt_t_CNOT_CZ = 0.9550977662365613

    @staticmethod
    def S(p, t, phi_c):
        A, sigma = p
        t_c = (t[0] + t[1]) / 2 if isinstance(t, (list, tuple)) else t / 2

        f = A * jnp.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        x = jnp.cos(PulseGates.omega_c * t + phi_c)

        return f * x

    @staticmethod
    def Rot(phi, theta, omega, wires): # TODO
        # RZ(phi) · RY(theta) · RZ(omega)
        pass

    @staticmethod
    def RX(w, wires, params=None):
        """
        Applies a rotation around the X axis pulse to the given wires.

        Parameters
        ----------
        w : float
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation to.
        params : Tuple[List[float], float], optional
            Tuple containing pulse parameters `[A, sigma]` and time `t` for the
            Gaussian envelope. Defaults to optimized parameters and time.
        """
        if params is None:
            params = [PulseGates.opt_params_RX]
            t = PulseGates.opt_t_RX
        else:
            params, t = params[:2], params[-1]
            params = [params]

        Sx = lambda p, t: PulseGates.S(p, t, phi_c=jnp.pi) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.X @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sx * _H

        return qml.evolve(H_eff)(params, t)

    @staticmethod
    def RY(w, wires, params=None):
        """
        Applies a rotation around the Y axis pulse to the given wires.

        Parameters
        ----------
        w : float
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation to.
        params : Tuple[List[float], float], optional
            Tuple containing pulse parameters `[A, sigma]` and time `t` for the
            Gaussian envelope. Defaults to optimized parameters and time.
        """
        if params is None:
            params = [PulseGates.opt_params_RY]
            t = PulseGates.opt_t_RY
        else:
            params, t = params[:2], params[-1]
            params = [params]

        Sy = lambda p, t: PulseGates.S(p, t, phi_c=-jnp.pi/2) * w

        _H = PulseGates.H_static.conj().T @ PulseGates.Y @ PulseGates.H_static
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Sy * _H

        return qml.evolve(H_eff)(params, t)

    @staticmethod
    def RZ(w, wires, params=None):
        """
        Applies a rotation around the Z axis to the given wires.

        Parameters
        ----------
        w : float
            The rotation angle in radians.
        wires : Union[int, List[int]]
            The wire(s) to apply the rotation to.
        params : float or None, optional
            Duration of the pulse. Rotation angle = w * 2 * t.
            Defaults to 0.5 if None.
        """
        if params is None:
            t = 0.5
        elif isinstance(params, (float, int)):
            t = params
        else:
            t = params[0]

        _H = qml.Hermitian(PulseGates.Z, wires=wires)
        Sz = lambda p, t: w

        H_eff = Sz * _H

        return qml.evolve(H_eff)([0], t)

    @staticmethod
    def CRX(w, wires): # TODO
        # H_t · CRZ(w) · H_t
        # =
        # H_t · RZ(w/2)_t · CZ · RZ(-w/2)_t · H_t
        pass

    @staticmethod
    def CRY(w, wires): # TODO
        # RX(-pi/2)_t · CRZ(w) · RX(pi/2)_t
        # =
        # RX(-pi/2)_t · RZ(w/2)_t · CZ · RZ(-w/2)_t · RX(pi/2)_t
        pass

    @staticmethod
    def CRZ(w, wires): # TODO
        # RZ(w/2)_t · CZ · RZ(-w/2)_t
        pass

    @staticmethod
    def CX(wires): # TODO
        # H_t · CZ · H_t
        pass

    @staticmethod
    def CY(wires): # TODO
        # RZ(-pi/2)_t · CX · RZ(pi/2)_t
        # =
        # RZ(-pi/2)_t · H_t · CZ · H_t · RZ(pi/2)_t
        pass

    @staticmethod
    def CZ(wires, params=None):
        """
        Applies a controlled Z gate to the given wires.

        Parameters
        ----------
        wires : List[int]
            The wire(s) to apply the controlled Z gate to.
        params : float or None, optional
            Time or time interval for the evolution.
            Defaults to optimized time if None.
        """
        if params is None:
            t = PulseGates.opt_t_CZ
        elif isinstance(params, (float, int)):
            t = params
        else:
            t = params[0]

        I_I = jnp.kron(PulseGates.Id, PulseGates.Id)
        Z_I = jnp.kron(PulseGates.Z, PulseGates.Id)
        I_Z = jnp.kron(PulseGates.Id, PulseGates.Z)
        Z_Z = jnp.kron(PulseGates.Z, PulseGates.Z)

        # NOTE: optimize this parameter too?
        Scz = lambda p, t: jnp.pi

        _H = (jnp.pi / 4) * (I_I - Z_I - I_Z + Z_Z)
        _H = qml.Hermitian(_H, wires=wires)
        H_eff = Scz * _H

        return qml.evolve(H_eff)([0], t)

    @staticmethod
    def CNOT(wires, params=None):
        """
        Applies a CNOT gate composed of Hadamard and controlled-Z pulses.

        Parameters
        ----------
        wires : List[int]
            The control and target wires for the CNOT gate.
        params : Tuple[List[float], Tuple[float, float]], optional
            Tuple containing pulse parameters `[A, sigma]` and a tuple of two times:
            - time for the Hadamard gates
            - time for the controlled-Z gate

            Defaults to optimized parameters and times if None.
        """
        if params is None:
            params = PulseGates.opt_params_CNOT
            t_H = PulseGates.opt_t_CNOT_H
            t_CZ = PulseGates.opt_t_CNOT_CZ
            params += [t_H]
        else:
            print(f"Shape params: {params.shape}\nParams: {params}")
            t_CZ = params[-1]
            params = params[:-1]

        PulseGates.H(wires=wires[1], params=params)
        PulseGates.CZ(wires=wires, params=[t_CZ])
        PulseGates.H(wires=wires[1], params=params)

        return

    @staticmethod
    def H(wires, params=None):
        """
        Applies Hadamard gate to the given wires.

        Parameters
        ----------
        wires : Union[int, List[int]]
            The wire(s) to apply the Hadamard gate to.
        params : Tuple[List[float], float], optional
            Tuple containing pulse parameters `[A, sigma]` and time `t`.
            Defaults to optimized parameters and time.
        """
        if params is None:
            params = PulseGates.opt_params_H
            t = PulseGates.opt_t_H
            params += [t]

        # qml.GlobalPhase(-jnp.pi / 2)
        Sc = lambda p, t: -1.0

        _H = jnp.pi / 2 * jnp.eye(2, dtype=jnp.complex64)
        _H = qml.Hermitian(_H, wires=wires)
        H_corr = Sc * _H

        qml.evolve(H_corr)([0], 1)

        PulseGates.RZ(jnp.pi, wires=wires)
        PulseGates.RY(jnp.pi / 2, wires=wires, params=params)

# Meta class to avoid instantiating the Gates class
class GatesMeta(type):
    def __getattr__(cls, gate_name):
        def handler(*args, **kwargs):
            return Gates._inner_getattr(gate_name, *args, **kwargs)
        return handler

class Gates(metaclass=GatesMeta):
    """
    Gate accessor that dynamically routes calls such as
    'Gates.RX(...)' to either the UnitaryGates or
    PulseGates backend, depending on the 'mode' keyword
    argument (defaults to 'unitary').
    """
    def __getattr__(self, gate_name):
        def handler(**kwargs):
            return self._inner_getattr(gate_name, **kwargs)
        return handler

    @staticmethod
    def _inner_getattr(gate_name, *args, **kwargs):
        mode = kwargs.pop("mode", "unitary")
        if mode == "unitary":
            gate_backend = UnitaryGates
        elif mode == "pulse":
            gate_backend = PulseGates
        else:
            raise ValueError(f"Unknown gate mode: {mode}. Use 'unitary' or 'pulse'.")

        gate = getattr(gate_backend, gate_name, None)
        if gate is None:
            raise AttributeError(f"'{gate_backend.__class__.__name__}' object has no attribute '{gate_name}'")

        return gate(*args, **kwargs)

# TODO: After final HEA solution: extend it to the other ansatzes
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
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            pass

    class GHZ(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return 0

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            Gates.H(0, **kwargs)

            for q in range(n_qubits - 1):
                Gates.CX([q, q + 1], **kwargs)

    # TODO: Include new method "n_pulse_params_per_layer"
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
                Number of qubits in the circuit

            Returns
            -------
            int
                Number of parameters required for one layer of the circuit
            """
            if n_qubits < 2:
                log.warning("Number of Qubits < 2, no entanglement available")
            return n_qubits * 3

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

        # TODO: Add pulse_params after w to build
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

            if n_qubits > 1:
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

            if n_qubits > 1:
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

            if n_qubits > 1:
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

            if n_qubits > 1:
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

            if n_qubits > 1:
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
