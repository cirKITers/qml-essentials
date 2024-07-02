from typing import Dict, Optional, Tuple, Callable, Union
import pennylane as qml
import pennylane.numpy as np
import hashlib
import os

from qml_essentials.ansaetze import Ansaetze

import logging

log = logging.getLogger(__name__)


class Model:
    """
    A quantum circuit model.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        circuit_type: str,
        data_reupload: bool = True,
        initialization: str = "random",
        output_qubit: int = 0,
        shots: Optional[int] = None,
    ) -> None:
        """
        Initialize the quantum circuit model.
        Parameters will have the shape [impl_n_layers, parameters_per_layer]
        where impl_n_layers is the number of layers provided and added by one
        depending if data_reupload is True and parameters_per_layer is given by
        the chosen ansatz.

        Args:
            n_qubits (int): The number of qubits in the circuit.
            n_layers (int): The number of layers in the circuit.
            circuit_type (str): The type of quantum circuit to use.
                If None, defaults to "no_ansatz".
            data_reupload (bool, optional): Whether to reupload data to the
                quantum device on each measurement. Defaults to True.
            output_qubit (int, optional): The index of the output qubit.
        Returns:
            None
        """
        # Initialize default parameters needed for circuit evaluation
        self.noise_params: Optional[Dict[str, float]] = None
        self.state_vector: bool = False
        self.exp_val: bool = True

        # Copy the parameters
        self.n_qubits: int = n_qubits
        self.n_layers: int = n_layers
        self.data_reupload: bool = data_reupload
        self.output_qubit: int = output_qubit

        # Initialize ansatz
        self.pqc: Callable[[Optional[np.ndarray], int], int] = getattr(
            Ansaetze, circuit_type or "no_ansatz"
        )()

        log.info(f"Using {circuit_type} circuit.")

        # Only reasonable number of shots
        self.shots = None if shots <= 0 else shots

        if data_reupload:
            impl_n_layers: int = n_layers + 1  # we need L+1 according to Schuld et al.
            self.degree = n_layers * n_qubits
        else:
            impl_n_layers: int = n_layers
            self.degree = 1

        log.info(f"Number of implicit layers set to {impl_n_layers}.")

        params_shape: Tuple[int, int] = (
            impl_n_layers,
            self.pqc.n_params_per_layer(self.n_qubits),
        )

        def set_control_params(params, value):
            indices = self.pqc.get_control_indices(self.n_qubits)
            if indices is None:
                log.warning(
                    f"Specified {initialization} but circuit does not contain controlled rotation gates. Parameters are intialized randomly."
                )
            else:
                params[:, indices[0] : indices[1] : indices[2]] = (
                    np.ones_like(params[:, indices[0] : indices[1] : indices[2]])
                    * value
                )
            return params

        if initialization == "random":
            self.params: np.ndarray = np.random.uniform(
                0, 2 * np.pi, params_shape, requires_grad=True
            )
        elif initialization == "zeros":
            self.params: np.ndarray = np.zeros(params_shape, requires_grad=True)
        elif initialization == "zero-controlled":
            self.params: np.ndarray = np.random.uniform(
                0, 2 * np.pi, params_shape, requires_grad=True
            )
            self.params = set_control_params(self.params, 0)
        elif initialization == "pi-controlled":
            self.params: np.ndarray = np.random.uniform(
                0, 2 * np.pi, params_shape, requires_grad=True
            )
            self.params = set_control_params(self.params, np.pi)
        else:
            raise Exception("Invalid initialization method")

        log.info(
            f"Initialized parameters with shape {self.params.shape} using strategy {initialization}."
        )

        device = "default.mixed" if self.shots is None else "default.qubit"
        self.dev: qml.Device = qml.device(device, shots=self.shots, wires=self.n_qubits)

        log.info(f"Using device {device}.")

        self.circuit: qml.QNode = qml.QNode(self._circuit, self.dev)

        log.debug(self._draw())

    def _iec(
        self,
        inputs: np.ndarray,
        data_reupload: bool = True,
    ) -> None:
        """
        Creates an AngleEncoding using RX gates

        Args:
            inputs (np.ndarray): length of vector must be 1, shape (1,)
            data_reupload (bool, optional): Whether to reupload the data for the IEC
                or not, default is True.

        Returns:
            None
        """
        if inputs is None:
            # initialize to zero
            inputs = np.array([[0]])
        elif len(inputs.shape) == 1:
            # add a batch dimension
            inputs = np.array([inputs])

        if data_reupload:
            if inputs.shape[1] == 1:
                for q in range(self.n_qubits):
                    qml.RX(inputs[:, 0], wires=q)
            elif inputs.shape[1] == 2:
                for q in range(self.n_qubits):
                    qml.RX(inputs[:, 0], wires=q)
                    qml.RY(inputs[:, 1], wires=q)
            elif inputs.shape[1] == 3:
                for q in range(self.n_qubits):
                    qml.Rot(inputs[:, 0], inputs[:, 1], inputs[:, 2], wires=q)
            else:
                raise ValueError(
                    "The number of parameters for this IEC cannot be greater than 3"
                )
        else:
            qml.RX(inputs, wires=0)

    def _circuit(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """
        Creates a circuit with noise.

        Args:
            params (np.ndarray): weight vector of shape [n_layers, n_qubits*n_params_per_layer]
            inputs (np.ndarray): input vector of size 1
        Returns:
            Union[float, np.ndarray]: Expectation value of PauliZ(0) of the circuit if
                state_vector is False and exp_val is True, otherwise the density matrix
                of all qubits.

        Raises:
            ValueError: If a) state_vector and exp_val are set, b) if either state_vector or
            exp_val is true and shots is not none, c) if state_vector and exp_val are both false
            but shots_is none
        """

        for l in range(0, self.n_layers):
            self.pqc(params[l], self.n_qubits)

            if self.data_reupload or l == 0:
                self._iec(inputs, data_reupload=self.data_reupload)

            if self.noise_params is not None:
                for q in range(self.n_qubits):
                    qml.BitFlip(self.noise_params.get("BitFlip", 0.0), wires=q)
                    qml.PhaseFlip(self.noise_params.get("PhaseFlip", 0.0), wires=q)
                    qml.AmplitudeDamping(
                        self.noise_params.get("AmplitudeDamping", 0.0), wires=q
                    )
                    qml.PhaseDamping(
                        self.noise_params.get("PhaseDamping", 0.0), wires=q
                    )
                    qml.DepolarizingChannel(
                        self.noise_params.get("DepolarizingChannel", 0.0), wires=q
                    )

        if self.data_reupload:
            self.pqc(params[-1], self.n_qubits)

        if self.state_vector and not self.exp_val and self.shots is None:
            return qml.density_matrix(wires=list(range(self.n_qubits)))
        elif self.exp_val and self.shots is None:
            return qml.expval(qml.PauliZ(self.output_qubit))
        elif self.shots is not None:
            if self.output_qubit == -1:
                return 2 * qml.probs(wires=list(range(self.n_qubits))) - 1
            else:
                return 2 * qml.probs(wires=self.output_qubit) - 1
        else:
            raise ValueError(
                f"Invalid combination of parameters state_vector:{self.state_vector}, 
                exp_val:{self.exp_val} and shots:{self.shots}"
            )

    def _draw(self) -> None:
        return qml.draw(self.circuit)(params=self.params, inputs=None)

    def __repr__(self) -> str:
        return self._draw()

    def __str__(self) -> str:
        return self._draw()

    def __call__(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        state_vector: bool = False,
        exp_val: bool = True,
    ) -> np.ndarray:
        """Perform a forward pass of the quantum circuit.

        Args:
            params (np.ndarray): Weight vector of size n_layers*(n_qubits*3-1).
            inputs (np.ndarray): Input vector of size 1.
            noise_params (Optional[Dict[str, float]], optional): Dictionary with noise parameters. Defaults to None.
            cache (Optional[bool], optional): Cache the circuit. Defaults to False.
            state_vector (bool, optional): Measure the state vector instead of the wave function. Defaults to False.
            exp_val (bool, optional): Compute the expectation value of PauliZ(0). Defaults to True.

        Returns:
            np.ndarray: Expectation value of PauliZ(0) of the circuit.
        """
        # Call forward method which handles the actual caching etc.
        return self._forward(params, inputs, noise_params, cache, state_vector, exp_val)

    def _forward(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        state_vector: bool = False,
        exp_val: bool = True,
    ) -> np.ndarray:
        """
        Perform a forward pass of the quantum circuit.

        Args:
            params (np.ndarray): Weight vector of size n_layers*(n_qubits*3-1).
            inputs (np.ndarray): Input vector of size 1.
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None.
            cache (Optional[bool], optional): Whether to cache the results. Defaults to False.
            state_vector (bool, optional): Whether to return the state vector instead of the
                expectation value. Defaults to False.
            exp_val (bool, optional): Whether to compute the expectation value. Defaults to True.

        Returns:
            np.ndarray: The output of the quantum circuit.

        Raises:
            NotImplementedError: If the number of shots is not None or if the expectation
                value is True.
        """
        # set the parameters as object attributes
        self.noise_params = noise_params
        self.state_vector = state_vector
        self.exp_val = exp_val

        # the qasm representation contains the bound parameters, thus it is ok to hash that
        hs = hashlib.md5(
            repr(
                {
                    "n_qubits": self.n_qubits,
                    "n_layers": self.n_layers,
                    "pqc": self.pqc.__class__.__name__,
                    "dru": self.data_reupload,
                    "params": params,
                    "noise_params": noise_params,
                }
            ).encode("utf-8")
        ).hexdigest()

        result: Optional[np.ndarray] = None
        if cache:
            if self.shots is not None or self.exp_val:
                raise NotImplementedError(
                    "Caching with shots or exp_val not yet implemented."
                )
            name: str = f"pqc_{hs}.npy"

            cache_folder: str = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path: str = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                result = np.load(file_path)

        if result is None:
            # execute the PQC circuit with the current set of parameters
            result = self.circuit(
                params=params,
                inputs=inputs,
                noise_params=noise_params,
                state_vector=state_vector,
                exp_val=exp_val,
            )

        if cache:
            np.save(file_path, result)

        return result
