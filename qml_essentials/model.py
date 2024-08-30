from typing import Dict, Optional, Tuple, Callable, Union, List
import pennylane as qml
import pennylane.numpy as np
import hashlib
import os
import warnings

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
        output_qubit: Union[List[int], int] = 0,
        shots: Optional[int] = None,
        random_seed: int = 1000,
    ) -> None:
        """
        Initialize the quantum circuit model.
        Parameters will have the shape [impl_n_layers, parameters_per_layer]
        where impl_n_layers is the number of layers provided and added by one
        depending if data_reupload is True and parameters_per_layer is given by
        the chosen ansatz.

        The model is initialized with the following parameters as defaults:
        - noise_params: None
        - execution_type: "expval"
        - shots: None

        Args:
            n_qubits (int): The number of qubits in the circuit.
            n_layers (int): The number of layers in the circuit.
            circuit_type (str): The type of quantum circuit to use.
                If None, defaults to "no_ansatz".
            data_reupload (bool, optional): Whether to reupload data to the
                quantum device on each measurement. Defaults to True.
            initialization (str, optional): The strategy to initialize the parameters.
                Can be "random", "zeros", "zero-controlled", "pi", or "pi-controlled".
                Defaults to "random".
            output_qubit (List[int], int, optional): The index of the output
                qubit (or qubits). When set to -1 all qubits are measured, or a
                global measurement is conducted, depending on the execution
                type.
            shots (Optional[int], optional): The number of shots to use for
                the quantum device. Defaults to None.
            random_seed (int, optional): seed for the random number generator
                in initialization is "random", Defaults to 1000.

        Returns:
            None
        """
        # Initialize default parameters needed for circuit evaluation
        self.noise_params: Optional[Dict[str, float]] = None
        self.execution_type: Optional[str] = "expval"
        self.shots = shots
        self.output_qubit: Union[List[int], int] = output_qubit

        # Copy the parameters
        self.n_qubits: int = n_qubits
        self.n_layers: int = n_layers
        self.data_reupload: bool = data_reupload

        # Initialize ansatz
        self.pqc: Callable[[Optional[np.ndarray], int], int] = getattr(
            Ansaetze, circuit_type or "no_ansatz"
        )()

        log.info(f"Using {circuit_type} circuit.")

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

        def set_control_params(params: np.ndarray, value: float) -> np.ndarray:
            indices = self.pqc.get_control_indices(self.n_qubits)
            if indices is None:
                warnings.warn(
                    f"Specified {initialization} but circuit\
                    does not contain controlled rotation gates.\
                    Parameters are intialized randomly.",
                    UserWarning,
                )
            else:
                params[:, indices[0] : indices[1] : indices[2]] = (
                    np.ones_like(params[:, indices[0] : indices[1] : indices[2]])
                    * value
                )
            return params

        rng = np.random.default_rng(random_seed)
        if initialization == "random":
            self.params: np.ndarray = rng.uniform(
                0, 2 * np.pi, params_shape, requires_grad=True
            )
        elif initialization == "zeros":
            self.params: np.ndarray = np.zeros(params_shape, requires_grad=True)
        elif initialization == "pi":
            self.params: np.ndarray = np.ones(params_shape, requires_grad=True) * np.pi
        elif initialization == "zero-controlled":
            self.params: np.ndarray = rng.uniform(
                0, 2 * np.pi, params_shape, requires_grad=True
            )
            self.params = set_control_params(self.params, 0)
        elif initialization == "pi-controlled":
            self.params: np.ndarray = rng.uniform(
                0, 2 * np.pi, params_shape, requires_grad=True
            )
            self.params = set_control_params(self.params, np.pi)
        else:
            raise Exception("Invalid initialization method")

        log.info(
            f"Initialized parameters with shape {self.params.shape}\
            using strategy {initialization}."
        )

        # Initialize two circuits, one with the default device and
        # one with the mixed device
        # which allows us to later route depending on the state_vector flag
        self.circuit: qml.QNode = qml.QNode(
            self._circuit,
            qml.device("default.qubit", shots=self.shots, wires=self.n_qubits),
        )
        self.circuit_mixed: qml.QNode = qml.QNode(
            self._circuit,
            qml.device("default.mixed", shots=self.shots, wires=self.n_qubits),
        )

        log.debug(self._draw())

    @property
    def noise_params(self) -> Optional[Dict[str, float]]:
        """
        Gets the noise parameters of the model.

        Returns:
            Optional[Dict[str, float]]: A dictionary of noise parameters or None if not set.
        """
        return self._noise_params

    @noise_params.setter
    def noise_params(self, value: Optional[Dict[str, float]]) -> None:
        """
        Sets the noise parameters of the model.

        Args:
            value (Optional[Dict[str, float]]): A dictionary of noise parameters.
                If all values are 0.0, the noise parameters are set to None.

        Returns:
            None
        """
        if value is not None and all(np == 0.0 for np in value.values()):
            value = None
        self._noise_params = value

    @property
    def execution_type(self) -> str:
        """
        Gets the execution type of the model.

        Returns:
            str: The execution type, one of 'density', 'expval', or 'probs'.
        """
        return self._execution_type

    @execution_type.setter
    def execution_type(self, value: str) -> None:
        if value not in ["density", "expval", "probs"]:
            raise ValueError(f"Invalid execution type: {value}.")

        if value == "density" and self.output_qubit != -1:
            warnings.warn(
                f"{value} measurement does ignore output_qubit, which is "
                f"{self.output_qubit}.",
                UserWarning,
            )

        if value == "probs" and self.shots is None:
            warnings.warn(
                "Setting execution_type to probs without specifying shots.", UserWarning
            )

        if value == "density" and self.shots is not None:
            warnings.warn(
                "Setting execution_type to density with specified shots.", UserWarning
            )

        self._execution_type = value

    @property
    def shots(self) -> Optional[int]:
        """
        Gets the number of shots to use for the quantum device.

        Returns:
            Optional[int]: The number of shots.
        """
        return self._shots

    @shots.setter
    def shots(self, value: Optional[int]) -> None:
        """
        Sets the number of shots to use for the quantum device.

        Args:
            value (Optional[int]): The number of shots. If an integer less than or equal to 0 is provided, it is set to None.

        Returns:
            None
        """
        if type(value) is int and value <= 0:
            value = None
        self._shots = value

    def _iec(
        self,
        inputs: np.ndarray,
        data_reupload: bool = True,
    ) -> None:
        """
        Creates an AngleEncoding using RX gates

        Args:
            inputs (np.ndarray): length of vector must be 1, shape (1,)
            data_reupload (bool, optional): Whether to reupload the data
                for the IEC or not, default is True.

        Returns:
            None
        """
        if inputs is None:
            # initialize to zero
            inputs = np.array([[0]])
        elif len(inputs.shape) == 1:
            # add a batch dimension
            inputs = inputs.reshape(-1, 1)

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
            if inputs.shape[1] == 1:
                qml.RX(inputs[:, 0], wires=0)
            elif inputs.shape[1] == 2:
                qml.RX(inputs[:, 0], wires=0)
                qml.RY(inputs[:, 1], wires=0)
            elif inputs.shape[1] == 3:
                qml.Rot(inputs[:, 0], inputs[:, 1], inputs[:, 2], wires=0)
            else:
                raise ValueError(
                    "The number of parameters for this IEC cannot be greater than 3"
                )

    def _circuit(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """
        Creates a circuit with noise.

        Args:
            params (np.ndarray): weight vector of shape
                [n_layers, n_qubits*n_params_per_layer]
            inputs (np.ndarray): input vector of size 1
        Returns:
            Union[float, np.ndarray]: Expectation value of PauliZ(0)
                of the circuit if state_vector is False and exp_val is True,
                otherwise the density matrix of all qubits.
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
                        self.noise_params.get("DepolarizingChannel", 0.0),
                        wires=q,
                    )

        if self.data_reupload:
            self.pqc(params[-1], self.n_qubits)

        # run mixed simualtion and get density matrix
        if self.execution_type == "density":
            return qml.density_matrix(wires=list(range(self.n_qubits)))
        # run default simulation and get expectation value
        elif self.execution_type == "expval":
            # global measurement (tensored Pauli Z, i.e. parity)
            if self.output_qubit == -1:
                return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
            # local measurement(s)
            elif isinstance(self.output_qubit, int):
                return qml.expval(qml.PauliZ(self.output_qubit))
            # n-local measurenment
            elif isinstance(self.output_qubit, list):
                obs = qml.simplify(
                    qml.Hamiltonian(
                        [1.0] * self.n_qubits,
                        [qml.PauliZ(q) for q in self.output_qubit],
                    )
                )
                return qml.expval(obs)
            else:
                raise ValueError(
                    f"Invalid parameter 'output_qubit': {self.output_qubit}.\
                        Must be int, list or -1."
                )
        # run default simulation and get probs
        elif self.execution_type == "probs":
            if self.output_qubit == -1:
                return qml.probs(wires=list(range(self.n_qubits)))
            else:
                return qml.probs(wires=self.output_qubit)
        else:
            raise ValueError(f"Invalid execution_type: {self.execution_type}.")

    def _draw(self, inputs=None) -> None:
        if isinstance(self.circuit, qml.qnn.torch.TorchLayer):
            # TODO: throws strange argument error if not catched
            return None
        result = qml.draw(self.circuit)(params=self.params, inputs=inputs)
        return result

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
        execution_type: Optional[str] = None,
        force_mean: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Perform a forward pass of the quantum circuit.

        Args:
            params (np.ndarray): Weight vector of shape
                [n_layers, n_qubits*n_params_per_layer].
            inputs (np.ndarray): Input vector of shape [1].
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None which results in the last
                set noise parameters being used.
            cache (Optional[bool], optional): Whether to cache the results.
                Defaults to False.
            execution_type (str, optional): The type of execution.
                Must be one of 'expval', 'density', or 'probs'.
                Defaults to None which results in the last set execution type
                being used.

        Returns:
            np.ndarray: The output of the quantum circuit.
                The shape depends on the execution_type.
                - If execution_type is 'expval', returns an ndarray of shape
                    (1,) if output_qubit is -1, else (len(output_qubit),).
                - If execution_type is 'density', returns an ndarray
                    of shape (2**n_qubits, 2**n_qubits).
                - If execution_type is 'probs', returns an ndarray
                    of shape (2**n_qubits,) if output_qubit is -1, else
                    (2**len(output_qubit),).
        """
        # Call forward method which handles the actual caching etc.
        return self._forward(
            params=params,
            inputs=inputs,
            noise_params=noise_params,
            cache=cache,
            execution_type=execution_type,
            force_mean=force_mean,
        )

    def _forward(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        execution_type: Optional[str] = None,
        force_mean: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Perform a forward pass of the quantum circuit.

        Args:
            params (np.ndarray): Weight vector of shape
                [n_layers, n_qubits*n_params_per_layer].
            inputs (np.ndarray): Input vector of shape [1].
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None which results in the last
                set noise parameters being used.
            cache (Optional[bool], optional): Whether to cache the results.
                Defaults to False.
            execution_type (str, optional): The type of execution.
                Must be one of 'expval', 'density', or 'probs'.
                Defaults to None which results in the last set execution type
                being used.

        Returns:
            np.ndarray: The output of the quantum circuit.
                The shape depends on the execution_type.
                - If execution_type is 'expval', returns an ndarray of shape
                    (1,) if output_qubit is -1, else (len(output_qubit),).
                - If execution_type is 'density', returns an ndarray
                    of shape (2**n_qubits, 2**n_qubits).
                - If execution_type is 'probs', returns an ndarray
                    of shape (2**n_qubits,) if output_qubit is -1, else
                    (2**len(output_qubit),).

        Raises:
            NotImplementedError: If the number of shots is not None or if the
                expectation value is True.
        """
        # set the parameters as object attributes
        if noise_params is not None:
            self.noise_params = noise_params
        if execution_type is not None:
            self.execution_type = execution_type

        # the qasm representation contains the bound parameters, thus it is ok to hash that
        hs = hashlib.md5(
            repr(
                {
                    "n_qubits": self.n_qubits,
                    "n_layers": self.n_layers,
                    "pqc": self.pqc.__class__.__name__,
                    "dru": self.data_reupload,
                    "params": params,
                    "noise_params": self.noise_params,
                    "execution_type": self.execution_type,
                    "inputs": inputs,
                    "output_qubit": self.output_qubit,
                }
            ).encode("utf-8")
        ).hexdigest()

        result: Optional[np.ndarray] = None
        if cache:
            name: str = f"pqc_{hs}.npy"

            cache_folder: str = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path: str = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                result = np.load(file_path)

        if result is None:
            # if density matrix requested or noise params used
            if self.execution_type == "density" or self.noise_params is not None:
                result = self.circuit_mixed(
                    params=params,
                    inputs=inputs,
                )
            else:
                if isinstance(self.circuit, qml.qnn.torch.TorchLayer):
                    result = self.circuit(
                        inputs=inputs,
                    )
                else:
                    result = self.circuit(
                        params=params,
                        inputs=inputs,
                    )

        if self.execution_type == "expval" and self.output_qubit == -1:
            if isinstance(result, list):
                result = np.stack(result)

            # Calculating mean value after stacking, to not
            # discard gradient information
            if force_mean:
                # exception for torch layer because it swaps batch and output dimension
                if isinstance(self.circuit, qml.qnn.torch.TorchLayer):
                    result = result.mean(axis=-1)
                else:
                    result = result.mean(axis=0)

        if cache:
            np.save(file_path, result)

        return result
