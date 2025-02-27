from typing import Dict, Optional, Tuple, Callable, Union, List
import pennylane as qml
import pennylane.numpy as np
import hashlib
import os
import warnings
from autograd.numpy import numpy_boxes
from copy import deepcopy

from qml_essentials.ansaetze import Gates, Ansaetze, Circuit
from qml_essentials.utils import PauliCircuit


import logging

log = logging.getLogger(__name__)


class Model:
    """
    A quantum circuit model.
    """

    lightning_threshold = 12

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        circuit_type: Union[str, Circuit],
        data_reupload: bool = True,
        encoding: Union[str, Callable, List[str], List[Callable]] = Gates.RX,
        initialization: str = "random",
        initialization_domain: List[float] = [0, 2 * np.pi],
        output_qubit: Union[List[int], int] = -1,
        shots: Optional[int] = None,
        random_seed: int = 1000,
        as_pauli_circuit: bool = False,
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
            circuit_type (str, Circuit): The type of quantum circuit to use.
                If None, defaults to "no_ansatz".
            data_reupload (bool, optional): Whether to reupload data to the
                quantum device on each measurement. Defaults to True.
            encoding (Union[str, Callable, List[str], List[Callable]], optional):
                The unitary to use for encoding the input data. Can be a string
                (e.g. "RX") or a callable (e.g. qml.RX). Defaults to qml.RX.
                If input is multidimensional it is assumed to be a list of
                unitaries or a list of strings.
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
                in initialization is "random" and for random noise parameters.
                Defaults to 1000.
            as_pauli_circuit (bool, optional): whether the circuit is
                transformed to a Pauli-Clifford circuit as described by Nemkov
                et al. (https://doi.org/10.1103/PhysRevA.108.032406), which is
                required for analytical Fourier coefficient computation.
                Defaults to False.

        Returns:
            None
        """
        # Initialize default parameters needed for circuit evaluation
        self.noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None
        self.execution_type: Optional[str] = "expval"
        self.shots = shots

        if isinstance(output_qubit, list):
            assert (
                len(output_qubit) <= n_qubits
            ), f"Size of output_qubit {len(output_qubit)} cannot be\
            larger than number of qubits {n_qubits}."
        self.output_qubit: Union[List[int], int] = output_qubit

        # Copy the parameters
        self.n_qubits: int = n_qubits
        self.n_layers: int = n_layers
        self.data_reupload: bool = data_reupload

        # Initialize ansatz
        # only weak check for str. We trust the user to provide sth useful
        if isinstance(circuit_type, str):
            self.pqc: Callable[[Optional[np.ndarray], int], int] = getattr(
                Ansaetze, circuit_type or "No_Ansatz"
            )()
        else:
            self.pqc = circuit_type()

        # Initialize rng in Gates
        Gates.init_rng(random_seed)

        # Initialize encoding
        # first check if we have a str, list or callable
        if isinstance(encoding, str):
            # if str, use the pennylane fct
            self._enc = getattr(Gates, f"{encoding}")
        elif isinstance(encoding, list):
            # if list, check if str or callable
            if isinstance(encoding[0], str):
                self._enc = [getattr(Gates, f"{enc}") for enc in encoding]
            else:
                self._enc = encoding

            if len(self._enc) == 1:
                self._enc = self._enc[0]
        else:
            # default to callable
            self._enc = encoding

        log.info(f"Using {circuit_type} circuit.")

        if data_reupload:
            impl_n_layers: int = n_layers + 1  # we need L+1 according to Schuld et al.
            self.degree = n_layers * n_qubits
        else:
            impl_n_layers: int = n_layers
            self.degree = 1

        log.info(f"Number of implicit layers set to {impl_n_layers}.")
        # calculate the shape of the parameter vector here, we will re-use this in init.
        self._params_shape: Tuple[int, int] = (
            impl_n_layers,
            self.pqc.n_params_per_layer(self.n_qubits),
        )
        # this will also be re-used in the init method,
        # however, only if nothing is provided
        self._inialization_strategy = initialization
        self._initialization_domain = initialization_domain

        # ..here! where we only require a rng
        self.initialize_params(np.random.default_rng(random_seed))

        # Initialize two circuits, one with the default device and
        # one with the mixed device
        # which allows us to later route depending on the state_vector flag
        self.as_pauli_circuit = as_pauli_circuit

        self.circuit_mixed: qml.QNode = qml.QNode(
            self._circuit,
            qml.device("default.mixed", shots=self.shots, wires=self.n_qubits),
        )

    @property
    def as_pauli_circuit(self) -> bool:
        return self._as_pauli_circuit

    @as_pauli_circuit.setter
    def as_pauli_circuit(self, value: bool) -> None:
        self._as_pauli_circuit = value

        self.circuit: qml.QNode = qml.QNode(
            self._circuit,
            qml.device(
                (
                    "default.qubit"
                    if self.n_qubits < self.lightning_threshold
                    else "lightning.qubit"
                ),
                shots=self.shots,
                wires=self.n_qubits,
            ),
            interface="autograd" if self.shots is not None else "auto",
            diff_method="parameter-shift" if self.shots is not None else "best",
        )

        if value:
            pauli_circuit_transform = qml.transform(
                PauliCircuit.from_parameterised_circuit
            )
            self.circuit = pauli_circuit_transform(self.circuit)

    @property
    def noise_params(self) -> Optional[Dict[str, Union[float, Dict[str, float]]]]:
        """
        Gets the noise parameters of the model.

        Returns:
            Optional[Dict[str, float]]: A dictionary of
            noise parameters or None if not set.
        """
        return self._noise_params

    @noise_params.setter
    def noise_params(
        self, kvs: Optional[Dict[str, Union[float, Dict[str, float]]]]
    ) -> None:
        """
        Sets the noise parameters of the model.

        Typically a "noise parameter" refers to the error probability.
        ThermalRelaxation is a special case, and supports a dict as value with
        structure:
            "ThermalRelaxation":
            {
                "t1": 2000, # relative t1 time.
                "t2": 1000, # relative t2 time
                "t_factor" 1: # relative gate time factor
            },

        Args:
            value (Optional[Dict[str, Union[float, Dict[str, float]]]]): A
            dictionary of noise parameters. If all values are 0.0, the noise
            parameters are set to None.

        Returns:
            None
        """
        # set to None if only zero values provided
        if kvs is not None and all(np == 0.0 for np in kvs.values()):
            kvs = None

        # set default values
        if kvs is not None:
            kvs.setdefault("BitFlip", 0.0)
            kvs.setdefault("PhaseFlip", 0.0)
            kvs.setdefault("Depolarizing", 0.0)
            kvs.setdefault("AmplitudeDamping", 0.0)
            kvs.setdefault("PhaseDamping", 0.0)
            kvs.setdefault("GateError", 0.0)
            kvs.setdefault("ThermalRelaxation", None)
            kvs.setdefault("StatePreparation", 0.0)
            kvs.setdefault("Measurement", 0.0)

            # check if there are any keys not supported
            for key in kvs.keys():
                if key not in [
                    "BitFlip",
                    "PhaseFlip",
                    "Depolarizing",
                    "AmplitudeDamping",
                    "PhaseDamping",
                    "GateError",
                    "ThermalRelaxation",
                    "StatePreparation",
                    "Measurement",
                ]:
                    warnings.warn(
                        f"Noise type {key} is not supported by this package",
                        UserWarning,
                    )

            # check valid params for thermal relaxation noise channel
            tr_params = kvs["ThermalRelaxation"]
            if isinstance(tr_params, dict):
                tr_params.setdefault("t1", 0.0)
                tr_params.setdefault("t2", 0.0)
                tr_params.setdefault("t_factor", 0.0)
                for k in tr_params.keys():
                    if k not in [
                        "t1",
                        "t2",
                        "t_factor",
                    ]:
                        warnings.warn(
                            f"Thermal Relaxation parameter {k} is not supported "
                            f"by this package",
                            UserWarning,
                        )
                if not all(tr_params.values()) or tr_params["t2"] > 2 * tr_params["t1"]:
                    warnings.warn(
                        "Received invalid values for Thermal Relaxation noise "
                        "parameter. Thermal relaxation is not applied!",
                        UserWarning,
                    )
                    kvs["ThermalRelaxation"] = 0.0

        self._noise_params = kvs

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
        if value not in ["density", "state", "expval", "probs"]:
            raise ValueError(f"Invalid execution type: {value}.")

        if (value == "density" or value == "state") and self.output_qubit != -1:
            warnings.warn(
                f"{value} measurement does ignore output_qubit, which is "
                f"{self.output_qubit}.",
                UserWarning,
            )

        if value == "probs" and self.shots is None:
            warnings.warn(
                "Setting execution_type to probs without specifying shots.",
                UserWarning,
            )

        if value == "density" and self.shots is not None:
            warnings.warn(
                "Setting execution_type to density with specified shots.",
                UserWarning,
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
            value (Optional[int]): The number of shots.
            If an integer less than or equal to 0 is provided, it is set to None.

        Returns:
            None
        """
        if type(value) is int and value <= 0:
            value = None
        self._shots = value

    def initialize_params(
        self,
        rng,
        repeat: int = None,
        initialization: str = None,
        initialization_domain: List[float] = None,
    ) -> None:
        """
        Initializes the parameters of the model.

        Args:
            rng: A random number generator to use for initialization.
            repeat: The number of times to repeat the parameters.
                If None, the number of layers is used.
            initialization: The strategy to use for parameter initialization.
                If None, the strategy specified in the constructor is used.
            initialization_domain: The domain to use for parameter initialization.
                If None, the domain specified in the constructor is used.

        Returns:
            None
        """
        params_shape = (
            self._params_shape if repeat is None else [*self._params_shape, repeat]
        )
        # use existing strategy if not specified
        initialization = initialization or self._inialization_strategy
        initialization_domain = initialization_domain or self._initialization_domain

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

        if initialization == "random":
            self.params: np.ndarray = rng.uniform(
                *initialization_domain, params_shape, requires_grad=True
            )
        elif initialization == "zeros":
            self.params: np.ndarray = np.zeros(params_shape, requires_grad=True)
        elif initialization == "pi":
            self.params: np.ndarray = np.ones(params_shape, requires_grad=True) * np.pi
        elif initialization == "zero-controlled":
            self.params: np.ndarray = rng.uniform(
                *initialization_domain, params_shape, requires_grad=True
            )
            self.params = set_control_params(self.params, 0)
        elif initialization == "pi-controlled":
            self.params: np.ndarray = rng.uniform(
                *initialization_domain, params_shape, requires_grad=True
            )
            self.params = set_control_params(self.params, np.pi)
        else:
            raise Exception("Invalid initialization method")

        log.info(
            f"Initialized parameters with shape {self.params.shape}\
            using strategy {initialization}."
        )

    def _iec(
        self,
        inputs: np.ndarray,
        data_reupload: bool,
        enc: Union[Callable, List[Callable]],
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
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
        # check for zero, because due to input validation, input cannot be none
        if not inputs.any():
            return

        if data_reupload:
            if inputs.shape[1] == 1:
                for q in range(self.n_qubits):
                    enc(inputs[:, 0], wires=q, noise_params=noise_params)
            else:
                for q in range(self.n_qubits):
                    for idx in range(inputs.shape[1]):
                        enc[idx](inputs[:, idx], wires=q, noise_params=noise_params)
        else:
            if inputs.shape[1] == 1:
                enc(inputs[:, 0], wires=0, noise_params=noise_params)
            else:
                for idx in range(inputs.shape[1]):
                    enc[idx](inputs[:, idx], wires=0, noise_params=noise_params)

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
        if self.noise_params is not None:
            self._apply_state_prep_noise()

        for layer in range(0, self.n_layers):
            self.pqc(params[layer], self.n_qubits, noise_params=self.noise_params)

            if self.data_reupload or layer == 0:
                self._iec(
                    inputs,
                    data_reupload=self.data_reupload,
                    enc=self._enc,
                    noise_params=self.noise_params,
                )

            qml.Barrier(wires=list(range(self.n_qubits)), only_visual=True)

        if self.data_reupload:
            self.pqc(params[-1], self.n_qubits, noise_params=self.noise_params)

        if self.noise_params is not None:
            self._apply_general_noise()

        # run mixed simualtion and get density matrix
        if self.execution_type == "density":
            return qml.density_matrix(wires=list(range(self.n_qubits)))
        elif self.execution_type == "state":
            return qml.state()
        # run default simulation and get expectation value
        elif self.execution_type == "expval":
            # n-local measurement
            if self.output_qubit == -1:
                return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
            # local measurement(s)
            elif isinstance(self.output_qubit, int):
                return qml.expval(qml.PauliZ(self.output_qubit))
            # parity measurenment
            elif isinstance(self.output_qubit, list):
                obs = qml.PauliZ(self.output_qubit[0])
                for out_qubit in self.output_qubit[1:]:
                    obs = obs @ qml.PauliZ(out_qubit)
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

    def _apply_state_prep_noise(self) -> None:
        """
        Applies a state preparation error on each qubit according to the
        probability for StatePreparation provided in the noise_params.
        """
        sp = self.noise_params.get("StatePreparation", 0.0)
        for q in range(self.n_qubits):
            if sp > 0:
                qml.BitFlip(sp, wires=q)

    def _apply_general_noise(self) -> None:
        """
        Applies general types of noise the full circuit (in contrast to gate
        errors, applied directly at gate level, see Gates.Noise).

        Possible types of noise are:
            - AmplitudeDamping (specified through probability)
            - PhaseDamping (specified through probability)
            - ThermalRelaxation (specified through a dict, containing keys
                                 "t1", "t2", "t_factor")
            - Measurement (specified through probability)
        """
        amp_damp = self.noise_params.get("AmplitudeDamping", 0.0)
        phase_damp = self.noise_params.get("PhaseDamping", 0.0)
        thermal_relax = self.noise_params.get("ThermalRelaxation", 0.0)
        meas = self.noise_params.get("Measurement", 0.0)
        for q in range(self.n_qubits):
            if amp_damp > 0:
                qml.AmplitudeDamping(amp_damp, wires=q)
            if phase_damp > 0:
                qml.PhaseDamping(phase_damp, wires=q)
            if meas > 0:
                qml.BitFlip(meas, wires=q)
            if isinstance(thermal_relax, dict):
                t1 = thermal_relax["t1"]
                t2 = thermal_relax["t2"]
                t_factor = thermal_relax["t_factor"]
                circuit_depth = self.get_circuit_depth()
                tg = circuit_depth * t_factor
                qml.ThermalRelaxationError(1.0, t1, t2, tg, q)

    def _draw(self, inputs=None, figure=False) -> None:
        if not isinstance(self.circuit, qml.QNode):
            # TODO: throws strange argument error if not catched
            return ""

        inputs = self._inputs_validation(inputs)

        if figure:
            result = qml.draw_mpl(self.circuit)(params=self.params, inputs=inputs)
        else:
            result = qml.draw(self.circuit)(params=self.params, inputs=inputs)
        return result

    def draw(self, inputs=None, figure=False) -> None:

        return self._draw(inputs, figure)

    def __repr__(self) -> str:
        return self._draw(figure=False)

    def __str__(self) -> str:
        return self._draw(figure=False)

    def __call__(
        self,
        params: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        cache: Optional[bool] = False,
        execution_type: Optional[str] = None,
        force_mean: bool = False,
    ) -> np.ndarray:
        """
        Perform a forward pass of the quantum circuit.

        Args:
            params (Optional[np.ndarray]): Weight vector of shape
                [n_layers, n_qubits*n_params_per_layer].
                If None, model internal parameters are used.
            inputs (Optional[np.ndarray]): Input vector of shape [1].
                If None, zeros are used.
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None which results in the last
                set noise parameters being used.
            cache (Optional[bool], optional): Whether to cache the results.
                Defaults to False.
            execution_type (str, optional): The type of execution.
                Must be one of 'expval', 'density', or 'probs'.
                Defaults to None which results in the last set execution type
                being used.
            force_mean (bool, optional): Whether to average
                when performing n-local measurements.
                Defaults to False.

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

    def _params_validation(self, params) -> np.ndarray:
        """
        Sets the parameters when calling the quantum circuit

        Args:
            params (np.ndarray): The parameters used for the call
        """
        if params is None:
            params = self.params
        else:
            if numpy_boxes.ArrayBox == type(params):
                self.params = params._value
            else:
                self.params = params
        return params

    def _inputs_validation(
        self, inputs: Union[None, List, float, int, np.ndarray]
    ) -> np.ndarray:
        """
        Validate the inputs to be a 2D numpy array of shape (batch_size, n_inputs).

        Args:
            inputs (Union[None, List, float, int, np.ndarray]): The input to validate.

        Returns:
            np.ndarray: The validated input.
        """
        if inputs is None:
            # initialize to zero
            inputs = np.array([[0]])
        elif isinstance(inputs, List):
            inputs = np.stack(inputs)
        elif isinstance(inputs, float) or isinstance(inputs, int):
            inputs = np.array([inputs])

        if len(inputs.shape) == 1:
            if isinstance(self._enc, List):
                inputs = inputs.reshape(-1, 1)
            else:
                # add a batch dimension
                inputs = inputs.reshape(inputs.shape[0], 1)

        return inputs

    def _forward(
        self,
        params: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        cache: Optional[bool] = False,
        execution_type: Optional[str] = None,
        force_mean: bool = False,
    ) -> np.ndarray:
        """
        Perform a forward pass of the quantum circuit.

        Args:
            params (Optional[np.ndarray]): Weight vector of shape
                [n_layers, n_qubits*n_params_per_layer].
                If None, model internal parameters are used.
            inputs (Optional[np.ndarray]): Input vector of shape [1].
                If None, zeros are used.
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None which results in the last
                set noise parameters being used.
            cache (Optional[bool], optional): Whether to cache the results.
                Defaults to False.
            execution_type (str, optional): The type of execution.
                Must be one of 'expval', 'density', or 'probs'.
                Defaults to None which results in the last set execution type
                being used.
            force_mean (bool, optional): Whether to average
                when performing n-local measurements.
                Defaults to False.

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

        params = self._params_validation(params)
        inputs = self._inputs_validation(inputs)

        # the qasm representation contains the bound parameters,
        # thus it is ok to hash that
        hs = hashlib.md5(
            repr(
                {
                    "n_qubits": self.n_qubits,
                    "n_layers": self.n_layers,
                    "pqc": self.pqc.__class__.__name__,
                    "dru": self.data_reupload,
                    "params": self.params,  # use safe-params
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
                    params=params,  # use arraybox params
                    inputs=inputs,
                )
            else:
                if not isinstance(self.circuit, qml.QNode):
                    result = self.circuit(
                        inputs=inputs,
                    )
                else:
                    result = self.circuit(
                        params=params,  # use arraybox params
                        inputs=inputs,
                    )

        if isinstance(result, list):
            result = np.stack(result)

        if self.execution_type == "expval" and force_mean and self.output_qubit == -1:
            # exception for torch layer because it swaps batch and output dimension
            if not isinstance(self.circuit, qml.QNode):
                result = result.mean(axis=-1)
            else:
                result = result.mean(axis=0)
        elif self.execution_type == "probs" and force_mean and self.output_qubit == -1:
            # exception for torch layer because it swaps batch and output dimension
            if not isinstance(self.circuit, qml.QNode):
                result = result[..., -1].sum(axis=-1)
            else:
                result = result[1:, ...].sum(axis=0)

        if len(result.shape) == 3 and result.shape[0] == 1:
            result = result[0]

        if cache:
            np.save(file_path, result)

        return result

    def get_specs(self, inputs: Optional[np.ndarray] = None) -> dict:
        """
        Get pennylane specs for the model.

        Args:
            inputs (Optional[np.ndarray]): The inputs, with which to call the
                circuit. Defaults to None.

        Returns:
            dict: Dictionary of specs. The key "resources" contains information
                about the circuit size and gate statistics.
        """
        inputs = self._inputs_validation(inputs)
        spec_model = deepcopy(self)
        spec_model.noise_params = None  # remove noise
        return qml.specs(spec_model.circuit)(self.params, inputs)

    def get_circuit_depth(self, inputs: Optional[np.ndarray] = None) -> int:
        """
        Obtain circuit depth for the model

        Args:
            inputs (Optional[np.ndarray]): The inputs, with which to call the
                circuit. Defaults to None.

        Returns:
            int: Circuit depth (longest path of gates in circuit.)
        """
        return self.get_specs(inputs)["resources"].depth
