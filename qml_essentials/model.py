from typing import Dict, Optional, Tuple, Callable, Union, List
import pennylane as qml
import warnings
from copy import deepcopy
import math
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from qml_essentials.ansaetze import Gates, Ansaetze, Circuit, Encoding
from qml_essentials.ansaetze import PulseInformation as pinfo
from qml_essentials.utils import QuanTikz

import logging

log = logging.getLogger(__name__)


class Model:
    """
    A quantum circuit model.
    """

    lightning_threshold = 12
    cpu_scaler = 0.9  # default cpu scaler, =1 means full CPU for MP

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        circuit_type: Union[str, Circuit] = "No_Ansatz",
        data_reupload: Union[bool, List[List[bool]], List[List[List[bool]]]] = True,
        state_preparation: Union[
            str, Callable, List[Union[str, Callable]], None
        ] = None,
        encoding: Union[Encoding, str, Callable, List[Union[str, Callable]]] = Gates.RX,
        trainable_frequencies: bool = False,
        initialization: str = "random",
        initialization_domain: List[float] = [0, 2 * jnp.pi],
        output_qubit: Union[List[int], int] = -1,
        shots: Optional[int] = None,
        random_seed: int = 1000,
        remove_zero_encoding: bool = True,
        use_multithreading: bool = False,
        repeat_batch_axis: List[bool] = [True, True, True],
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
            data_reupload (Union[bool, List[bool], List[List[bool]]], optional):
                Whether to reupload data to the quantum device on each
                layer and qubit. Detailed re-uploading instructions can be given
                as a list/array of 0/False and 1/True with shape (n_qubits,
                n_layers) to specify where to upload the data. Defaults to True
                for applying data re-uploading to the full circuit.
            encoding (Union[str, Callable, List[str], List[Callable]], optional):
                The unitary to use for encoding the input data. Can be a string
                (e.g. "RX") or a callable (e.g. qml.RX). Defaults to qml.RX.
                If input is multidimensional it is assumed to be a list of
                unitaries or a list of strings.
            trainable_frequencies (bool, optional):
                Sets trainable encoding parameters for trainable frequencies.
                Defaults to False.
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
            remove_zero_encoding (bool, optional): whether to
                remove the zero encoding from the circuit. Defaults to True.
            run_parallel (bool, optional): whether to use JAX
                pmap instead of vmap to parallelise over batch dimension.

        Returns:
            None
        """
        # Initialize default parameters needed for circuit evaluation
        self.n_qubits: int = n_qubits
        self.output_qubit: Union[List[int], int] = output_qubit
        self.n_layers: int = n_layers
        self.noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None
        self.shots = shots
        self.remove_zero_encoding = remove_zero_encoding
        self.use_multithreading = use_multithreading
        self.trainable_frequencies: bool = trainable_frequencies
        self.execution_type: str = "expval"
        self.repeat_batch_axis: List[bool] = repeat_batch_axis

        # Initialize random key in Gates
        Gates.init_rng(random_seed)

        # --- State Preparation ---
        try:
            self._sp = Gates.parse_gates(state_preparation, Gates)
        except ValueError as e:
            raise ValueError(f"Error parsing encodings: {e}")

        # prepare corresponding pulse parameters (always optimized pulses)
        self.sp_pulse_params = []
        for sp in self._sp:
            sp_name = sp.__name__ if hasattr(sp, "__name__") else str(sp)

            if pinfo.gate_by_name(sp_name) is not None:
                params = jnp.array(pinfo.gate_by_name(sp_name))
                self.sp_pulse_params.append(params)
            else:
                # gate has no pulse parametrization
                self.sp_pulse_params.append(None)

        # --- Encoding ---
        if isinstance(encoding, Encoding):
            # user wants custom strategy? do it!
            self._enc = encoding
        else:
            # use hammming encoding by default
            self._enc = Encoding("hamming", encoding)

        # Number of possible inputs
        self.n_input_feat = len(self._enc)
        log.debug(f"Number of input features: {self.n_input_feat}")

        # Trainable frequencies, default initialization as in arXiv:2309.03279v2
        self.enc_params = jnp.ones((self.n_qubits, self.n_input_feat))

        self._zero_inputs = False

        # --- Data-Reuploading ---
        # Process data reuploading strategy and set degree
        if not isinstance(data_reupload, bool):
            if not isinstance(data_reupload, np.ndarray):
                data_reupload = np.array(data_reupload)

            if len(data_reupload.shape) == 2:
                assert data_reupload.shape == (
                    n_layers,
                    n_qubits,
                ), f"Data reuploading array has wrong shape. \
                    Expected {(n_layers, n_qubits)} or\
                    {(n_layers, n_qubits, self.n_input_feat)},\
                    got {data_reupload.shape}."
                data_reupload = data_reupload.reshape(*data_reupload.shape, 1)
                data_reupload = np.repeat(data_reupload, self.n_input_feat, axis=2)

            assert data_reupload.shape == (
                n_layers,
                n_qubits,
                self.n_input_feat,
            ), f"Data reuploading array has wrong shape. \
                Expected {(n_layers, n_qubits, self.n_input_feat)},\
                got {data_reupload.shape}."

            log.debug(f"Data reuploading array:\n{data_reupload}")
        else:
            if data_reupload:
                impl_n_layers: int = (
                    n_layers + 1
                )  # we need L+1 according to Schuld et al.
                data_reupload = np.ones((n_layers, n_qubits, self.n_input_feat))
                log.debug("Full data reuploading.")
            else:
                impl_n_layers: int = n_layers
                data_reupload = np.zeros((n_layers, n_qubits, self.n_input_feat))
                data_reupload[0][0] = 1
                log.debug("No data reuploading.")

        # convert to boolean values
        data_reupload = data_reupload.astype(bool)
        self.data_reupload = jnp.array(data_reupload)

        self.degree: Tuple = tuple(
            self._enc.get_n_freqs(jnp.count_nonzero(self.data_reupload[..., i]))
            for i in range(self.n_input_feat)
        )

        self.frequencies: Tuple = tuple(
            self._enc.get_spectrum(jnp.count_nonzero(self.data_reupload[..., i]))
            for i in range(self.n_input_feat)
        )

        self.has_dru = jnp.max(jnp.array([jnp.max(f) for f in self.frequencies])) > 1

        # check for the highest degree among all input dimensions
        if self.has_dru:
            impl_n_layers: int = n_layers + 1  # we need L+1 according to Schuld et al.
        else:
            impl_n_layers = n_layers
        log.info(f"Number of implicit layers: {impl_n_layers}.")

        # --- Ansatz ---
        # only weak check for str. We trust the user to provide sth useful
        if isinstance(circuit_type, str):
            self.pqc: Callable[[Optional[jnp.ndarray], int], int] = getattr(
                Ansaetze, circuit_type or "No_Ansatz"
            )()
        else:
            self.pqc = circuit_type()
        log.info(f"Using Ansatz {circuit_type}.")

        # calculate the shape of the parameter vector here, we will re-use this in init.
        params_per_layer = self.pqc.n_params_per_layer(self.n_qubits)
        self._params_shape: Tuple[int, int] = (impl_n_layers, params_per_layer)
        log.info(f"Parameters per layer: {params_per_layer}")

        pulse_params_per_layer = self.pqc.n_pulse_params_per_layer(self.n_qubits)
        self._pulse_params_shape: Tuple[int, int] = (
            impl_n_layers,
            pulse_params_per_layer,
        )

        self.batch_shape = (1, 1, 1)
        # this will also be re-used in the init method,
        # however, only if nothing is provided
        self._inialization_strategy = initialization
        self._initialization_domain = initialization_domain

        # ..here! where we only require a JAX random key
        self.initialize_params(random.key(random_seed))

        # Initializing pulse params
        self.pulse_params: jnp.ndarray = jnp.ones((*self._pulse_params_shape, 1))

        log.info(f"Initialized pulse parameters with shape {self.pulse_params.shape}.")

        # Initialize two circuits, one with the default device and
        # one with the mixed device
        # which allows us to later route depending on the state_vector flag
        if self.n_qubits < self.lightning_threshold:
            device = "default.qubit"
        else:
            device = "lightning.qubit"
            self.use_multithreading = False
        self.circuit: qml.QNode = qml.QNode(
            self._circuit,
            qml.device(
                device,
                shots=self.shots,
                wires=self.n_qubits,
            ),
            interface="jax-jit",
            diff_method="parameter-shift" if self.shots is not None else "best",
        )

        self.circuit_mixed: qml.QNode = qml.QNode(
            self._circuit,
            qml.device("default.mixed", shots=self.shots, wires=self.n_qubits),
            interface="jax-jit",
            diff_method="parameter-shift" if self.shots is not None else "best",
        )

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
            kvs (Optional[Dict[str, Union[float, Dict[str, float]]]]): A
            dictionary of noise parameters. If all values are 0.0, the noise
            parameters are set to None.

        Returns:
            None
        """
        # set to None if only zero values provided
        if kvs is not None and all(jnp == 0.0 for jnp in kvs.values()):
            kvs = None

        # set default values
        if kvs is not None:
            kvs.setdefault("BitFlip", 0.0)
            kvs.setdefault("PhaseFlip", 0.0)
            kvs.setdefault("Depolarizing", 0.0)
            kvs.setdefault("MultiQubitDepolarizing", 0.0)
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
                    "MultiQubitDepolarizing",
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
    def output_qubit(self) -> int:
        return self._output_qubit

    @output_qubit.setter
    def output_qubit(self, value: int) -> None:
        if isinstance(value, list):
            assert (
                len(value) <= self.n_qubits
            ), f"Size of output_qubit {len(value)} cannot be\
            larger than number of qubits {self.n_qubits}."
        elif isinstance(value, int):
            if value == -1:
                value = list(range(self.n_qubits))
            else:
                assert (
                    value < self.n_qubits
                ), f"Output qubit {value} cannot be larger than {self.n_qubits}."
                value = [value]

        self._output_qubit = value

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
        if value == "density":
            self._result_shape = (
                2 ** len(self.output_qubit),
                2 ** len(self.output_qubit),
            )
        elif value == "expval":
            # check if all qubits are used
            if len(self.output_qubit) == self.n_qubits:
                self._result_shape = (len(self.output_qubit),)
            # if not -> parity measurement with only 1D output per pair
            # or n_local measurement
            else:
                self._result_shape = (len(self.output_qubit),)
        elif value == "probs":
            # in case this is a list of parities,
            # each pair has 2^len(qubits) probabilities
            n_parity = (
                2 ** len(self.output_qubit[0])
                if isinstance(self.output_qubit[0], Tuple)
                else 2
            )
            self._result_shape = (len(self.output_qubit), n_parity)
        elif value == "state":
            self._result_shape = (2 ** len(self.output_qubit),)
        else:
            raise ValueError(f"Invalid execution type: {value}.")

        if value == "state" and not self.all_qubit_measurement:
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

    @property
    def params(self) -> jnp.ndarray:
        return self._params

    @params.setter
    def params(self, value: jnp.ndarray) -> None:
        if len(value.shape) == 2:
            value = value.reshape(*value.shape, 1)

        self._params = value

    @property
    def enc_params(self) -> jnp.ndarray:
        return self._enc_params

    @enc_params.setter
    def enc_params(self, value: jnp.ndarray) -> None:
        self._enc_params = value

    @property
    def pulse_params(self) -> jnp.ndarray:
        return self._pulse_params

    @pulse_params.setter
    def pulse_params(self, value: jnp.ndarray) -> None:
        self._pulse_params = value

    @property
    def all_qubit_measurement(self) -> bool:
        return self.output_qubit == list(range(self.n_qubits))

    def initialize_params(
        self,
        random_key: random.PRNGKey,
        repeat: int = 1,
        initialization: str = None,
        initialization_domain: List[float] = None,
    ) -> None:
        """
        Initializes the parameters of the model.

        Args:
            random_key: An initial random key for JAX to use for initialization.
            repeat: The number of times to repeat the parameters.
                If None, the number of layers is used.
            initialization: The strategy to use for parameter initialization.
                If None, the strategy specified in the constructor is used.
            initialization_domain: The domain to use for parameter initialization.
                If None, the domain specified in the constructor is used.

        Returns:
            None
        """
        # Initializing params
        params_shape = (*self._params_shape, repeat)

        # use existing strategy if not specified
        initialization = initialization or self._inialization_strategy
        initialization_domain = initialization_domain or self._initialization_domain

        def set_control_params(params: jnp.ndarray, value: float) -> jnp.ndarray:
            indices = self.pqc.get_control_indices(self.n_qubits)
            if indices is None:
                warnings.warn(
                    f"Specified {initialization} but circuit\
                    does not contain controlled rotation gates.\
                    Parameters are intialized randomly.",
                    UserWarning,
                )
            else:
                np_params = np.array(params)
                np_params[:, indices[0] : indices[1] : indices[2]] = (
                    np.ones_like(params[:, indices[0] : indices[1] : indices[2]])
                    * value
                )
                params = jnp.array(np_params)
            return params

        if initialization == "random":
            self.params: jnp.ndarray = random.uniform(
                random_key,
                params_shape,
                minval=initialization_domain[0],
                maxval=initialization_domain[1],
            )
        elif initialization == "zeros":
            self.params: jnp.ndarray = jnp.zeros(params_shape)
        elif initialization == "pi":
            self.params: jnp.ndarray = jnp.ones(params_shape) * jnp.pi
        elif initialization == "zero-controlled":
            self.params: jnp.ndarray = random.uniform(
                random_key,
                params_shape,
                minval=initialization_domain[0],
                maxval=initialization_domain[1],
            )
            self.params = set_control_params(self.params, 0)
        elif initialization == "pi-controlled":
            self.params: jnp.ndarray = random.uniform(
                random_key,
                params_shape,
                minval=initialization_domain[0],
                maxval=initialization_domain[1],
            )
            self.params = set_control_params(self.params, jnp.pi)
        else:
            raise Exception("Invalid initialization method")

        log.info(
            f"Initialized parameters with shape {self.params.shape}\
            using strategy {initialization}."
        )

        return random.split(random_key)

    def transform_input(
        self, inputs: jnp.ndarray, enc_params: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Transforms the input as in arXiv:2309.03279v2

        Args:
            inputs (jnp.ndarray): single input point of shape (1, n_input_feat)
            enc_params (jnp.ndarray): encoding weight vector of
                shape (n_qubits)

        Returns:
            jnp.ndarray: transformed input of shape (1,), linearly scaled by
            enc_params, ready for encoding
        """
        return inputs * enc_params

    def _iec(
        self,
        inputs: jnp.ndarray,
        data_reupload: jnp.ndarray,
        enc: Union[Callable, List[Callable]],
        enc_params: jnp.ndarray,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    ) -> None:
        """
        Creates an AngleEncoding using RX gates

        Args:
            inputs (jnp.ndarray): single input point of shape (1, n_input_feat)
            data_reupload (jnp.ndarray): Boolean array to indicate positions in
                the circuit for data re-uploading for the IEC, shape is
                (n_qubits, n_layers).
            enc: Callable or List[Callable]: encoding function or list of encoding
                functions
            enc_params (jnp.ndarray): encoding weight vector
                of shape [n_qubits, n_inputs]
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                The noise parameters.
        Returns:
            None
        """
        # check for zero, because due to input validation, input cannot be none
        if self.remove_zero_encoding and self._zero_inputs and self.batch_shape[0] == 1:
            return

        for q in range(self.n_qubits):
            # use the last dimension of the inputs (feature dimension)
            for idx in range(inputs.shape[-1]):
                if data_reupload[q, idx]:
                    # use elipsis to indiex only the last dimension
                    # as inputs are generally *not* qubit dependent
                    enc[idx](
                        self.transform_input(inputs[..., idx], enc_params[q, idx]),
                        wires=q,
                        noise_params=noise_params,
                    )

    def _circuit(
        self,
        params: jnp.ndarray,
        inputs: jnp.ndarray,
        pulse_params: jnp.ndarray = None,
        enc_params: Optional[jnp.ndarray] = None,
        gate_mode: str = "unitary",
    ) -> Union[float, jnp.ndarray]:
        # TODO: Is the shape of params below correct?
        """
        Creates a quantum circuit, optionally with noise or pulse simulation.

        Args:
            params (jnp.ndarray): weight vector of shape
                [n_layers, n_qubits*(n_params_per_layer+trainable_frequencies)]
            inputs (jnp.ndarray): input vector of size 1
            pulse_params Optional[jnp.ndarray]: pulse parameter scaler weights of shape
                [n_layers, n_pulse_params_per_layer]
            enc_params Optional[jnp.ndarray]: encoding weight vector
                of shape [n_qubits, n_inputs]
            gate_mode (str): Backend mode for gate execution. Can be
                "unitary" (default) or "pulse".
        Returns:
            Union[float, jnp.ndarray]: Expectation value of PauliZ(0)
                of the circuit if state_vector is False and expval is True,
                otherwise the density matrix of all qubits.
        """

        self._variational(
            params=params,
            inputs=inputs,
            pulse_params=pulse_params,
            enc_params=enc_params,
            gate_mode=gate_mode,
        )
        return self._observable()

    def _variational(
        self,
        params: jnp.ndarray,
        inputs: jnp.ndarray,
        pulse_params: Optional[jnp.ndarray] = None,
        enc_params: Optional[jnp.ndarray] = None,
        gate_mode: str = "unitary",
    ) -> None:
        """
        Builds the variational quantum circuit with state preparation,
        variational ansatz layers, and intertwined encoding layers.

        Args:
            params (jnp.ndarray): weight vector of shape
                [n_layers, n_qubits*(n_params_per_layer+trainable_frequencies)]
            inputs (jnp.ndarray): input vector of size 1
            pulse_params Optional[jnp.ndarray]: pulse parameter scaler weights of shape
                [n_layers, n_pulse_params_per_layer]
            enc_params Optional[jnp.ndarray]: encoding weight vector
                of shape [n_qubits, n_inputs]
            gate_mode (str): Backend mode for gate execution. Can be
                "unitary" (default) or "pulse".

        Returns:
            None
        """
        # TODO: rework
        if len(params.shape) > 2 and params.shape[2] == 1:
            params = params[:, :, 0]

        if len(inputs.shape) > 1 and inputs.shape[0] == 1:
            inputs = inputs[0]

        if enc_params is None:
            # TODO: Raise warning if trainable frequencies is True, or similar. I.e., no
            #   warning if user does not care for frequencies or enc_params
            if self.trainable_frequencies:
                warnings.warn(
                    "Explicit call to `_circuit` or `_variational` detected: "
                    "`enc_params` is None, using `self.enc_params` instead.",
                    RuntimeWarning,
                )
            enc_params = self.enc_params

        if pulse_params is None:
            if gate_mode == "pulse":
                warnings.warn(
                    "Explicit call to `_circuit` or `_variational` detected: "
                    "`pulse_params` is None, using `self.pulse_params` instead.",
                    RuntimeWarning,
                )
            pulse_params = self.pulse_params

        if self.noise_params is not None:
            self._apply_state_prep_noise()

        # state preparation
        for q in range(self.n_qubits):
            for _sp, sp_pulse_params in zip(self._sp, self.sp_pulse_params):
                _sp(
                    wires=q,
                    pulse_params=sp_pulse_params,
                    noise_params=self.noise_params,
                    gate_mode=gate_mode,
                )

        # circuit building
        for layer in range(0, self.n_layers):
            # ansatz layers
            self.pqc(
                params[layer],
                self.n_qubits,
                pulse_params=pulse_params[layer],
                noise_params=self.noise_params,
                gate_mode=gate_mode,
            )

            # encoding layers
            self._iec(
                inputs,
                data_reupload=self.data_reupload[layer],
                enc=self._enc,
                enc_params=enc_params,
                noise_params=self.noise_params,
            )

            # visual barrier
            if self.has_dru:
                qml.Barrier(wires=list(range(self.n_qubits)), only_visual=True)

        # final ansatz layer
        if self.has_dru:  # same check as in init
            self.pqc(
                params[self.n_layers],
                self.n_qubits,
                pulse_params=pulse_params[-1],
                noise_params=self.noise_params,
                gate_mode=gate_mode,
            )

        # channel noise
        if self.noise_params is not None:
            self._apply_general_noise()

    def _observable(self):
        # run mixed simualtion and get density matrix
        if self.execution_type == "density":
            return qml.density_matrix(wires=self.output_qubit)
        elif self.execution_type == "state":
            return qml.state()
        # run default simulation and get expectation value
        elif self.execution_type == "expval":
            # n-local measurement
            if self.all_qubit_measurement:
                return [qml.expval(qml.PauliZ(q)) for q in self.output_qubit]
            # parity or local measurement(s)
            elif isinstance(self.output_qubit, list):
                ret = []
                # list of parity pairs
                for pair in self.output_qubit:
                    if isinstance(pair, int):
                        ret.append(qml.expval(qml.PauliZ(pair)))
                    else:
                        obs = qml.PauliZ(pair[0])
                        for q in pair[1:]:
                            obs = obs @ qml.PauliZ(q)
                        ret.append(qml.expval(obs))
                return ret
            else:
                raise ValueError(
                    f"Invalid parameter `output_qubit`: {self.output_qubit}.\
                        Must be int, list or -1."
                )
        # run default simulation and get probs
        elif self.execution_type == "probs":
            # n-local measurement
            if self.all_qubit_measurement:
                return qml.probs(wires=self.output_qubit)
            # parity or local measurement(s)
            elif isinstance(self.output_qubit, list):
                ret = []
                # list of parity pairs
                for pair in self.output_qubit:
                    if isinstance(pair, int):
                        ret.append(qml.probs(wires=[pair]))
                    else:
                        ret.append(qml.probs(wires=pair))
                return ret
            else:
                raise ValueError(
                    f"Invalid parameter `output_qubit`: {self.output_qubit}.\
                        Must be int, list or -1."
                )
        else:
            raise ValueError(f"Invalid execution_type: {self.execution_type}.")

    def _apply_state_prep_noise(self) -> None:
        """
        Applies a state preparation error on each qubit according to the
        probability for StatePreparation provided in the noise_params.
        """
        p = self.noise_params.get("StatePreparation", 0.0)
        for q in range(self.n_qubits):
            if p > 0:
                qml.BitFlip(p, wires=q)

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
                circuit_depth = self._get_circuit_depth()
                tg = circuit_depth * t_factor
                qml.ThermalRelaxationError(1.0, t1, t2, tg, q)

    def _get_circuit_depth(self, inputs: Optional[jnp.ndarray] = None) -> int:
        """
        Obtain circuit depth for the model

        Args:
            inputs (Optional[jnp.ndarray]): The inputs, with which to call the
                circuit. Defaults to None.

        Returns:
            int: Circuit depth (longest path of gates in circuit.)
        """
        inputs = self._inputs_validation(inputs)
        spec_model = deepcopy(self)
        spec_model.noise_params = None  # remove noise
        specs = qml.specs(spec_model.circuit)(self.params, inputs)

        return specs["resources"].depth

    def draw(self, inputs=None, figure="text", *args, **kwargs):
        """
        Draws the quantum circuit using the specified visualization method.

        Args:
            inputs (Optional[jnp.ndarray]): Input vector for the circuit. If None,
                the default inputs are used.
            figure (str, optional): The type of figure to generate. Must be one of
                'text', 'mpl', or 'tikz'. Defaults to 'text'.
        Returns:
            Either a string, matplotlib figure or TikzFigure object (similar to string)
            depending on the chosen visualization.
        *args:
            Additional arguments to be passed to the visualization method.
        **kwargs:
            Additional keyword arguments to be passed to the visualization method.
            Can include `pulse_params`, `gate_mode`, `enc_params`, or `noise_params`.

        Raises:
            AssertionError: If the 'figure' argument is not one of the accepted values.
        """

        if not isinstance(self.circuit, qml.QNode):
            # TODO: throws strange argument error if not catched
            return ""

        assert figure in [
            "text",
            "mpl",
            "tikz",
        ], f"Invalid figure: {figure}. Must be 'text', 'mpl' or 'tikz'."

        inputs = self._inputs_validation(inputs)

        if figure == "mpl":
            result = qml.draw_mpl(self.circuit)(
                params=self.params,
                inputs=inputs,
                *args,
                **kwargs,
            )
        elif figure == "tikz":
            result = QuanTikz.build(
                self.circuit,
                params=self.params,
                inputs=inputs,
                *args,
                **kwargs,
            )
        else:
            result = qml.draw(self.circuit)(params=self.params, inputs=inputs)
        return result

    def __repr__(self) -> str:
        return self.draw(figure="text")

    def __str__(self) -> str:
        return self.draw(figure="text")

    def _params_validation(self, params) -> jnp.ndarray:
        """
        Sets the parameters when calling the quantum circuit.

        Args:
            params (jnp.ndarray): The parameters used for the call.

        Returns:
            jnp.ndarray: Validated parameters.
        """
        # append batch axis if not provided

        # TODO: replace with getter/setter
        if params is not None:
            if len(params.shape) == 2:
                params = np.expand_dims(params, axis=-1)

            self.params = params
        else:
            params = self.params

        return params

    def _pulse_params_validation(self, pulse_params) -> jnp.ndarray:
        """
        Sets the pulse parameters when calling the quantum circuit.

        Args:
            pulse_params (jnp.ndarray): The pulse parameter scalers used for the call.

        Returns:
            jnp.ndarray: Validated pulse parameters
        """
        if pulse_params is None:
            pulse_params = self.pulse_params
        else:
            self.pulse_params = pulse_params

        return pulse_params

    def _enc_params_validation(self, enc_params) -> jnp.ndarray:
        """
        Sets the encoding parameters when calling the quantum circuit

        Args:
            enc_params (jnp.ndarray): The encoding parameters used for the call
        """
        if enc_params is None:
            enc_params = self.enc_params
        else:
            if self.trainable_frequencies:
                self.enc_params = enc_params
            else:
                self.enc_params = jnp.array(enc_params)

        if len(enc_params.shape) == 1 and self.n_input_feat == 1:
            enc_params = enc_params.reshape(-1, 1)
        elif len(enc_params.shape) == 1 and self.n_input_feat > 1:
            raise ValueError(
                f"Input dimension {self.n_input_feat} >1 but \
                `enc_params` has shape {enc_params.shape}"
            )

        return enc_params

    def _inputs_validation(
        self, inputs: Union[None, List, float, int, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Validate the inputs to be a 2D numpy array of shape (batch_size, n_inputs).

        Args:
            inputs (Union[None, List, float, int, jnp.ndarray]): The input to validate.

        Returns:
            jnp.ndarray: The validated input.
        """
        self._zero_inputs = False
        if isinstance(inputs, List):
            inputs = jnp.array(np.stack(inputs))
        elif isinstance(inputs, float) or isinstance(inputs, int):
            inputs = jnp.array([inputs])
        elif inputs is None:
            inputs = jnp.array([[0] * self.n_input_feat])

        if not inputs.any():
            self._zero_inputs = True

        if len(inputs.shape) <= 1:
            if self.n_input_feat == 1:
                # add a batch dimension
                inputs = inputs.reshape(-1, 1)
            else:
                if inputs.shape[0] == self.n_input_feat:
                    inputs = inputs.reshape(1, -1)
                else:
                    inputs = inputs.reshape(-1, 1)
                    inputs = inputs.repeat(self.n_input_feat, axis=1)
                    warnings.warn(
                        f"Expected {self.n_input_feat} inputs, but {inputs.shape[0]} "
                        "was provided, replicating input for all input features.",
                        UserWarning,
                    )
        else:
            if inputs.shape[1] != self.n_input_feat:
                raise ValueError(
                    f"Wrong number of inputs provided. Expected {self.n_input_feat} "
                    f"inputs, but input has shape {inputs.shape}."
                )

        return inputs

    def _mp_executor(self, f, params, pulse_params, inputs, enc_params, gate_mode):
        """
        Execute a function f in parallel over parameters.

        Args:
            f: A function that takes two arguments, params and inputs,
                and returns a numpy array.
            params: A 3D numpy array of parameters where the first dimension is
                the layer index, the second dimension is the parameter index in
                the layer, and the third dimension is the sample index.
            pulse_params (jnp.ndarray): array of pulse parameter scalers for pulse-mode
                gates.
            inputs: A 2D numpy array of inputs where the first dimension is
                the sample index and the second dimension is the input feature index.
            enc_params: A 1D numpy array of encoding parameters where the dimension is
                the qubit index.
            gate_mode (str): Mode for gate execution ("unitary" or "pulse").

        Returns:
            A numpy array of the output of f applied to each batch of
            samples in params, enc_params, and inputs.
        """
        combined_batch_size = math.prod(self.batch_shape)

        if (
            gate_mode == "pulse" or self.use_multithreading
        ) and combined_batch_size > 1:
            # wrapper to allow kwargs (not supported by jax)
            result = jax.vmap(
                f,
                in_axes=(
                    2 if self.batch_shape[1] > 1 else None,
                    0 if self.batch_shape[0] > 1 else None,
                    2 if self.batch_shape[2] > 1 else None,
                    None,
                    None,
                ),
            )(params, inputs, pulse_params, enc_params, gate_mode)
        else:
            result = f(
                params=params,
                pulse_params=pulse_params,
                inputs=inputs,
                enc_params=enc_params,
                gate_mode=gate_mode,
            )

        return self._postprocess_res(result)

    def _postprocess_res(self, result: Union[list, jnp.ndarray]) -> jnp.ndarray:
        """
        Reshapes results for uniformity.

        Args:
            result (Union[list, jnp.ndarray]): result of a computation

        Returns:
            jnp.ndarray: Result with shape [B_P, ...]
        """
        if isinstance(result, list):
            # we use moveaxis here because in case of parity measure,
            # there is another dimension appended to the end and
            # simply transposing would result in a wrong shape
            if isinstance(result[0], jnp.ndarray):
                result = jnp.stack(result)
                if len(result.shape) > 1:
                    result = jnp.moveaxis(result, 0, 1)
            else:
                result = jnp.stack(result)
                if len(result.shape) > 1:
                    result = jnp.moveaxis(result, 0, 1)
        return result

    def _assimilate_batch(self, inputs, params, pulse_params):
        """
        inputs:        [B_I, ...]
        params:        [..., ..., B_P]
        pulse_params:  [..., ..., B_R]

        Returns:
        inputs_      [B_I * B_P * B_R, ...]
        params_      [..., ..., B_I * B_P * B_R]
        pulse_params_[..., ..., B_I * B_P * B_R]
        batch_shape  (B_I, B_P, B_R)
        """
        B_I = inputs.shape[0]
        # we check for the product because there is a chance that
        # there are no params. In this case we want B_P to be 1
        B_P = 1 if 0 in params.shape else params.shape[-1]
        B_R = pulse_params.shape[-1]

        batch_shape = (B_I, B_P, B_R)
        batch_shape_enabled = np.array(batch_shape) * self.repeat_batch_axis
        batch_shape_enabled = batch_shape_enabled[batch_shape_enabled != 0]
        B = np.prod(batch_shape_enabled)

        # [B_I, ...] -> [B_I, B_P, B_R, ...] -> [B, ...]
        if B_I > 1 and self.repeat_batch_axis[0]:
            if self.repeat_batch_axis[1]:
                inputs = jnp.repeat(inputs[:, None, None, ...], B_P, axis=1)
            if self.repeat_batch_axis[2]:
                inputs = jnp.repeat(inputs, B_R, axis=2)
            inputs = inputs.reshape(B, *inputs.shape[3:])

        # [..., ..., B_P] -> [..., ..., B_I, B_P, B_R] -> [..., ..., B]
        if B_P > 1 and self.repeat_batch_axis[1]:
            # add B_I axis before last, and B_R axis after last
            params = params[..., None, :, None]  # [..., B_I(=1), B_P, B_R(=1)]
            if self.repeat_batch_axis[0]:
                params = jnp.repeat(params, B_I, axis=-3)  # [..., B_I, B_P, 1]
            if self.repeat_batch_axis[2]:
                params = jnp.repeat(params, B_R, axis=-1)  # [..., B_I, B_P, B_R]
            params = params.reshape(*params.shape[:-3], B)

        # [..., ..., B_R] -> [..., ..., B_I, B_P, B_R] -> [..., ..., B]
        if B_R > 1 and self.repeat_batch_axis[2]:
            # add B_I axis before last, and B_P axis before last (after adding B_I)
            pulse_params = pulse_params[
                ..., None, None, :
            ]  # [..., B_I(=1), B_P(=1), B_R]
            if self.repeat_batch_axis[0]:
                pulse_params = jnp.repeat(
                    pulse_params, B_I, axis=-3
                )  # [..., B_I, 1, B_R]
            if self.repeat_batch_axis[1]:
                pulse_params = jnp.repeat(
                    pulse_params, B_P, axis=-2
                )  # [..., B_I, B_P, B_R]
            pulse_params = pulse_params.reshape(*pulse_params.shape[:-3], B)

        return inputs, params, pulse_params, batch_shape, batch_shape_enabled

    def _requires_density(self):
        """
        Checks if the current model requires density matrix simulation or not
        based on the noise_params variable and the execution type

        Returns:
            bool: True if model requires density simulation
        """
        if self.execution_type == "density":
            return True

        if self.noise_params is not None:
            coherent_noise = ["GateError"]
            for k, v in self.noise_params.items():
                if k in coherent_noise:
                    continue
                if v is not None and v > 0:
                    return True
        return False

    def __call__(
        self,
        params: Optional[jnp.ndarray] = None,
        inputs: Optional[jnp.ndarray] = None,
        pulse_params: Optional[jnp.ndarray] = None,
        enc_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        execution_type: Optional[str] = None,
        force_mean: bool = False,
        gate_mode: str = "unitary",
    ) -> jnp.ndarray:
        """
        Perform a forward pass of the quantum circuit with optional noise or
        pulse level simulation.

        Args:
            params (Optional[jnp.ndarray]): Weight vector of shape
                [n_layers, n_qubits*n_params_per_layer].
                If None, model internal parameters are used.
            inputs (Optional[jnp.ndarray]): Input vector of shape [1].
                If None, zeros are used.
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers for pulse-mode
                gates.
            enc_params (Optional[jnp.ndarray]): Weight vector of shape
                [n_qubits, n_input_features]. If None, model internal encoding
                parameters are used.
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None which results in the last
                set noise parameters being used.
            execution_type (str, optional): The type of execution.
                Must be one of 'expval', 'density', or 'probs'.
                Defaults to None which results in the last set execution type
                being used.
            force_mean (bool, optional): Whether to average
                when performing n-local measurements.
                Defaults to False.
            gate_mode (str, optional): Gate backend mode ("unitary" or "pulse").
                Defaults to "unitary".

        Returns:
            jnp.ndarray: The output of the quantum circuit.
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
            pulse_params=pulse_params,
            enc_params=enc_params,
            noise_params=noise_params,
            execution_type=execution_type,
            force_mean=force_mean,
            gate_mode=gate_mode,
        )

    def _forward(
        self,
        params: Optional[jnp.ndarray] = None,
        inputs: Optional[jnp.ndarray] = None,
        pulse_params: Optional[jnp.ndarray] = None,
        enc_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        execution_type: Optional[str] = None,
        force_mean: bool = False,
        gate_mode: str = "unitary",
    ) -> jnp.ndarray:
        """
        Perform a forward pass of the quantum circuit.

        Args:
            params (Optional[jnp.ndarray]): Weight vector of shape
                [n_layers, n_qubits*n_params_per_layer].
                If None, model internal parameters are used.
            inputs (Optional[jnp.ndarray]): Input vector of shape [1].
                If None, zeros are used.
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers for pulse-mode
                gates.
            enc_params (Optional[jnp.ndarray]): Weight vector of shape
                [n_qubits, n_input_features]. If None, model internal encoding
                parameters are used.
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None which results in the last
                set noise parameters being used.
            execution_type (str, optional): The type of execution.
                Must be one of 'expval', 'density', or 'probs'.
                Defaults to None which results in the last set execution type
                being used.
            force_mean (bool, optional): Whether to average
                when performing n-local measurements.
                Defaults to False.
            gate_mode (str, optional): Gate backend mode ("unitary" or "pulse").
                Defaults to "unitary".


        Returns:
            jnp.ndarray: The output of the quantum circuit.
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
            ValueError:
                - If `pulse_params` are provided but `gate_mode` is not "pulse".
                - If `noise_params` are provided while `gate_mode` is "pulse" (noise
                not supported in pulse mode).
        """
        # set the parameters as object attributes
        if noise_params is not None:
            self.noise_params = noise_params
        if execution_type is not None:
            self.execution_type = execution_type
        self.gate_mode = gate_mode

        # consistency checks
        if pulse_params is not None and gate_mode != "pulse":
            raise ValueError(
                "pulse_params were provided but gate_mode is not 'pulse'. "
                "Either switch gate_mode='pulse' or do not pass pulse_params."
            )

        if noise_params is not None and gate_mode == "pulse":
            raise ValueError(
                "Noise is not supported in 'pulse' gate_mode. "
                "Either remove noise_params or use gate_mode='unitary'."
            )

        params = self._params_validation(params)
        pulse_params = self._pulse_params_validation(pulse_params)
        inputs = self._inputs_validation(inputs)
        enc_params = self._enc_params_validation(enc_params)

        inputs, params, pulse_params, self.batch_shape, batch_shape_enabled = (
            self._assimilate_batch(
                inputs,
                params,
                pulse_params,
            )
        )

        result: Optional[jnp.ndarray] = None

        # if density matrix requested or noise params used
        if self._requires_density():
            result = self._mp_executor(
                f=self.circuit_mixed,
                params=params,
                pulse_params=pulse_params,
                inputs=inputs,
                enc_params=enc_params,
                gate_mode=gate_mode,
            )
        else:
            if not isinstance(self.circuit, qml.QNode):
                result = self.circuit(
                    inputs=inputs,
                )
            else:
                result = self._mp_executor(
                    f=self.circuit,
                    params=params,
                    pulse_params=pulse_params,
                    inputs=inputs,
                    enc_params=enc_params,
                    gate_mode=gate_mode,
                )

        result = result.reshape((*batch_shape_enabled, *self._result_shape)).squeeze()

        if (
            (self.execution_type == "expval" or self.execution_type == "probs")
            and force_mean
            and len(result.shape) > 0
            and self._result_shape[0] > 1
        ):
            result = result.mean(axis=-1)

        return result
