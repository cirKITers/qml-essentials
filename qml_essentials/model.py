from typing import Any, Dict, Optional, Tuple, Callable, Union, List

from qml_essentials import operations as op
from qml_essentials import yaqsi as ys
import warnings
from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from qml_essentials.ansaetze import Ansaetze, Circuit, Encoding
from qml_essentials.gates import Gates
from qml_essentials.gates import PulseInformation as pinfo
from qml_essentials.utils import QuanTikz, safe_random_split

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
                (e.g. "RX") or a callable (e.g. op.RX). Defaults to op.RX.
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
            use_multithreading (bool, optional): whether to use JAX
                multithreading to parallelise over batch dimension.

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
                self.sp_pulse_params.append(pinfo.gate_by_name(sp_name).params)
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
        # Keep as NumPy array (not JAX) so that ``if data_reupload[q, idx]``
        # in _iec remains a concrete Python bool even under jax.jit tracing.
        self.data_reupload = np.array(data_reupload)

        self.degree: Tuple = tuple(
            self._enc.get_n_freqs(np.count_nonzero(self.data_reupload[..., i]))
            for i in range(self.n_input_feat)
        )

        self.frequencies: Tuple = tuple(
            self._enc.get_spectrum(np.count_nonzero(self.data_reupload[..., i]))
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

        # intialize to None as we can't know this yet
        self._batch_shape = None

        # this will also be re-used in the init method,
        # however, only if nothing is provided
        self._inialization_strategy = initialization
        self._initialization_domain = initialization_domain

        # ..here! where we only require a JAX random key
        self.random_key = self.initialize_params(random.key(random_seed))

        # Initializing pulse params
        self.pulse_params: jnp.ndarray = jnp.ones((*self._pulse_params_shape, 1))

        log.info(f"Initialized pulse parameters with shape {self.pulse_params.shape}.")

        # Initialise the yaqsi QuantumScript that wraps _variational.
        # No device selection needed — yaqsi auto-routes between statevector
        # and density-matrix simulation based on whether noise channels are
        # present on the tape.
        self.script = ys.QuantumScript(f=self._variational, n_qubits=self.n_qubits)

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
        if kvs is not None and all(v == 0.0 for v in kvs.values()):
            kvs = None

        # set default values
        if kvs is not None:
            defaults = {
                "BitFlip": 0.0,
                "PhaseFlip": 0.0,
                "Depolarizing": 0.0,
                "MultiQubitDepolarizing": 0.0,
                "AmplitudeDamping": 0.0,
                "PhaseDamping": 0.0,
                "GateError": 0.0,
                "ThermalRelaxation": None,
                "StatePreparation": 0.0,
                "Measurement": 0.0,
            }
            for key, default_val in defaults.items():
                kvs.setdefault(key, default_val)

            # check if there are any keys not supported
            for key in kvs.keys():
                if key not in defaults:
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
                valid_tr_keys = {"t1", "t2", "t_factor"}
                for k in tr_params.keys():
                    if k not in valid_tr_keys:
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
    def output_qubit(self) -> List[int]:
        """Get the output qubit indices for measurement."""
        return self._output_qubit

    @output_qubit.setter
    def output_qubit(self, value: Union[int, List[int]]) -> None:
        """
        Set the output qubit(s) for measurement.

        Args:
            value: Qubit index or list of indices. Use -1 for all qubits.
        """
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
            raise ValueError("Setting execution_type to density with shots not None.")

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
        """Get the variational parameters of the model."""
        return self._params

    @params.setter
    def params(self, value: jnp.ndarray) -> None:
        """Set the variational parameters, ensuring batch dimension exists."""
        if len(value.shape) == 2:
            value = value.reshape(*value.shape, 1)

        self._params = value

    @property
    def enc_params(self) -> jnp.ndarray:
        """Get the encoding parameters used for input transformation."""
        return self._enc_params

    @enc_params.setter
    def enc_params(self, value: jnp.ndarray) -> None:
        """Set the encoding parameters."""
        self._enc_params = value

    @property
    def pulse_params(self) -> jnp.ndarray:
        """Get the pulse parameters for pulse-mode gate execution."""
        return self._pulse_params

    @pulse_params.setter
    def pulse_params(self, value: jnp.ndarray) -> None:
        """Set the pulse parameters."""
        self._pulse_params = value

    @property
    def all_qubit_measurement(self) -> bool:
        """Check if measurement is performed on all qubits."""
        return self.output_qubit == list(range(self.n_qubits))

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """
        Get the batch shape (B_I, B_P, B_R).
        If the model was not called before,
        it returns (1, 1, 1).

        Returns:
            Tuple[int, ...]: Tuple of (input_batch, param_batch, pulse_batch).
                Returns (1, 1, 1) if model has not been called yet.
        """
        if self._batch_shape is None:
            log.debug("Model was not called yet. Returning (1,1,1) as batch shape.")
            return (1, 1, 1)
        return self._batch_shape

    @property
    def eff_batch_shape(self) -> Tuple[int, ...]:
        """
        Get the effective batch shape after applying repeat_batch_axis mask.

        Returns:
            Tuple[int, ...]: Effective batch dimensions, excluding zeros.
        """
        batch_shape = np.array(self.batch_shape) * self.repeat_batch_axis
        batch_shape = batch_shape[batch_shape != 0]
        return batch_shape

    def initialize_params(
        self,
        random_key: Optional[random.PRNGKey] = None,
        repeat: int = 1,
        initialization: Optional[str] = None,
        initialization_domain: Optional[List[float]] = None,
    ) -> random.PRNGKey:
        """
        Initialize the variational parameters of the model.

        Args:
            random_key (Optional[random.PRNGKey]): JAX random key for initialization.
                If None, uses the model's internal random key.
            repeat (int): Number of parameter sets to create (batch dimension).
                Defaults to 1.
            initialization (Optional[str]): Strategy for parameter initialization.
                Options: "random", "zeros", "pi", "zero-controlled", "pi-controlled".
                If None, uses the strategy specified in the constructor.
            initialization_domain (Optional[List[float]]): Domain [min, max] for
                random initialization. If None, uses the domain from constructor.

        Returns:
            random.PRNGKey: Updated random key after initialization.

        Raises:
            Exception: If an invalid initialization method is specified.
        """
        # Initializing params
        params_shape = (*self._params_shape, repeat)

        # use existing strategy if not specified
        initialization = initialization or self._inialization_strategy
        initialization_domain = initialization_domain or self._initialization_domain

        random_key, sub_key = safe_random_split(
            random_key if random_key is not None else self.random_key
        )

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
                sub_key,
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
                sub_key,
                params_shape,
                minval=initialization_domain[0],
                maxval=initialization_domain[1],
            )
            self.params = set_control_params(self.params, 0)
        elif initialization == "pi-controlled":
            self.params: jnp.ndarray = random.uniform(
                sub_key,
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

        return random_key

    def transform_input(
        self, inputs: jnp.ndarray, enc_params: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Transform input data by scaling with encoding parameters.

        Implements the input transformation as described in arXiv:2309.03279v2,
        where inputs are linearly scaled by encoding parameters before being
        used in the quantum circuit.

        Args:
            inputs (jnp.ndarray): Input data point of shape (n_input_feat,) or
                (batch_size, n_input_feat).
            enc_params (jnp.ndarray): Encoding weight scalar or vector used to
                scale the input.

        Returns:
            jnp.ndarray: Transformed input, element-wise product of inputs
                and enc_params.
        """
        return inputs * enc_params

    def _iec(
        self,
        inputs: jnp.ndarray,
        data_reupload: jnp.ndarray,
        enc: Encoding,
        enc_params: jnp.ndarray,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        random_key: Optional[random.PRNGKey] = None,
    ) -> None:
        """
        Apply Input Encoding Circuit (IEC) with angle encoding.

        Encodes classical input data into the quantum circuit using rotation
        gates (e.g., RX, RY, RZ). Supports data re-uploading at specified
        positions in the circuit.

        Args:
            inputs (jnp.ndarray): Input data of shape (n_input_feat,) or
                (batch_size, n_input_feat).
            data_reupload (jnp.ndarray): Boolean array of shape (n_qubits, n_input_feat)
                indicating where to apply encoding gates.
            enc (Encoding): Encoding strategy containing the encoding gate functions.
            enc_params (jnp.ndarray): Encoding parameters of shape
                (n_qubits, n_input_feat) used to scale inputs.
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise parameters for gate-level noise simulation. Defaults to None.
            random_key (Optional[random.PRNGKey]): JAX random key for stochastic
                noise. Defaults to None.

        Returns:
            None: Gates are applied in-place to the quantum circuit.
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
                    random_key, sub_key = safe_random_split(random_key)
                    enc[idx](
                        self.transform_input(inputs[..., idx], enc_params[q, idx]),
                        wires=q,
                        noise_params=noise_params,
                        random_key=sub_key,
                    )

    def _variational(
        self,
        params: jnp.ndarray,
        inputs: jnp.ndarray,
        pulse_params: Optional[jnp.ndarray] = None,
        random_key: Optional[random.PRNGKey] = None,
        enc_params: Optional[jnp.ndarray] = None,
        gate_mode: str = "unitary",
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    ) -> None:
        """
        Build the variational quantum circuit structure.

        Constructs the circuit by applying state preparation, alternating
        variational ansatz layers with input encoding layers, and optional
        noise channels.

        The first four parameters (after ``self``) — ``params``, ``inputs``,
        ``pulse_params``, ``random_key`` — are the batchable positional
        arguments that ``_mp_executor`` passes via ``QuantumScript.execute``.
        The remaining keyword arguments are broadcast across the batch.

        Args:
            params (jnp.ndarray): Variational parameters of shape
                (n_layers, n_params_per_layer).
            inputs (jnp.ndarray): Input data of shape (n_input_feat,).
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers of shape
                (n_layers, n_pulse_params_per_layer) for pulse-mode execution.
                Defaults to None (uses model's pulse_params).
            random_key (Optional[random.PRNGKey]): JAX random key for stochastic
                operations. Defaults to None.
            enc_params (Optional[jnp.ndarray]): Encoding parameters of shape
                (n_qubits, n_input_feat). Defaults to None (uses model's enc_params).
            gate_mode (str): Gate execution mode, either "unitary" or "pulse".
                Defaults to "unitary".
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise parameters for simulation. Defaults to None.

        Returns:
            None: Gates are applied in-place to the quantum circuit.

        Note:
            Issues RuntimeWarning if called directly without providing parameters
            that would normally be passed through the forward method.
        """
        # TODO: rework and double check params shape
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

        if noise_params is None:
            if self.noise_params is not None:
                warnings.warn(
                    "Explicit call to `_circuit` or `_variational` detected: "
                    "`noise_params` is None, using `self.noise_params` instead.",
                    RuntimeWarning,
                )
                noise_params = self.noise_params

        if noise_params is not None:
            if random_key is None:
                warnings.warn(
                    "Explicit call to `_circuit` or `_variational` detected: "
                    "`random_key` is None, using `random.PRNGKey(0)` instead.",
                    RuntimeWarning,
                )
                random_key = self.random_key
            self._apply_state_prep_noise(noise_params=noise_params)

        # state preparation
        for q in range(self.n_qubits):
            for _sp, sp_pulse_params in zip(self._sp, self.sp_pulse_params):
                random_key, sub_key = safe_random_split(random_key)
                _sp(
                    wires=q,
                    pulse_params=sp_pulse_params,
                    noise_params=noise_params,
                    random_key=sub_key,
                    gate_mode=gate_mode,
                )

        # circuit building
        for layer in range(0, self.n_layers):
            random_key, sub_key = safe_random_split(random_key)
            # ansatz layers
            self.pqc(
                params[layer],
                self.n_qubits,
                pulse_params=pulse_params[layer],
                noise_params=noise_params,
                random_key=sub_key,
                gate_mode=gate_mode,
            )

            random_key, sub_key = safe_random_split(random_key)
            # encoding layers
            self._iec(
                inputs,
                data_reupload=self.data_reupload[layer],
                enc=self._enc,
                enc_params=enc_params,
                noise_params=noise_params,
                random_key=sub_key,
            )

            # visual barrier (no-op in yaqsi, purely cosmetic in PennyLane)

        # final ansatz layer
        if self.has_dru:  # same check as in init
            random_key, sub_key = safe_random_split(random_key)
            self.pqc(
                params[self.n_layers],
                self.n_qubits,
                pulse_params=pulse_params[-1],
                noise_params=noise_params,
                random_key=sub_key,
                gate_mode=gate_mode,
            )

        # channel noise
        if noise_params is not None:
            self._apply_general_noise(noise_params=noise_params)

    def _build_obs(self) -> Tuple[str, List[op.Operation]]:
        """Build the yaqsi measurement type and observable list.

        Translates the model's ``execution_type`` and ``output_qubit``
        settings into parameters suitable for
        :meth:`~qml_essentials.yaqsi.QuantumScript.execute`.

        Returns:
            Tuple ``(meas_type, obs)`` where *meas_type* is one of
            ``"expval"``, ``"probs"``, ``"density"``, ``"state"`` and *obs*
            is a (possibly empty) list of :class:`Operation` observables.
        """
        if self.execution_type == "density":
            return "density", []

        if self.execution_type == "state":
            return "state", []

        if self.execution_type == "expval":
            obs: List[op.Operation] = []
            for qubit_spec in self.output_qubit:
                if isinstance(qubit_spec, int):
                    obs.append(op.PauliZ(wires=qubit_spec))
                else:
                    # parity: Z ⊗ Z ⊗ …
                    obs.append(ys.build_parity_observable(list(qubit_spec)))
            return "expval", obs

        if self.execution_type == "probs":
            # probs are computed on the full system; subsystem
            # marginalisation is handled in _postprocess_res
            return "probs", []

        raise ValueError(f"Invalid execution_type: {self.execution_type}.")

    def _apply_state_prep_noise(
        self, noise_params: Dict[str, Union[float, Dict[str, float]]]
    ) -> None:
        """
        Apply state preparation noise to all qubits.

        Simulates imperfect state preparation by applying BitFlip errors
        to each qubit with the specified probability.

        Args:
            noise_params (Dict[str, Union[float, Dict[str, float]]]): Dictionary
                containing noise parameters. Uses the "StatePreparation" key
                for the BitFlip probability.

        Returns:
            None: Noise channels are applied in-place to the circuit.
        """
        p = noise_params.get("StatePreparation", 0.0)
        if p > 0:
            for q in range(self.n_qubits):
                op.BitFlip(p, wires=q)

    def _apply_general_noise(
        self, noise_params: Dict[str, Union[float, Dict[str, float]]]
    ) -> None:
        """
        Apply general noise channels to all qubits.

        Applies various decoherence and error channels after the circuit
        execution, simulating environmental noise effects.

        Args:
            noise_params (Dict[str, Union[float, Dict[str, float]]]): Dictionary
                containing noise parameters with the following supported keys:
                - "AmplitudeDamping" (float): Probability for amplitude damping.
                - "PhaseDamping" (float): Probability for phase damping.
                - "Measurement" (float): Probability for measurement error (BitFlip).
                - "ThermalRelaxation" (Dict): Dictionary with keys "t1", "t2",
                  "t_factor" for thermal relaxation simulation.

        Returns:
            None: Noise channels are applied in-place to the circuit.

        Note:
            Gate-level noise (e.g., GateError) is handled separately in the
            Gates.Noise module and applied at the individual gate level.
        """
        amp_damp = noise_params.get("AmplitudeDamping", 0.0)
        phase_damp = noise_params.get("PhaseDamping", 0.0)
        thermal_relax = noise_params.get("ThermalRelaxation", 0.0)
        meas = noise_params.get("Measurement", 0.0)
        for q in range(self.n_qubits):
            if amp_damp > 0:
                op.AmplitudeDamping(amp_damp, wires=q)
            if phase_damp > 0:
                op.PhaseDamping(phase_damp, wires=q)
            if meas > 0:
                op.BitFlip(meas, wires=q)
            if isinstance(thermal_relax, dict):
                t1 = thermal_relax["t1"]
                t2 = thermal_relax["t2"]
                t_factor = thermal_relax["t_factor"]
                circuit_depth = self._get_circuit_depth()
                tg = circuit_depth * t_factor
                op.ThermalRelaxationError(1.0, t1, t2, tg, q)

    def _get_circuit_depth(self, inputs: Optional[jnp.ndarray] = None) -> int:
        """
        Calculate the depth of the quantum circuit.

        Records the circuit onto a tape (without noise) and computes the
        depth as the length of the critical path: each gate is scheduled
        at the earliest time step after all of its qubits are free.

        Args:
            inputs (Optional[jnp.ndarray]): Input data for circuit evaluation.
                If None, default zero inputs are used.

        Returns:
            int: The circuit depth (longest path of gates in the circuit).
        """
        # Return cached value if available
        if hasattr(self, "_cached_circuit_depth"):
            return self._cached_circuit_depth

        from qml_essentials.tape import recording
        from qml_essentials.operations import KrausChannel

        inputs = self._inputs_validation(inputs)

        # Temporarily clear noise_params to prevent _variational from
        # picking them up (which would call _apply_general_noise →
        # _get_circuit_depth again, causing infinite recursion).
        saved_noise = self._noise_params
        self._noise_params = None

        with recording() as tape:
            self._variational(
                self.params[:, :, 0] if self.params.ndim == 3 else self.params,
                inputs[0] if inputs.ndim == 2 else inputs,
                noise_params=None,
            )

        self._noise_params = saved_noise

        # Filter out noise channels — only count unitary gates
        ops = [o for o in tape if not isinstance(o, KrausChannel)]

        if not ops:
            self._cached_circuit_depth = 0
            return 0

        # Schedule each gate at the earliest time step where all its wires
        # are free.  ``wire_busy[q]`` tracks the next free time step for
        # qubit ``q``.
        wire_busy: Dict[int, int] = {}
        depth = 0
        for gate in ops:
            start = max((wire_busy.get(w, 0) for w in gate.wires), default=0)
            end = start + 1
            for w in gate.wires:
                wire_busy[w] = end
            depth = max(depth, end)

        self._cached_circuit_depth = depth
        return depth

    def draw(
        self,
        inputs: Optional[jnp.ndarray] = None,
        figure: str = "text",
        **kwargs: Any,
    ) -> Union[str, Any]:
        """Visualize the quantum circuit.

        Records the circuit tape (without noise) and renders the gate
        sequence using the requested backend.

        Args:
            inputs (Optional[jnp.ndarray]): Input data for the circuit.
                If ``None``, default zero inputs are used.
            figure (str): Rendering backend.  One of:

                * ``"text"``  — ASCII art (returned as a ``str``).
                * ``"mpl"``   — Matplotlib figure (returns ``(fig, ax)``).
                * ``"tikz"``  — LaTeX/TikZ ``quantikz`` code (returns a
                  :class:`~qml_essentials.utils.QuanTikz.TikzFigure`).

            **kwargs: Extra options forwarded to the drawing backend
                (e.g. ``gate_values=True``, ``inputs_symbols="x"``).

        Returns:
            Depends on *figure*:

            * ``"text"``  → ``str``
            * ``"mpl"``   → ``(matplotlib.figure.Figure, matplotlib.axes.Axes)``
            * ``"tikz"``  → :class:`QuanTikz.TikzFigure`

        Raises:
            ValueError: If *figure* is not one of the supported modes.
        """
        inputs = self._inputs_validation(inputs)
        params = self.params[:, :, 0] if self.params.ndim == 3 else self.params
        inp = inputs[0] if inputs.ndim == 2 else inputs

        # Record without noise to get a clean circuit
        saved_noise = self._noise_params
        self._noise_params = None

        draw_script = ys.QuantumScript(f=self._variational, n_qubits=self.n_qubits)
        result = draw_script.draw(
            figure=figure,
            args=(params, inp),
            kwargs={"noise_params": None},
            **kwargs,
        )

        self._noise_params = saved_noise
        return result

    def __repr__(self) -> str:
        """Return text representation of the quantum circuit model."""
        return self.draw(figure="text")

    def __str__(self) -> str:
        """Return string representation of the quantum circuit model."""
        return self.draw(figure="text")

    def _params_validation(self, params: Optional[jnp.ndarray]) -> jnp.ndarray:
        """
        Validate and normalize variational parameters.

        Ensures parameters have the correct shape with a batch dimension,
        and updates the model's internal parameters if new ones are provided.

        Args:
            params (Optional[jnp.ndarray]): Variational parameters to validate.
                If None, returns the model's current parameters.

        Returns:
            jnp.ndarray: Validated parameters with shape
                (n_layers, n_params_per_layer, batch_size).
        """
        # append batch axis if not provided
        if params is not None:
            if len(params.shape) == 2:
                params = np.expand_dims(params, axis=-1)

            self.params = params
        else:
            params = self.params

        return params

    def _pulse_params_validation(
        self, pulse_params: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Validate and normalize pulse parameters.

        Ensures pulse parameters are set, using model defaults if not provided.

        Args:
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers.
                If None, returns the model's current pulse parameters.

        Returns:
            jnp.ndarray: Validated pulse parameters with shape
                (n_layers, n_pulse_params_per_layer, batch_size).
        """
        if pulse_params is None:
            pulse_params = self.pulse_params
        else:
            self.pulse_params = pulse_params

        return pulse_params

    def _enc_params_validation(self, enc_params: Optional[jnp.ndarray]) -> jnp.ndarray:
        """
        Validate and normalize encoding parameters.

        Ensures encoding parameters have the correct shape for the model's
        input feature dimensions.

        Args:
            enc_params (Optional[jnp.ndarray]): Encoding parameters to validate.
                If None, returns the model's current encoding parameters.

        Returns:
            jnp.ndarray: Validated encoding parameters with shape
                (n_qubits, n_input_feat).

        Raises:
            ValueError: If enc_params shape is incompatible with n_input_feat > 1.
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
        Validate and normalize input data.

        Converts various input formats to a standardized 2D array shape
        suitable for batch processing in the quantum circuit.

        Args:
            inputs (Union[None, List, float, int, jnp.ndarray]): Input data in
                various formats:
                - None: Returns zeros with shape (1, n_input_feat)
                - float/int: Single scalar value
                - List: List of values or batched inputs
                - jnp.ndarray: NumPy/JAX array

        Returns:
            jnp.ndarray: Validated inputs with shape (batch_size, n_input_feat).

        Raises:
            ValueError: If input shape is incompatible with expected n_input_feat.

        Warns:
            UserWarning: If input is replicated to match n_input_feat.
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

    def _mp_executor(
        self,
        params: jnp.ndarray,
        pulse_params: jnp.ndarray,
        inputs: jnp.ndarray,
        enc_params: jnp.ndarray,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]],
        random_key: random.PRNGKey,
        gate_mode: str,
        meas_type: str,
        obs: List[op.Operation],
    ) -> jnp.ndarray:
        """
        Execute circuit function with optional parallelization over batches.

        Uses the yaqsi QuantumScript to execute the circuit.  When batching
        is needed (B > 1), ``QuantumScript.execute`` is called with
        ``in_axes`` so that ``jax.vmap`` is applied internally.

        Args:
            params (jnp.ndarray): Variational parameters of shape
                (n_layers, n_params_per_layer, batch_size).
            pulse_params (jnp.ndarray): Pulse parameters of shape
                (n_layers, n_pulse_params_per_layer, batch_size).
            inputs (jnp.ndarray): Input data of shape (batch_size, n_input_feat).
            enc_params (jnp.ndarray): Encoding parameters of shape
                (n_qubits, n_input_feat).
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise configuration dictionary.
            random_key (random.PRNGKey): JAX random key for stochastic operations.
            gate_mode (str): Gate execution mode ("unitary" or "pulse").
            meas_type (str): Measurement type for yaqsi execute.
            obs (List[op.Operation]): Observable list for expval measurements.

        Returns:
            jnp.ndarray: Circuit execution results, post-processed for uniformity.
        """
        B = np.prod(self.eff_batch_shape)

        # kwargs are broadcast (not vmapped over)
        exec_kwargs = dict(
            noise_params=noise_params,
            enc_params=enc_params,
            gate_mode=gate_mode,
        )

        if B > 1:
            random_keys = safe_random_split(random_key, num=B)

            in_axes = (
                2 if self.batch_shape[1] > 1 else None,  # params
                0 if self.batch_shape[0] > 1 else None,  # inputs
                2 if self.batch_shape[2] > 1 else None,  # pulse_params
                0,  # random_keys
            )

            result = self.script.execute(
                type=meas_type,
                obs=obs,
                args=(params, inputs, pulse_params, random_keys),
                kwargs=exec_kwargs,
                in_axes=in_axes,
            )
        else:
            result = self.script.execute(
                type=meas_type,
                obs=obs,
                args=(params, inputs, pulse_params, random_key),
                kwargs=exec_kwargs,
            )

        return self._postprocess_res(result)

    def _postprocess_res(self, result: Union[List, jnp.ndarray]) -> jnp.ndarray:
        """
        Post-process circuit execution results for uniform shape.

        Converts list outputs (from multiple measurements) to stacked arrays
        and reorders axes for consistent batch dimension placement.

        Args:
            result (Union[List, jnp.ndarray]): Raw circuit output, either a
                list of measurement results or a single array.

        Returns:
            jnp.ndarray: Uniformly shaped result array with batch dimension first.
        """
        if isinstance(result, list):
            # we use moveaxis here because in case of parity measure,
            # there is another dimension appended to the end and
            # simply transposing would result in a wrong shape
            result = jnp.stack(result)
            if len(result.shape) > 1:
                result = jnp.moveaxis(result, 0, 1)
        return result

    def _assimilate_batch(
        self,
        inputs: jnp.ndarray,
        params: jnp.ndarray,
        pulse_params: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Align batch dimensions across inputs, parameters, and pulse parameters.

        Broadcasts and reshapes arrays to have compatible batch dimensions
        for vectorized circuit execution. Sets the internal batch_shape.

        Args:
            inputs (jnp.ndarray): Input data of shape (B_I, n_input_feat).
            params (jnp.ndarray): Parameters of shape (n_layers, n_params, B_P).
            pulse_params (jnp.ndarray): Pulse params of shape (n_layers, n_pulse, B_R).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Tuple containing:
                - inputs: Reshaped to (B, n_input_feat) where B = B_I * B_P * B_R
                - params: Reshaped to (n_layers, n_params, B)
                - pulse_params: Reshaped to (n_layers, n_pulse, B)

        Note:
            The effective batch shape depends on repeat_batch_axis configuration.
            This is the only method that sets self._batch_shape.
        """
        B_I = inputs.shape[0]
        # we check for the product because there is a chance that
        # there are no params. In this case we want B_P to be 1
        B_P = 1 if 0 in params.shape else params.shape[-1]
        B_R = pulse_params.shape[-1]

        # THIS is the only place where we set the batch shape
        self._batch_shape = (B_I, B_P, B_R)
        B = np.prod(self.eff_batch_shape)

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

        return inputs, params, pulse_params

    def _requires_density(self) -> bool:
        """
        Check if density matrix simulation is required.

        Determines whether the circuit must be executed with the mixed-state
        simulator based on execution type and noise configuration.

        Returns:
            bool: True if density matrix simulation is required, False otherwise.
                Returns True if:
                - execution_type is "density", or
                - Any non-coherent noise channel has non-zero probability
        """
        if self.execution_type == "density":
            return True

        if self.noise_params is None:
            return False

        coherent_noise = {"GateError"}
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
        Execute the quantum circuit (callable interface).

        Provides a convenient callable interface for circuit execution,
        delegating to the _forward method.

        Args:
            params (Optional[jnp.ndarray]): Variational parameters of shape
                (n_layers, n_params_per_layer) or (n_layers, n_params_per_layer, batch).
                If None, uses model's internal parameters.
            inputs (Optional[jnp.ndarray]): Input data of shape
                (batch_size, n_input_feat). If None, uses zero inputs.
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers for
                pulse-mode gate execution.
            enc_params (Optional[jnp.ndarray]): Encoding parameters of shape
                (n_qubits, n_input_feat). If None, uses model's encoding parameters.
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise configuration. If None, uses previously set noise parameters.
            execution_type (Optional[str]): Measurement type: "expval", "density",
                "probs", or "state". If None, uses current execution_type setting.
            force_mean (bool): If True, averages results over measurement qubits.
                Defaults to False.
            gate_mode (str): Gate execution backend, "unitary" or "pulse".
                Defaults to "unitary".

        Returns:
            jnp.ndarray: Circuit output with shape depending on execution_type:
                - "expval": (n_output_qubits,) or scalar
                - "density": (2^n_output, 2^n_output)
                - "probs": (2^n_output,) or (n_pairs, 2^pair_size)
                - "state": (2^n_qubits,)
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
        Execute the quantum circuit forward pass.

        Internal implementation of the forward pass that handles parameter
        validation, batch alignment, and circuit execution routing.

        Args:
            params (Optional[jnp.ndarray]): Variational parameters of shape
                (n_layers, n_params_per_layer) or
                (n_layers, n_params_per_layer, batch).
                If None, uses model's internal parameters.
            inputs (Optional[jnp.ndarray]): Input data of shape
                (batch_size, n_input_feat).
                If None, uses zero inputs.
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers for
                pulse-mode gate execution.
            enc_params (Optional[jnp.ndarray]): Encoding parameters of shape
                (n_qubits, n_input_feat). If None, uses model's encoding parameters.
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise configuration. If None, uses previously set noise parameters.
            execution_type (Optional[str]): Measurement type: "expval", "density",
                "probs", or "state". If None, uses current execution_type setting.
            force_mean (bool): If True, averages results over measurement qubits.
                Defaults to False.
            gate_mode (str): Gate execution backend, "unitary" or "pulse".
                Defaults to "unitary".

        Returns:
            jnp.ndarray: Circuit output with shape depending on execution_type:
                - "expval": (n_output_qubits,) or scalar
                - "density": (2^n_output, 2^n_output)
                - "probs": (2^n_output,) or (n_pairs, 2^pair_size)
                - "state": (2^n_qubits,)

        Raises:
            ValueError: If pulse_params provided without pulse gate_mode, or
                if noise_params provided with pulse gate_mode.
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

        inputs, params, pulse_params = self._assimilate_batch(
            inputs,
            params,
            pulse_params,
        )

        self.random_key, subkey = safe_random_split(self.random_key)

        # Build measurement type & observables from execution_type / output_qubit
        meas_type, obs = self._build_obs()

        # Yaqsi auto-routes between statevector and density-matrix simulation
        # based on whether noise channels appear on the tape, so a single
        # _mp_executor call handles both branches.
        result = self._mp_executor(
            params=params,
            pulse_params=pulse_params,
            inputs=inputs,
            enc_params=enc_params,
            noise_params=self.noise_params,
            random_key=subkey,
            gate_mode=gate_mode,
            meas_type=meas_type,
            obs=obs,
        )

        # --- Post-processing for partial-qubit measurements ---------------
        if self.execution_type == "density" and not self.all_qubit_measurement:
            result = ys.partial_trace(result, self.n_qubits, self.output_qubit)

        if self.execution_type == "probs" and not self.all_qubit_measurement:
            if isinstance(self.output_qubit[0], (list, tuple)):
                # list of qubit groups — marginalize each independently
                result = jnp.stack(
                    [
                        ys.marginalize_probs(result, self.n_qubits, list(group))
                        for group in self.output_qubit
                    ]
                )
            else:
                result = ys.marginalize_probs(result, self.n_qubits, self.output_qubit)

        result = jnp.asarray(result)
        result = result.reshape((*self.eff_batch_shape, *self._result_shape)).squeeze()

        if (
            self.execution_type in ("expval", "probs")
            and force_mean
            and len(result.shape) > 0
            and self._result_shape[0] > 1
        ):
            result = result.mean(axis=-1)

        return result
