from typing import Any, Dict, Optional, Tuple, Callable, Union, List

import warnings
import jax.numpy as jnp
import numpy as np
import jax
from jax import random

from qml_essentials import jaqsi as js
from qml_essentials import operations as op
from qml_essentials.tape import recording
from qml_essentials.operations import KrausChannel
from qml_essentials.ansaetze import Ansaetze, Circuit, Encoding
from qml_essentials.gates import Gates, PulseInformation as pinfo
from qml_essentials.script import _make_hashable
from qml_essentials.utils import safe_random_split

import logging

log = logging.getLogger(__name__)

GATE_MODES = {
    "unitary": ("unitary", "unitary"),
    "ansatz_pulse": ("pulse", "unitary"),
    "enc_pulse": ("unitary", "pulse"),
    "all_pulse": ("pulse", "pulse"),
}

# the modes that run the respective gate group at pulse level. Both only serve
# the deprecated explicit gate_mode argument and can go with it.
_ANSATZ_PULSE_MODES = ("ansatz_pulse", "all_pulse")
_ENC_PULSE_MODES = ("enc_pulse", "all_pulse")


class Model:
    """
    A quantum circuit model.
    """

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
        output_qubit: Union[List[int], int, None] = None,
        observables: Union[
            int, List[Union[int, List[int]]], List[op.Operation], None
        ] = None,
        shots: Optional[int] = None,
        random_seed: int = 1000,
        repeat_batch_axis: List[bool] = [True, True, True, True],
        pulse_shape: str = "gaussian",
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
            output_qubit (List[int], int, optional): Deprecated alias for
                ``observables``. Forwards to ``observables`` and will be removed
                in a future release. Defaults to None.
            observables (int, List[int], List[List[int]], List[op.Operation],
                optional): Measurement specification. A qubit index, a list of
                indices, or a list of qubit groups (for $Z$-parity) selects the
                measured subsystem with the default PauliZ readout.
                Alternatively, a list of
                :class:`~qml_essentials.operations.Operation` observables makes
                ``execution_type="expval"`` return one expectation value per
                observable. When None all qubits are measured. Defaults to None.
            shots (Optional[int], optional): The number of shots to use for
                the quantum device. Defaults to None.
            random_seed (int, optional): seed for the random number generator
                in initialization is "random" and for random noise parameters.
                Defaults to 1000.
            repeat_batch_axis (List[bool], optional): Each boolean in the array
                determines over which axes to parallelise computation. The axes
                correspond to [inputs, params, pulse_params, enc_pulse_params].
                Defaults to [True, True, True, True], meaning that batching is
                enabled over all axes. A 3-element list (legacy) is accepted and
                extended with a trailing True for the enc_pulse_params axis.
            pulse_shape (str, optional): Pulse envelope shape for pulse-level
                simulation. One of ``PulseEnvelope.available()``.
                Defaults to ``"gaussian"``.

        Returns:
            None
        """
        # Initialize default parameters needed for circuit evaluation
        self.n_qubits: int = n_qubits
        if output_qubit is not None:
            if observables is not None:
                raise ValueError("Pass either output_qubit or observables, not both.")
            warnings.warn(
                "output_qubit is deprecated, use observables instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            observables = output_qubit
        self.observables = observables
        self.n_layers: int = n_layers
        self.noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None
        self.shots = shots
        self.trainable_frequencies: bool = trainable_frequencies
        self.execution_type: str = "expval"
        # backward compatibility
        # TODO: consider making this more generic in future
        # (for someone wanting to control this without bothering with pulse stuff)
        if len(repeat_batch_axis) == 3:
            log.warning("Batch axis should have length 4")
            repeat_batch_axis = list(repeat_batch_axis) + [True]
        self.repeat_batch_axis: List[bool] = repeat_batch_axis

        # --- Pulse envelope ---
        pinfo.set_envelope(pulse_shape)

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

        if self._enc.is_golomb:
            self._enc._n_qubits = n_qubits

        # Number of possible inputs
        self.n_input_feat = len(self._enc)
        log.debug(f"Number of input features: {self.n_input_feat}")

        # Trainable frequencies, default initialization as in arXiv:2309.03279v2
        self.enc_params = jnp.ones((self.n_layers, self.n_qubits, self.n_input_feat))

        # Per-feature pulse-parameter sizes/offsets used to slice
        # enc_pulse_params in _iec under "all_pulse" mode. Only encodings whose
        # gates all have a pulse parametrization are supported (golomb and
        # custom callables do not).
        # TODO: golomb should be doable but needs a closer investigation
        self._enc_pulse_sizes: List[int] = []
        self._enc_pulse_capable = not self._enc.is_golomb
        if self._enc_pulse_capable:
            for g in self._enc._gates:
                if pinfo.gate_by_name(g) is None:
                    self._enc_pulse_capable = False
                    self._enc_pulse_sizes = []
                    break
                self._enc_pulse_sizes.append(pinfo.gate_by_name(g).size)

        self._enc_pulse_offsets: List[int] = list(
            np.cumsum([0, *self._enc_pulse_sizes[:-1]])
        )
        self._enc_pulse_shape: Tuple[int, int, int] = (
            self.n_layers,
            self.n_qubits,
            sum(self._enc_pulse_sizes),
        )

        # --- Data-Reuploading ---

        # Keep as NumPy array (not JAX) so that ``if data_reupload[q, idx]``
        # in _iec remains a concrete Python bool even under jax.jit tracing.
        # note that setting this will also update self.degree and self.frequencies
        # and in consequence also self.has_dru
        self.data_reupload = data_reupload

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
        self.pulse_params: jnp.ndarray = jnp.ones((1, *self._pulse_params_shape))

        log.info(f"Initialized pulse parameters with shape {self.pulse_params.shape}.")

        # Initializing encoding pulse params (element-wise scalers, ones by
        # default). Batch-first convention, mirroring pulse_params.
        self.enc_pulse_params: jnp.ndarray = jnp.ones((1, *self._enc_pulse_shape))

        log.info(
            f"Initialized encoding pulse parameters with shape "
            f"{self.enc_pulse_params.shape}."
        )

        # Initialise the jaqsi Script that wraps _variational.
        # No device selection needed - jaqsi auto-routes between statevector
        # and density-matrix simulation based on whether noise channels are
        # present on the tape.
        self.script = js.Script(f=self._variational, n_qubits=self.n_qubits)

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
        """Deprecated alias for :attr:`observables`; returns the measured wires."""
        warnings.warn(
            "output_qubit is deprecated, use observables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._measured_wires

    @output_qubit.setter
    def output_qubit(self, value: Union[int, List[int]]) -> None:
        warnings.warn(
            "output_qubit is deprecated, use observables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.observables = value

    @property
    def observables(self) -> List:
        """The custom :class:`~qml_essentials.operations.Operation` observables,
        or the list of measured wires when using the default PauliZ readout.

        With a list of observables, ``__call__`` and ``execution_type="expval"``
        returns one expectation value per observable instead of one ``PauliZ``
        per measured qubit.
        """
        return (
            self._observables if self._observables is not None else self._measured_wires
        )

    @observables.setter
    def observables(self, value: Union[int, List, None]) -> None:
        if value is None:
            self._observables = None
            self._measured_wires = list(range(self.n_qubits))
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(o, op.Operation) for o in value)
        ):
            self._observables = list(value)
            self._measured_wires = list(range(self.n_qubits))
        elif isinstance(value, list) and any(
            isinstance(o, op.Operation) for o in value
        ):
            raise ValueError(
                "observables list must contain either qubit indices or "
                "Operation objects, not a mix."
            )
        else:
            # qubit specification: normalize into the measured wire list
            self._observables = None
            if isinstance(value, list):
                assert len(value) <= self.n_qubits, (
                    f"Size of observables {len(value)} cannot be larger than "
                    f"number of qubits {self.n_qubits}."
                )
                self._measured_wires = value
            elif isinstance(value, int):
                if value == -1:
                    self._measured_wires = list(range(self.n_qubits))
                else:
                    assert value < self.n_qubits, (
                        f"Output qubit {value} cannot be larger than {self.n_qubits}."
                    )
                    self._measured_wires = [value]
            else:
                self._measured_wires = value

        # recompute the result shape for the (possibly new) observable count
        if hasattr(self, "_execution_type"):
            self.execution_type = self.execution_type

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
                2 ** len(self._measured_wires),
                2 ** len(self._measured_wires),
            )
        elif value == "expval":
            # custom observables (if provided) fix the number of expectation
            # values; otherwise one PauliZ (or Z-parity) per measured qubit.
            if getattr(self, "_observables", None) is not None:
                self._result_shape = (len(self._observables),)
            else:
                self._result_shape = (len(self._measured_wires),)
        elif value == "probs":
            # in case this is a list of parities,
            # each pair has 2^len(qubits) probabilities
            n_parity = (
                (2,) * len(self._measured_wires)
                if isinstance(self._measured_wires, (Tuple, List))
                else (2,)
            )
            self._result_shape = n_parity
        elif value == "state":
            self._result_shape = (2 ** len(self._measured_wires),)
        else:
            raise ValueError(f"Invalid execution type: {value}.")

        if value == "state" and not self.all_qubit_measurement:
            warnings.warn(
                f"{value} measurement ignores the measured subsystem, which is "
                f"{self._measured_wires}.",
                UserWarning,
            )

        if value != "expval" and getattr(self, "_observables", None) is not None:
            warnings.warn(
                f"Custom observables are ignored for execution_type={value!r}.",
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
            value = value.reshape(1, *value.shape)

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
    def enc_pulse_params(self) -> jnp.ndarray:
        """Get the encoding pulse parameters for all_pulse-mode execution."""
        return self._enc_pulse_params

    @enc_pulse_params.setter
    def enc_pulse_params(self, value: jnp.ndarray) -> None:
        """Set the encoding pulse parameters."""
        self._enc_pulse_params = value

    @property
    def data_reupload(self) -> jnp.ndarray:
        """Get the data reupload mask."""
        return self._data_reupload

    @data_reupload.setter
    def data_reupload(self, value: jnp.ndarray) -> None:
        """Set the data reupload mask.

        Always converts to a concrete NumPy boolean array so that
        ``if data_reupload[q, idx]`` in :meth:`_iec` remains a plain
        Python ``bool`` even inside JAX-traced functions (jit / grad / vmap).
        """
        # Process data reuploading strategy and set degree
        if not isinstance(value, bool):
            if not isinstance(value, np.ndarray):
                value = np.array(value)

            if len(value.shape) == 2:
                assert value.shape == (
                    self.n_layers,
                    self.n_qubits,
                ), (
                    f"Data reuploading array has wrong shape. \
                    Expected {(self.n_layers, self.n_qubits)} or\
                    {(self.n_layers, self.n_qubits, self.n_input_feat)},\
                    got {value.shape}."
                )
                value = value.reshape(*value.shape, 1)
                value = np.repeat(value, self.n_input_feat, axis=2)

            assert value.shape == (
                self.n_layers,
                self.n_qubits,
                self.n_input_feat,
            ), (
                f"Data reuploading array has wrong shape. \
                Expected {(self.n_layers, self.n_qubits, self.n_input_feat)},\
                got {value.shape}."
            )

            log.debug(f"Data reuploading array:\n{value}")
        else:
            if value:
                value = np.ones((self.n_layers, self.n_qubits, self.n_input_feat))
                log.debug("Full data reuploading.")
            else:
                value = np.zeros((self.n_layers, self.n_qubits, self.n_input_feat))
                value[0][0] = 1
                log.debug("No data reuploading.")

        # convert to boolean values
        self._data_reupload = np.asarray(value).astype(bool)

        self.degree: Tuple = tuple(
            self._enc.get_n_freqs(self.data_reupload[..., i])
            for i in range(self.n_input_feat)
        )

        self.frequencies: Tuple = tuple(
            self._enc.get_spectrum(self.data_reupload[..., i])
            for i in range(self.n_input_feat)
        )

        # Cache has_dru as a plain Python bool so that it can be used in
        # Python ``if`` statements even inside JAX-traced functions.
        self._has_dru: bool = bool(max(int(np.max(f)) for f in self._frequencies) > 1)

    @property
    def degree(self) -> Tuple:
        """Get the degree of the model."""
        return self._degree

    @degree.setter
    def degree(self, value: Tuple):
        self._degree = value

    @property
    def frequencies(self) -> Tuple:
        """Get the frequencies of the model."""
        return self._frequencies

    @frequencies.setter
    def frequencies(self, value: Tuple):
        self._frequencies = value

    def exact_spectrum(self, method: str = "tree") -> Tuple[np.ndarray, ...]:
        """Compute the exact per-feature Fourier spectrum via the FourierTree.

        Unlike :attr:`frequencies` -- a naive per-feature estimate derived purely
        from the encoding, which can *overestimate* the spectrum (some
        coefficients are constrained to zero for all parameters) -- this builds
        the analytical Fourier tree (Nemkov et al.) and returns, for each input
        feature, the integer frequencies whose Fourier coefficient is not
        identically zero.  The result is always a subset of :attr:`frequencies`.

        The support is derived purely symbolically (no parameter sampling): see
        :meth:`~qml_essentials.coefficients.FourierTree.get_exact_support`.
        With ``method="tree"`` (default), frequencies whose contributions cancel
        identically across tree paths (e.g. two consecutive encodings combining
        into a single rotation) are excluded exactly; this enumerates the
        explicit tree, which can be infeasible for deep entangling circuits.
        With ``method="dp"``, a merged-state dynamic program derives the support
        without enumerating paths, which scales to deep circuits at the cost of
        not detecting identical cross-path cancellations.

        Requires a Clifford + Pauli-rotation ansatz (see
        :class:`~qml_essentials.pauli.PauliCircuit`); other gate sets raise
        ``NotImplementedError`` during tree construction.

        Args:
            method (str): ``"tree"`` (fully exact) or ``"dp"`` (scalable).

        Returns:
            Tuple[np.ndarray, ...]: One sorted integer frequency array per input
            feature (same layout as :attr:`frequencies`).
        """
        from qml_essentials.coefficients import FourierTree  # avoid circular imp.

        tree = FourierTree(self)

        # Position of each model feature within the tree's frequency vectors.
        feature_pos = {feat: i for i, feat in enumerate(tree.features)}

        # Union of the symbolic supports over all observables (roots).
        support = set()
        for freqs in tree.get_exact_support(method=method):
            farr = np.asarray(freqs)
            for k in range(farr.shape[0]):
                key = (
                    (int(farr[k]),)
                    if farr.ndim == 1
                    else tuple(int(v) for v in farr[k])
                )
                support.add(key)

        spectrum = []
        for feat in range(self.n_input_feat):
            if support and feat in feature_pos:
                pos = feature_pos[feat]
                vals = sorted({k[pos] for k in support})
            else:
                vals = [0]
            spectrum.append(np.array(vals, dtype=int))
        return tuple(spectrum)

    @property
    def has_dru(self) -> bool:
        """Check if the model has data reupload."""
        return self._has_dru

    @property
    def all_qubit_measurement(self) -> bool:
        """Check if measurement is performed on all qubits."""
        return self._measured_wires == list(range(self.n_qubits))

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """
        Get the batch shape (B_I, B_P, B_R, B_E).
        If the model was not called before,
        it returns (1, 1, 1, 1).

        Returns:
            Tuple[int, ...]: Tuple of (input_batch, param_batch, pulse_batch,
                enc_pulse_batch). Returns (1, 1, 1, 1) if model has not been
                called yet.
        """
        if self._batch_shape is None:
            log.debug("Model was not called yet. Returning (1,1,1,1) as batch shape.")
            return (1, 1, 1, 1)
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
        params_shape = (repeat, *self._params_shape)

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
                np_params[:, :, indices[0] : indices[1] : indices[2]] = (
                    np.ones_like(params[:, :, indices[0] : indices[1] : indices[2]])
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

    def next_key(self) -> random.PRNGKey:
        """
        Advance the internal random key and return a fresh sub key.

        Intended for stochastic execution inside a JAX transform: a jitted
        call is traced once and replays the key that was current at trace
        time, so fresh randomness has to enter as an argument. Call this
        outside the transform and pass the result as ``random_key``. Since the
        key is an argument rather than a constant, this does not trigger
        recompilation.

        Returns:
            random.PRNGKey: Fresh sub key, split off the internal key.
        """
        self.random_key, sub_key = safe_random_split(self.random_key)
        return sub_key

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
        enc_pulse_params: Optional[jnp.ndarray] = None,
        gate_mode: str = "unitary",
    ) -> None:
        """
        Apply Input Encoding Circuit (IEC) with angle encoding.

        Encodes classical input data into the quantum circuit using rotation
        gates (e.g., RX, RY, RZ). Supports data re-uploading at specified
        positions in the circuit.

        For Golomb encoding, a single multi-qubit diagonal unitary is applied
        to all qubits simultaneously instead of per-qubit rotation gates.

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
            enc_pulse_params (Optional[jnp.ndarray]): Encoding pulse-parameter
                scalers of shape (n_qubits, n_enc_pulse_per_qubit) for the
                current layer. Used when the encoding gates run at pulse level,
                i.e. the model-level mode is "enc_pulse" or "all_pulse".
                Defaults to None.
            gate_mode (str): Resolved per-gate encoding backend, "unitary"
                (ideal) or "pulse". This is the backend selected for the
                encoding group, distinct from the model-level modes
                (unitary, ansatz_pulse, enc_pulse, all_pulse). Defaults to
                "unitary".

        Returns:
            None: Gates are applied in-place to the quantum circuit.
        """
        # --- Golomb encoding: single multi-qubit gate on all qubits --------
        if enc.is_golomb:
            idx = 0  # Golomb encoding supports a single input feature
            # Check if any qubit has re-uploading enabled for this layer
            if data_reupload[:, idx].any():
                random_key, sub_key = safe_random_split(random_key)
                # Use the mean of enc_params across qubits as scalar scaling
                # (Golomb acts on all qubits jointly)
                mean_enc_param = jnp.mean(enc_params[:, idx])
                all_wires = list(range(self.n_qubits))
                enc[idx](
                    self.transform_input(inputs[..., idx], mean_enc_param),
                    wires=all_wires,
                    noise_params=noise_params,
                    random_key=sub_key,
                )
            return

        # --- Standard per-qubit encoding -----------------------------------
        for q in range(self.n_qubits):
            # use the last dimension of the inputs (feature dimension)
            for idx in range(inputs.shape[-1]):
                if data_reupload[q, idx]:
                    random_key, sub_key = safe_random_split(random_key)
                    # TODO: consider merging this with the pulses.py manager
                    pulse_kwargs = {}
                    if gate_mode == "pulse":
                        # scale the calibrated pulse params by this gate's
                        # scalers, as the pulse manager does for the ansatz
                        off = self._enc_pulse_offsets[idx]
                        size = self._enc_pulse_sizes[idx]
                        base = pinfo.gate_by_name(enc._gates[idx]).params
                        pulse_kwargs = dict(
                            pulse_params=base * enc_pulse_params[q, off : off + size],
                            gate_mode="pulse",
                        )

                    # use elipsis to index only the last dimension
                    # as inputs are generally *not* qubit dependent
                    enc[idx](
                        self.transform_input(inputs[..., idx], enc_params[q, idx]),
                        wires=q,
                        noise_params=noise_params,
                        random_key=sub_key,
                        **pulse_kwargs,
                    )

    @staticmethod
    def _debatch(value: jnp.ndarray, ndim: int) -> jnp.ndarray:
        """
        Drop a leading singleton batch axis (batch-first convention).

        Args:
            value (jnp.ndarray): Array to de-batch.
            ndim (int): Rank of a single (un-batched) element.

        Returns:
            jnp.ndarray: The array without its leading axis if that axis is a
                singleton batch dimension, otherwise the array unchanged.
        """
        if len(value.shape) > ndim and value.shape[0] == 1:
            return value[0]
        return value

    def _self_fallback(self, value: Any, name: str, warn: bool) -> Any:
        """
        Fall back to the model's own attribute when a parameter is not given.

        Args:
            value (Any): The provided value, or None.
            name (str): Name of the attribute to fall back to.
            warn (bool): Whether to warn when the fallback is used.

        Returns:
            Any: The provided value, or ``self.<name>`` if value is None.
        """
        if value is not None:
            return value
        if warn:
            warnings.warn(
                "Explicit call to `_circuit` or `_variational` detected: "
                f"`{name}` is None, using `self.{name}` instead.",
                RuntimeWarning,
            )
        return getattr(self, name)

    def _variational(
        self,
        params: jnp.ndarray,
        inputs: jnp.ndarray,
        pulse_params: Optional[jnp.ndarray] = None,
        random_key: Optional[random.PRNGKey] = None,
        enc_params: Optional[jnp.ndarray] = None,
        enc_pulse_params: Optional[jnp.ndarray] = None,
        gate_mode: str = "unitary",
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    ) -> None:
        """
        Build the variational quantum circuit structure.

        Constructs the circuit by applying state preparation, alternating
        variational ansatz layers with input encoding layers, and optional
        noise channels.

        The first six parameters (after ``self``) - ``params``, ``inputs``,
        ``pulse_params``, ``random_key``, ``enc_params``, ``enc_pulse_params`` -
        are the batchable positional arguments.
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
            enc_pulse_params (Optional[jnp.ndarray]): Encoding pulse-parameter
                scalers of shape (n_layers, n_qubits, n_enc_pulse_per_qubit) for
                "all_pulse" execution. Defaults to None (uses model's
                enc_pulse_params).
            gate_mode (str): Gate execution mode, one of "unitary",
                "ansatz_pulse", "enc_pulse" or "all_pulse". "ansatz_pulse" runs
                the ansatz and state preparation as pulses (encoding stays
                unitary); "enc_pulse" runs only the encoding gates as pulses;
                "all_pulse" runs both as pulses. Defaults to "unitary".
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise parameters for simulation. Defaults to None.

        Returns:
            None: Gates are applied in-place to the quantum circuit.

        Note:
            Issues RuntimeWarning if called directly without providing parameters
            that would normally be passed through the forward method.
        """
        # which backend the ansatz / state-prep gates and the encoding gates use
        sub_mode, enc_gate_mode = GATE_MODES[gate_mode]

        # TODO: rework and double check params shape
        params = self._debatch(params, 2)
        inputs = self._debatch(inputs, 1)

        # TODO: Raise warning if trainable frequencies is True, or similar. I.e., no
        #   warning if user does not care for frequencies or enc_params
        enc_params = self._self_fallback(
            enc_params, "enc_params", self.trainable_frequencies
        )

        pulse_params = self._self_fallback(
            pulse_params, "pulse_params", sub_mode == "pulse"
        )
        pulse_params = self._debatch(pulse_params, 2)

        enc_pulse_params = self._self_fallback(
            enc_pulse_params, "enc_pulse_params", enc_gate_mode == "pulse"
        )
        enc_pulse_params = self._debatch(enc_pulse_params, 3)

        noise_params = self._self_fallback(
            noise_params, "noise_params", self.noise_params is not None
        )

        if noise_params is not None:
            random_key = self._self_fallback(random_key, "random_key", True)
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
                    gate_mode=sub_mode,
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
                gate_mode=sub_mode,
            )

            random_key, sub_key = safe_random_split(random_key)
            # encoding layers
            self._iec(
                inputs,
                data_reupload=self.data_reupload[layer],
                enc=self._enc,
                enc_params=enc_params[layer],
                noise_params=noise_params,
                random_key=sub_key,
                enc_pulse_params=enc_pulse_params[layer],
                gate_mode=enc_gate_mode,
            )

        # final ansatz layer
        if self.has_dru:  # same check as in init
            random_key, sub_key = safe_random_split(random_key)
            self.pqc(
                params[self.n_layers],
                self.n_qubits,
                pulse_params=pulse_params[-1],
                noise_params=noise_params,
                random_key=sub_key,
                gate_mode=sub_mode,
            )

        # channel noise
        if noise_params is not None:
            self._apply_general_noise(noise_params=noise_params)

    def _build_obs(self) -> Tuple[str, List[op.Operation]]:
        """Build the jaqsi measurement type and observable list.

        Translates the model's ``execution_type`` and ``observables``
        settings into parameters suitable for
        :meth:`~qml_essentials.jaqsi.Script.execute`.

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
            if self._observables is not None:
                return "expval", list(self._observables)
            obs: List[op.Operation] = []
            for qubit_spec in self._measured_wires:
                if isinstance(qubit_spec, int):
                    obs.append(op.PauliZ(wires=qubit_spec))
                else:
                    # parity: Z \\otimes Z \\otimes …
                    obs.append(js.build_parity_observable(list(qubit_spec)))
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

        inputs = self._inputs_validation(inputs)

        # Temporarily clear noise_params to prevent _variational from
        # picking them up (which would call _apply_general_noise ->
        # _get_circuit_depth again, causing infinite recursion).
        saved_noise = self._noise_params
        self._noise_params = None

        with recording() as tape:
            self._variational(
                self.params[0] if self.params.ndim == 3 else self.params,
                inputs[0] if inputs.ndim == 2 else inputs,
                noise_params=None,
            )

        self._noise_params = saved_noise

        # Filter out noise channels - only count unitary gates
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

                * ``"text"``  - ASCII art (returned as a ``str``).
                * ``"mpl"``   - Matplotlib figure (returns ``(fig, ax)``).
                * ``"tikz"``  - LaTeX/TikZ ``quantikz`` code (returns a
                  :class:`TikzFigure`).
                * ``"pulse"`` - Pulse schedule (returns ``(fig, axes)``).
                  Only meaningful for pulse-mode models.

            **kwargs: Extra options forwarded to the drawing backend
                (e.g. ``gate_values=True``).

        Returns:
            Depends on figure:

            * ``"text"``  -> ``str``
            * ``"mpl"``   -> ``(matplotlib.figure.Figure, matplotlib.axes.Axes)``
            * ``"tikz"``  -> :class:`TikzFigure`

        Raises:
            ValueError: If figure is not one of the supported modes.
        """
        inputs = self._inputs_validation(inputs)
        params = self.params[0] if self.params.ndim == 3 else self.params
        inp = inputs[0] if inputs.ndim == 2 else inputs

        if figure == "pulse":
            return self.draw_pulse(inputs=inputs, **kwargs)

        # Record without noise to get a clean circuit
        saved_noise = self._noise_params
        self._noise_params = None

        draw_script = js.Script(f=self._variational, n_qubits=self.n_qubits)
        result = draw_script.draw(
            figure=figure,
            args=(params, inp),
            kwargs={"noise_params": None},
            **kwargs,
        )

        self._noise_params = saved_noise
        return result

    def draw_pulse(
        self,
        inputs: Optional[jnp.ndarray] = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize the pulse schedule for the circuit.

        Records the circuit in pulse mode and collects PulseEvents
        automatically via the pulse-event tape, then renders them.

        State preparation, ansatz and encoding gates are all rendered as
        pulses. Encodings without a pulse parametrization (golomb and custom
        callables) are omitted from the schedule.

        Args:
            inputs: Input data.  If ``None``, default zero inputs are used.
            **kwargs: Forwarded to
                :func:`~qml_essentials.drawing.draw_pulse_schedule`
                (e.g. ``show_carrier=True``, ``n_samples=300``).

        Returns:
            ``(fig, axes)`` — Matplotlib Figure and array of Axes.
        """
        if "gate_mode" in kwargs:
            warnings.warn(
                "draw_pulse no longer takes gate_mode, every gate group with a "
                "pulse representation is drawn.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.pop("gate_mode")

        inputs = self._inputs_validation(inputs)
        params = self.params[0] if self.params.ndim == 3 else self.params
        inp = inputs[0] if inputs.ndim == 2 else inputs

        # pass the model's own pulse parameters, so that _variational does not
        # fall back to them with a warning. Both are batch-first, so drawing
        # picks the first set, same as params above
        record_kwargs: Dict[str, Any] = {
            "gate_mode": "all_pulse" if self._enc_pulse_capable else "ansatz_pulse",
            "noise_params": None,
            "pulse_params": self.pulse_params[0],
        }
        if self._enc_pulse_capable:
            record_kwargs["enc_pulse_params"] = self.enc_pulse_params[0]

        draw_script = js.Script(f=self._variational, n_qubits=self.n_qubits)
        return draw_script.draw(
            figure="pulse",
            args=(params, inp),
            kwargs=record_kwargs,
            **kwargs,
        )

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
                (batch_size, n_layers, n_params_per_layer).
        """
        # append batch axis if not provided
        if params is not None:
            if len(params.shape) == 2:
                # jnp (not np) so params stays a JAX array under autodiff /
                # jit; mirrors the pulse_params handling below.
                params = jnp.expand_dims(params, axis=0)

            # Avoid stashing JAX tracers on ``self``: under an outer
            # transform (e.g. ``jit``/``jacrev``) the tracer becomes invalid
            # once the transform returns, and a subsequent read of
            # ``self.params`` would feed a leaked tracer into the next
            # call (raising ``UnexpectedTracerError``).
            if not isinstance(params, jax.core.Tracer):
                self.params = params
            else:
                log.debug(
                    "`params` is a JAX tracer; `self.params` is left at its "
                    "previous value. Anything reading model state afterwards "
                    "(draw, Entanglement, Expressibility, or a call that omits "
                    "`params`) will see the stale parameters - assign "
                    "`model.params` explicitly if you need the state to follow."
                )
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
                (batch_size, n_layers, n_pulse_params_per_layer).
        """
        if pulse_params is None:
            pulse_params = self.pulse_params
        else:
            # ensure batch dimension exists (batch-first convention)
            if len(pulse_params.shape) == 2:
                pulse_params = jnp.expand_dims(pulse_params, axis=0)
            # See note in _params_validation: never stash JAX tracers on
            # ``self``.
            if not isinstance(pulse_params, jax.core.Tracer):
                self.pulse_params = pulse_params
            else:
                log.debug(
                    "`pulse_params` is a JAX tracer; `self.pulse_params` is "
                    "left at its previous value."
                )

        return pulse_params

    def _enc_pulse_params_validation(
        self, enc_pulse_params: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Validate and normalize encoding pulse parameters.

        Ensures encoding pulse parameters are set (using model defaults if not
        provided) and carry a leading batch dimension.

        Args:
            enc_pulse_params (Optional[jnp.ndarray]): Encoding pulse-parameter
                scalers. If None, returns the model's current encoding pulse
                parameters.

        Returns:
            jnp.ndarray: Validated encoding pulse parameters with shape
                (batch_size, n_layers, n_qubits, n_enc_pulse_per_qubit).

        Raises:
            ValueError: If the trailing dimensions do not match the model's
                encoding pulse parameter shape.
        """
        if enc_pulse_params is None:
            enc_pulse_params = self.enc_pulse_params
        else:
            # ensure batch dimension exists (batch-first convention)
            if len(enc_pulse_params.shape) == 3:
                enc_pulse_params = jnp.expand_dims(enc_pulse_params, axis=0)
            if enc_pulse_params.shape[1:] != self._enc_pulse_shape:
                raise ValueError(
                    f"enc_pulse_params trailing shape {enc_pulse_params.shape[1:]} "
                    f"does not match expected {self._enc_pulse_shape}."
                )
            # See note in _params_validation: never stash JAX tracers on
            # ``self``.
            self.enc_pulse_params = enc_pulse_params

        return enc_pulse_params

    def _resolve_gate_mode(
        self,
        gate_mode: Optional[str],
        pulse_params: Optional[jnp.ndarray],
        enc_pulse_params: Optional[jnp.ndarray],
    ) -> str:
        """
        Determine which gate groups run at pulse level.

        The mode follows from the pulse parameters that were provided:
        ``pulse_params`` lowers the ansatz and state-preparation gates,
        ``enc_pulse_params`` lowers the encoding gates, both together lower
        everything.

        Args:
            gate_mode (Optional[str]): Deprecated explicit mode. If None, the
                mode is inferred from the pulse parameters.
            pulse_params (Optional[jnp.ndarray]): Ansatz pulse-parameter scalers.
            enc_pulse_params (Optional[jnp.ndarray]): Encoding pulse-parameter
                scalers.

        Returns:
            str: One of the keys of ``GATE_MODES``.

        Raises:
            ValueError: If the encoding gates would run at pulse level but the
                encoding has no pulse parametrization, or if an explicitly
                passed (deprecated) gate_mode is unknown or inconsistent with
                the provided pulse parameters.
        """
        if gate_mode is None:
            if pulse_params is not None and enc_pulse_params is not None:
                gate_mode = "all_pulse"
            elif pulse_params is not None:
                gate_mode = "ansatz_pulse"
            elif enc_pulse_params is not None:
                gate_mode = "enc_pulse"
            else:
                gate_mode = "unitary"
        else:
            warnings.warn(
                "gate_mode is deprecated, the mode is inferred from the "
                "provided pulse parameters instead. Pass pulse_params to run "
                "the ansatz at pulse level and enc_pulse_params to run the "
                "encoding at pulse level, e.g. "
                "model(pulse_params=model.pulse_params).",
                DeprecationWarning,
                stacklevel=4,
            )
            # consistency checks, only reachable via the deprecated argument
            if gate_mode not in GATE_MODES:
                raise ValueError(
                    f"Unknown gate_mode: {gate_mode}. Use one of {list(GATE_MODES)}."
                )
            if pulse_params is not None and gate_mode not in _ANSATZ_PULSE_MODES:
                raise ValueError(
                    f"pulse_params were provided but gate_mode is not one of "
                    f"{list(_ANSATZ_PULSE_MODES)}. Either switch gate_mode or do "
                    "not pass pulse_params."
                )
            if enc_pulse_params is not None and gate_mode not in _ENC_PULSE_MODES:
                raise ValueError(
                    f"enc_pulse_params were provided but gate_mode is not one of "
                    f"{list(_ENC_PULSE_MODES)}. Either switch gate_mode or do not "
                    "pass enc_pulse_params."
                )

        if gate_mode in _ENC_PULSE_MODES and not self._enc_pulse_capable:
            raise ValueError(
                "Pulse-level encoding requires an encoding whose gates have a "
                "pulse parametrization (golomb and custom callables do not). "
                "Do not pass enc_pulse_params for this model."
            )

        return gate_mode

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
            # See note in _params_validation: never stash JAX tracers on
            # ``self``.
            if not isinstance(enc_params, jax.core.Tracer):
                if self.trainable_frequencies:
                    self.enc_params = enc_params
                else:
                    self.enc_params = jnp.array(enc_params)
            else:
                log.debug(
                    "`enc_params` is a JAX tracer; `self.enc_params` is left "
                    "at its previous value."
                )

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
        if isinstance(inputs, List):
            inputs = jnp.array(np.stack(inputs))
        elif isinstance(inputs, float) or isinstance(inputs, int):
            inputs = jnp.array([inputs])
        elif inputs is None:
            inputs = jnp.array([[0] * self.n_input_feat])

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
        enc_pulse_params: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Align batch dimensions across inputs, parameters, pulse parameters and
        encoding pulse parameters.

        Broadcasts and reshapes arrays to have compatible batch dimensions
        for vectorized circuit execution. Sets the internal batch_shape.

        The batch layout is ``[B_I, B_P, B_R, B_E, <payload>]`` where each array
        "owns" one batch axis and is replicated across the others (subject to
        the ``repeat_batch_axis`` mask) before being flattened to ``B``.

        Args:
            inputs (jnp.ndarray): Input data of shape (B_I, n_input_feat).
            params (jnp.ndarray): Parameters of shape (B_P, n_layers, n_params).
            pulse_params (jnp.ndarray): Pulse params of shape (B_R, n_layers, n_pulse).
            enc_pulse_params (jnp.ndarray): Encoding pulse params of shape
                (B_E, n_layers, n_qubits, n_enc_pulse).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: The four
            arrays, each reshaped to leading dimension B = B_I * B_P * B_R * B_E
            (subject to repeat_batch_axis).

        Note:
            The effective batch shape depends on repeat_batch_axis configuration.
            This is the only method that sets self._batch_shape.
        """
        B_I = inputs.shape[0]
        # we check for the product because there is a chance that
        # there are no params. In this case we want B_P to be 1
        B_P = 1 if 0 in params.shape else params.shape[0]
        B_R = pulse_params.shape[0]
        B_E = enc_pulse_params.shape[0]

        # THIS is the only place where we set the batch shape
        self._batch_shape = (B_I, B_P, B_R, B_E)
        B = np.prod(self.eff_batch_shape)

        # [B_I, ...] -> [B_I, B_P, B_R, B_E, ...] -> [B, ...]
        if B_I > 1 and self.repeat_batch_axis[0]:
            inputs = inputs[:, None, None, None, ...]
            if self.repeat_batch_axis[1]:
                inputs = jnp.repeat(inputs, B_P, axis=1)
            if self.repeat_batch_axis[2]:
                inputs = jnp.repeat(inputs, B_R, axis=2)
            if self.repeat_batch_axis[3]:
                inputs = jnp.repeat(inputs, B_E, axis=3)
            inputs = inputs.reshape(B, *inputs.shape[4:])

        # [B_P, ...] -> [B_I, B_P, B_R, B_E, ...] -> [B, ...]
        if B_P > 1 and self.repeat_batch_axis[1]:
            params = params[None, :, None, None, ...]  # [1, B_P, 1, 1, ...]
            if self.repeat_batch_axis[0]:
                params = jnp.repeat(params, B_I, axis=0)
            if self.repeat_batch_axis[2]:
                params = jnp.repeat(params, B_R, axis=2)
            if self.repeat_batch_axis[3]:
                params = jnp.repeat(params, B_E, axis=3)
            params = params.reshape(B, *params.shape[4:])

        # [B_R, ...] -> [B_I, B_P, B_R, B_E, ...] -> [B, ...]
        if B_R > 1 and self.repeat_batch_axis[2]:
            pulse_params = pulse_params[None, None, :, None, ...]  # [1, 1, B_R, 1, ...]
            if self.repeat_batch_axis[0]:
                pulse_params = jnp.repeat(pulse_params, B_I, axis=0)
            if self.repeat_batch_axis[1]:
                pulse_params = jnp.repeat(pulse_params, B_P, axis=1)
            if self.repeat_batch_axis[3]:
                pulse_params = jnp.repeat(pulse_params, B_E, axis=3)
            pulse_params = pulse_params.reshape(B, *pulse_params.shape[4:])

        # [B_E, ...] -> [B_I, B_P, B_R, B_E, ...] -> [B, ...]
        if B_E > 1 and self.repeat_batch_axis[3]:
            enc_pulse_params = enc_pulse_params[
                None, None, None, ...
            ]  # [1,1,1,B_E,...]
            if self.repeat_batch_axis[0]:
                enc_pulse_params = jnp.repeat(enc_pulse_params, B_I, axis=0)
            if self.repeat_batch_axis[1]:
                enc_pulse_params = jnp.repeat(enc_pulse_params, B_P, axis=1)
            if self.repeat_batch_axis[2]:
                enc_pulse_params = jnp.repeat(enc_pulse_params, B_R, axis=2)
            enc_pulse_params = enc_pulse_params.reshape(B, *enc_pulse_params.shape[4:])

        return inputs, params, pulse_params, enc_pulse_params

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

    def _is_stochastic(self) -> bool:
        """
        Check if execution draws random numbers at runtime.

        Only coherent gate errors and shot sampling are stochastic; the Kraus
        channels are deterministic maps on the density matrix.

        Returns:
            bool: True if the result depends on the random key.
        """
        gate_error = (self.noise_params or {}).get("GateError") or 0
        return gate_error > 0 or self.shots is not None

    @staticmethod
    def _args_are_traced(*args: Any) -> bool:
        """
        Check if any argument is a JAX tracer.

        Args:
            *args (Any): Values to inspect, may be pytrees.

        Returns:
            bool: True if the call runs inside a JAX transform.
        """
        return any(
            isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(args)
        )

    @staticmethod
    def _observable_id(obs: op.Operation) -> Any:
        """
        Get a stable identity for an observable.

        Uses the Pauli label where available and otherwise a hash of the
        matrix, memoized on the instance because reading the bytes copies the
        full $2^n \\times 2^n$ array. Mutating a matrix in place is not
        detected.

        Args:
            obs (op.Operation): Observable to identify.

        Returns:
            Any: Hashable identity of the observable.
        """
        label = getattr(obs, "_pauli_label", None)
        if label is not None:
            return label
        if getattr(obs, "_fingerprint_hash", None) is None:
            obs._fingerprint_hash = hash(np.asarray(obs.matrix).tobytes())
        return obs._fingerprint_hash

    def _structural_fingerprint(self) -> Tuple:
        """
        Summarize the circuit structure for the execution plan cache.

        Covers everything that :meth:`_variational` and :meth:`_iec` read from
        the model while recording the tape and that can change after
        initialization without changing the shapes of the execution arguments.
        Without it, a batched call would silently reuse a plan that was
        compiled for the previous structure.

        Attributes that are fixed at initialization (the encoding, the state
        preparation, the number of qubits and layers) are omitted, as replacing
        them afterwards is not supported.

        Returns:
            Tuple: Hashable structure summary, passed to
                :meth:`~qml_essentials.jaqsi.Script.execute`.
        """
        if self._observables is None:
            obs_fingerprint = None
        else:
            obs_fingerprint = tuple(
                (o.name, tuple(o.wires), self._observable_id(o))
                for o in self._observables
            )

        return (
            self._data_reupload.shape,
            # covers the derived degree, frequencies and has_dru as well
            self._data_reupload.tobytes(),
            _make_hashable(self._measured_wires),
            obs_fingerprint,
            # hashed by identity, which also covers a replaced ansatz callable
            self.pqc,
        )

    def __call__(
        self,
        params: Optional[jnp.ndarray] = None,
        inputs: Optional[jnp.ndarray] = None,
        pulse_params: Optional[jnp.ndarray] = None,
        enc_params: Optional[jnp.ndarray] = None,
        data_reupload: Union[bool, List[List[bool]], List[List[List[bool]]]] = None,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        execution_type: Optional[str] = None,
        force_mean: bool = False,
        gate_mode: Optional[str] = None,
        enc_pulse_params: Optional[jnp.ndarray] = None,
        random_key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Execute the quantum circuit (callable interface).

        Provides a convenient callable interface for circuit execution,
        delegating to the _forward method.

        Args:
            params (Optional[jnp.ndarray]): Variational parameters of shape
                (n_layers, n_params_per_layer) or (batch, n_layers, n_params_per_layer).
                If None, uses model's internal parameters.
            inputs (Optional[jnp.ndarray]): Input data of shape
                (batch_size, n_input_feat). If None, uses zero inputs.
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers for
                the ansatz and state-preparation gates. Passing them runs those
                gates at pulse level. If None, they stay unitary.
            enc_params (Optional[jnp.ndarray]): Encoding parameters of shape
                (n_qubits, n_input_feat). If None, uses model's encoding parameters.
            data_reupload (Union[bool, List[List[bool]], List[List[List[bool]]]]):
                Data reupload configuration. If None, uses previously set reupload
                configuration.
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise configuration. If None, uses previously set noise parameters.
            execution_type (Optional[str]): Measurement type: "expval", "density",
                "probs", or "state". If None, uses current execution_type setting.
            force_mean (bool): If True, averages results over measurement qubits.
                Defaults to False.
            gate_mode (Optional[str]): Deprecated. If None (default), the gate
                execution backend is inferred from the provided pulse
                parameters: ``pulse_params`` runs the ansatz and state
                preparation at pulse level, ``enc_pulse_params`` the encoding
                gates, both together everything. Passing "unitary",
                "ansatz_pulse", "enc_pulse" or "all_pulse" explicitly still
                works but emits a DeprecationWarning.
            enc_pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers
                for the encoding gates. Passing them runs the encoding gates at
                pulse level. If None, they stay unitary.
            random_key (Optional[random.PRNGKey]): JAX random key for stochastic
                execution (``GateError`` noise and shot sampling). If provided,
                the caller owns key advancement and the model's internal
                ``random_key`` is left untouched - this is the jit-safe way to
                get fresh randomness per call, since the internal key cannot be
                advanced from inside a trace. Use :meth:`next_key` to obtain
                one. If None, the internal key is used and advanced (eager
                calls only).

        Returns:
            jnp.ndarray: Circuit output with shape depending on execution_type:
                - "expval": (n_measured_wiress,) or scalar
                - "density": (2^n_output, 2^n_output)
                - "probs": (2^n_output,) or (n_pairs, 2^pair_size)
                - "state": (2^n_qubits,)

        Note:
            An eager call stores ``params``, ``pulse_params`` and ``enc_params``
            on the model, but a traced call (``jit``, ``grad``, ``vmap``) does
            not: JAX tracers must not outlive their transform, so the model
            state keeps its previous value. Two consequences:

            - Anything reading model state after a traced call - ``draw``,
              :class:`~qml_essentials.entanglement.Entanglement`,
              :class:`~qml_essentials.expressibility.Expressibility`, or a
              later call that omits ``params`` - sees the *old* parameters.
              Assign ``model.params = params`` yourself if the state should
              follow a traced optimization step.
            - Omitting ``params`` in a second call inside the same trace falls
              back to that stale state, so the result does not depend on the
              traced parameters (its gradient is zero). Pass ``params``
              explicitly on every call inside a trace.

            The skipped writes are reported at debug log level.
        """
        # Call forward method which handles the actual caching etc.
        return self._forward(
            params=params,
            inputs=inputs,
            pulse_params=pulse_params,
            enc_params=enc_params,
            data_reupload=data_reupload,
            noise_params=noise_params,
            execution_type=execution_type,
            force_mean=force_mean,
            gate_mode=gate_mode,
            enc_pulse_params=enc_pulse_params,
            random_key=random_key,
        )

    def _forward(
        self,
        params: Optional[jnp.ndarray] = None,
        inputs: Optional[jnp.ndarray] = None,
        pulse_params: Optional[jnp.ndarray] = None,
        enc_params: Optional[jnp.ndarray] = None,
        data_reupload: Union[bool, List[List[bool]], List[List[List[bool]]]] = None,
        noise_params: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        execution_type: Optional[str] = None,
        force_mean: bool = False,
        gate_mode: Optional[str] = None,
        enc_pulse_params: Optional[jnp.ndarray] = None,
        random_key: Optional[random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Execute the quantum circuit forward pass.

        Internal implementation of the forward pass that handles parameter
        validation, batch alignment, and circuit execution routing.

        Args:
            params (Optional[jnp.ndarray]): Variational parameters of shape
                (n_layers, n_params_per_layer) or
                (batch, n_layers, n_params_per_layer).
                If None, uses model's internal parameters.
            inputs (Optional[jnp.ndarray]): Input data of shape
                (batch_size, n_input_feat).
                If None, uses zero inputs.
            pulse_params (Optional[jnp.ndarray]): Pulse parameter scalers for
                pulse-mode gate execution.
            enc_params (Optional[jnp.ndarray]): Encoding parameters of shape
                (n_qubits, n_input_feat). If None, uses model's encoding parameters.
            data_reupload (Union[bool, List[List[bool]], List[List[List[bool]]]]):
                Data reupload configuration. If None, uses previously set reupload
                configuration.
            noise_params (Optional[Dict[str, Union[float, Dict[str, float]]]]):
                Noise configuration. If None, uses previously set noise parameters.
            execution_type (Optional[str]): Measurement type: "expval", "density",
                "probs", or "state". If None, uses current execution_type setting.
            force_mean (bool): If True, averages results over measurement qubits.
                Defaults to False.
            gate_mode (Optional[str]): Deprecated. If None (default), the mode
                is inferred from the provided pulse parameters. Passing
                "unitary", "ansatz_pulse", "enc_pulse" or "all_pulse"
                explicitly emits a DeprecationWarning.
            random_key (Optional[random.PRNGKey]): JAX random key for stochastic
                execution. If provided, it is used instead of (and does not
                modify) the model's internal ``random_key``. See
                :meth:`__call__` for details.

        Returns:
            jnp.ndarray: Circuit output with shape depending on execution_type:
                - "expval": (n_measured_wiress,) or scalar
                - "density": (2^n_output, 2^n_output)
                - "probs": (2^n_output,) or (n_pairs, 2^pair_size)
                - "state": (2^n_qubits,)

        Raises:
            ValueError: If the encoding gates would run at pulse level but the
                encoding has no pulse parametrization, or if an explicitly
                passed (deprecated) gate_mode is unknown or inconsistent with
                the provided pulse parameters.
        """
        # set the parameters as object attributes
        if noise_params is not None:
            self.noise_params = noise_params
        if execution_type is not None:
            self.execution_type = execution_type

        gate_mode = self._resolve_gate_mode(gate_mode, pulse_params, enc_pulse_params)

        # TODO: add testing
        if data_reupload is not None:
            self.data_reupload = data_reupload

        params = self._params_validation(params)
        pulse_params = self._pulse_params_validation(pulse_params)
        inputs = self._inputs_validation(inputs)
        enc_params = self._enc_params_validation(enc_params)
        enc_pulse_params = self._enc_pulse_params_validation(enc_pulse_params)

        inputs, params, pulse_params, enc_pulse_params = self._assimilate_batch(
            inputs,
            params,
            pulse_params,
            enc_pulse_params,
        )

        # split to generate a sub_key, required for actual execution.
        if random_key is not None:
            # explicit key: purely functional, the caller advances it. This is
            # the only way to get fresh randomness inside a trace, because a
            # jitted call is traced once and then replays the trace-time key.
            _, sub_key = safe_random_split(random_key)
        else:
            if self._is_stochastic() and self._args_are_traced(
                params, inputs, pulse_params, enc_pulse_params
            ):
                warnings.warn(
                    "Stochastic execution (`GateError` or `shots`) without an "
                    "explicit `random_key` inside a JAX transform: the key is "
                    "read at trace time, so a jitted function replays the same "
                    "noise realization on every call. Pass "
                    "`random_key=model.next_key()` from outside the transform.",
                    UserWarning,
                )
            # Under JAX tracing (jit) the split result is a tracer; stashing it
            # on ``self`` leaks the tracer across calls (UnexpectedTracerError),
            # so only advance the key eagerly. Note that a jitted call
            # therefore reuses the same key on every execution - pass
            # ``random_key`` explicitly if that matters.
            new_key, sub_key = safe_random_split(self.random_key)
            if not isinstance(new_key, jax.core.Tracer):
                self.random_key = new_key

        # Build measurement type & observables from execution_type / output_qubit
        meas_type, obs = self._build_obs()

        # Jaqsi auto-routes between statevector and density-matrix simulation
        # based on whether noise channels appear on the tape, so a single
        B = np.prod(self.eff_batch_shape)

        # kwargs are broadcast (not vmapped over)
        exec_kwargs = dict(
            noise_params=self.noise_params,
            gate_mode=gate_mode,
        )

        # Build a shot key from the random_key if shots are requested
        shot_key = None
        if self.shots is not None:
            # overwrite subkey and split shot_key
            sub_key, shot_key = safe_random_split(sub_key)

        if B > 1:
            # use random keys, derived from the subkey
            random_keys = safe_random_split(sub_key, num=B)

            in_axes = (
                0 if self.batch_shape[1] > 1 else None,  # params
                0 if self.batch_shape[0] > 1 else None,  # inputs
                0 if self.batch_shape[2] > 1 else None,  # pulse_params
                0,  # random_keys
                None,  # enc_params (broadcast, not batched)
                0 if self.batch_shape[3] > 1 else None,  # enc_pulse_params
            )

            result = self.script.execute(
                type=meas_type,
                obs=obs,
                args=(
                    params,
                    inputs,
                    pulse_params,
                    random_keys,
                    enc_params,
                    enc_pulse_params,
                ),
                kwargs=exec_kwargs,
                in_axes=in_axes,
                shots=self.shots,
                key=shot_key,
                fingerprint=self._structural_fingerprint(),
            )
        else:
            # use the subkey directly
            result = self.script.execute(
                type=meas_type,
                obs=obs,
                args=(
                    params,
                    inputs,
                    pulse_params,
                    sub_key,
                    enc_params,
                    enc_pulse_params,
                ),
                kwargs=exec_kwargs,
                shots=self.shots,
                key=shot_key,
                fingerprint=self._structural_fingerprint(),
            )

        result = self._postprocess_res(result)

        # --- Post-processing for partial-qubit measurements ---------------
        if self.execution_type == "density" and not self.all_qubit_measurement:
            result = js.partial_trace(result, self.n_qubits, self._measured_wires)

        if self.execution_type == "probs" and not self.all_qubit_measurement:
            if isinstance(self._measured_wires[0], (list, tuple)):
                # list of qubit groups - marginalize each independently
                result = jnp.stack(
                    [
                        js.marginalize_probs(result, self.n_qubits, list(group))
                        for group in self._measured_wires
                    ]
                )
            else:
                result = js.marginalize_probs(
                    result, self.n_qubits, self._measured_wires
                )

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
