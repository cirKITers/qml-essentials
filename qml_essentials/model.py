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
        self.n_qubits: int = n_qubits
        self.n_layers: int = n_layers
        self.data_reupload: bool = data_reupload
        self.output_qubit: int = output_qubit
        self.pqc: Callable[[Optional[np.ndarray], int], int] = getattr(
            Ansaetze, circuit_type or "no_ansatz"
        )()

        log.info(f"Using {circuit_type} circuit.")

        if data_reupload:
            impl_n_layers: int = n_layers + 1  # we need L+1 according to Schuld et al.
            self.degree = n_layers * n_qubits
        else:
            impl_n_layers: int = n_layers
            self.degree = 0

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

        self.dev: qml.Device = qml.device("default.mixed", wires=n_qubits)

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
        if data_reupload:
            for q in range(self.n_qubits):
                qml.RX(inputs, wires=q)
        else:
            qml.RX(inputs, wires=0)

    def _circuit(
        self,
        params: np.ndarray,
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        state_vector: Optional[bool] = False,
        exp_val: Optional[bool] = True,
    ) -> Union[float, np.ndarray]:
        """
        Creates a circuit with noise.
        This involves, Bit Flip, Phase Flip, Amplitude Damping,
        Phase Damping and Depolarization.
        The Circuit consists of a PQC and IEC in each layer
        with the PQC as specified in the construction of the model.

        Args:
            params (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
            inputs (np.ndarray): input vector of size 1
            noise_params (Optional[Dict[str, float]]): dictionary with noise parameters
                - "BitFlip": float, default = 0.0
                - "PhaseFlip": float, default = 0.0
                - "AmplitudeDamping": float, default = 0.0
                - "PhaseDamping": float, default = 0.0
                - "DepolarizingChannel": float, default = 0.0
            state_vector (bool, optional): Whether to measure the state vector
                instead of the wave function. Defaults to False.
            exp_val (bool, optional): Whether to measure the expectation value
                of PauliZ(0) of the circuit. Defaults to True.

        Returns:
            Union[float, np.ndarray]: Expectation value of PauliZ(0) of the circuit if
                state_vector is False and exp_val is True, otherwise the density matrix
                of all qubits.
        """

        for l in range(0, self.n_layers):
            self.pqc(params[l], self.n_qubits)

            if self.data_reupload or l == 0:
                self._iec(inputs, data_reupload=self.data_reupload)

            if noise_params is not None:
                for q in range(self.n_qubits):
                    qml.BitFlip(noise_params.get("BitFlip", 0.0), wires=q)
                    qml.PhaseFlip(noise_params.get("PhaseFlip", 0.0), wires=q)
                    qml.AmplitudeDamping(
                        noise_params.get("AmplitudeDamping", 0.0), wires=q
                    )
                    qml.PhaseDamping(noise_params.get("PhaseDamping", 0.0), wires=q)
                    qml.DepolarizingChannel(
                        noise_params.get("DepolarizingChannel", 0.0), wires=q
                    )

        if self.data_reupload:
            self.pqc(params[-1], self.n_qubits)

        if state_vector:
            return qml.density_matrix(wires=list(range(self.n_qubits)))
        elif exp_val:
            return qml.expval(qml.PauliZ(self.output_qubit))
        else:
            if self.output_qubit == -1:
                return qml.probs(wires=list(range(self.n_qubits)))
            else:
                return qml.probs(wires=self.output_qubit)

    def _draw(self) -> None:
        return qml.draw(self.circuit)(params=self.params, inputs=[0])

    def __repr__(self) -> str:
        return self._draw()

    def __str__(self) -> str:
        return self._draw()

    def __call__(
        self,
        params: Optional[np.ndarray],
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        state_vector: bool = False,
        exp_val: bool = True,
    ) -> np.ndarray:
        """Perform a forward pass of the quantum circuit.

        Args:
            inputs (np.ndarray): input vector of size 1
            params (Optional[np.ndarray], optional): weight vector of size n_layers*(n_qubits*3-1). Defaults to None.
            noise_params (Optional[Dict[str, float]], optional): dictionary with noise parameters. Defaults to None.
            cache (Optional[bool], optional): cache the circuit. Defaults to False.
            state_vector (bool, optional): measure the state vector instead of the wave function. Defaults to False.

        Returns:
            np.ndarray: Expectation value of PauliZ(0) of the circuit.
        """
        # Call forward method which handles the actual caching etc.
        return self._forward(params, inputs, noise_params, cache, state_vector, exp_val)

    def _forward(
        self,
        params: Optional[np.ndarray],
        inputs: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
        cache: Optional[bool] = False,
        state_vector: bool = False,
        exp_val: bool = True,
    ) -> np.ndarray:
        """Perform a forward pass of the quantum circuit.

        Args:
            inputs (np.ndarray): The input data.
            params (Optional[np.ndarray], optional): The weights of the quantum
                circuit. If None, uses the current weights of the Instructor instance.
                Defaults to None.
            noise_params (Optional[Dict[str, float]], optional): The noise parameters.
                Defaults to None.
            cache (Optional[bool], optional): Whether to cache the results. Defaults to False.
            state_vector (bool, optional): Whether to return the state vector instead of the
                expectation value. Defaults to False.

        Returns:
            np.ndarray: The output of the quantum circuit.
        """
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

        result = None
        if cache:
            name = f"pqc_{hs}.npy"

            cache_folder = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path = os.path.join(cache_folder, name)

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
