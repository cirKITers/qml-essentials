from typing import List, Tuple
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.typing import PostprocessingFn
from qml_essentials.model import Model
from functools import partial
from pennylane.fourier import coefficients
import numpy as np

CLIFFORD_GATES = (
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.X,
    qml.Y,
    qml.Z,
    qml.Hadamard,
    qml.S,
    qml.CNOT,
)

PAULI_ROTATION_GATES = (
    qml.RX,
    qml.RY,
    qml.RZ,
)

SKIPPABLE_OPERATIONS = (qml.Barrier,)


class Coefficients:

    @staticmethod
    def sample_coefficients(model: Model, **kwargs) -> np.ndarray:
        """
        Sample the Fourier coefficients of a given model
        using Pennylane fourier.coefficients function.

        Note that the coefficients are complex numbers, but the imaginary part
        of the coefficients should be very close to zero, since the expectation
        values of the Pauli operators are real numbers.

        Args:
            model (Model): The model to sample.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            np.ndarray: The sampled Fourier coefficients.
        """
        kwargs.setdefault("force_mean", True)
        kwargs.setdefault("execution_type", "expval")

        partial_circuit = partial(model, model.params, **kwargs)
        coeffs = coefficients(partial_circuit, 1, model.degree)

        if not np.isclose(np.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is:\
                {np.sum(coeffs).imag}"
            )

        return coeffs

    @staticmethod
    def evaluate_Fourier_series(
        coefficients: np.ndarray, input: float
    ) -> float:
        """
        Evaluate the function value of a Fourier series at one point.

        Args:
            coefficients (np.ndarray): Coefficients of the Fourier series.
            input (float): Point at which to evaluate the function.

        Returns:
            float: The function value at the input point.
        """
        n_freq = len(coefficients) // 2
        pos_coeff = coefficients[1 : n_freq + 1]
        neg_coeff = coefficients[n_freq + 1 :][::-1]

        assert all(np.isclose(np.conjugate(pos_coeff), neg_coeff, atol=1e-5)), (
            "Coefficients for negative frequencies should be the complex "
            "conjugate of the respective positive ones."
        )

        exp = coefficients[0]
        for omega in range(1, n_freq + 1):
            exp += pos_coeff[omega - 1] * np.exp(1j * omega * input)
            exp += neg_coeff[omega - 1] * np.exp(-1j * omega * input)
        return exp


class PauliCircuit:
    """
    Wrapper for Pauli-Clifford Circuits described by Nemkov et al.
    (https://doi.org/10.1103/PhysRevA.108.032406). The code is inspired
    by the corresponding implementation: https://github.com/idnm/FourierVQA.

    A Pauli Circuit only consists of parameterised Pauli-rotations and Clifford
    gates, which is the default for the most common VQCs.
    """

    def __init__(self, paulis, final_clifford=None, parameters=None):
        self.paulis = paulis
        self.final_clifford = final_clifford
        self.parameters = parameters

    @qml.transform
    @staticmethod
    def from_parameterised_circuit(
        tape: QuantumScript,
    ) -> tuple[QuantumScriptBatch, PostprocessingFn]:

        operations = PauliCircuit.get_clifford_pauli_gates(tape)

        pauli_gates, final_clifford = (
            PauliCircuit.commute_all_cliffords_to_the_end(operations)
        )

        with QuantumTape() as tape_new:
            for op in pauli_gates:
                tape_new.append(op)
            for op in final_clifford:
                # TODO: actually move clifford gates to observable
                tape_new.append(op)
            for obs in tape.observables:
                tape_new.observables.append(qml.expval(obs))

        def postprocess(res):
            return res

        return [
            tape_new,
        ], postprocess

    @staticmethod
    def commute_all_cliffords_to_the_end(
        operations: List[Operator],
    ) -> Tuple[List[Operator], List[Operator]]:
        raise NotImplementedError("TODO")

    @staticmethod
    def get_clifford_pauli_gates(tape: QuantumScript) -> List[Operator]:
        """
        Unroll the circuit and identify each gate either as a Clifford gate or
        as a Pauli rotation.
        """
        operations = []
        for operation in tape.operations:
            if PauliCircuit._is_clifford(
                operation
            ) or PauliCircuit._is_pauli_rotation(operation):
                operations.append(operation)
            elif PauliCircuit._is_skippable(operation):
                continue
            elif PauliCircuit._is_skippable(operation):
                continue
            else:
                raise NotImplementedError(
                    f"Gate {operation.name} is neither Clifford nor Pauli rotation "
                    "and a conversion to Cifford+Pauli gates is not "
                    "implemented, yet."
                )

        return operations

    @staticmethod
    def _is_skippable(operation: Operator) -> bool:
        return isinstance(operation, SKIPPABLE_OPERATIONS)

    @staticmethod
    def _is_clifford(operation: Operator) -> bool:
        return isinstance(operation, CLIFFORD_GATES)

    @staticmethod
    def _is_pauli_rotation(operation: Operator) -> bool:
        return isinstance(operation, PAULI_ROTATION_GATES)
