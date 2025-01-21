from typing import List, Tuple
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.typing import PostprocessingFn

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
    qml.PauliRot,
)

SKIPPABLE_OPERATIONS = (qml.Barrier,)


class PauliCircuit:
    """
    Wrapper for Pauli-Clifford Circuits described by Nemkov et al.
    (https://doi.org/10.1103/PhysRevA.108.032406). The code is inspired
    by the corresponding implementation: https://github.com/idnm/FourierVQA.

    A Pauli Circuit only consists of parameterised Pauli-rotations and Clifford
    gates, which is the default for the most common VQCs.
    """

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
                op.queue()
            for op in final_clifford:
                # TODO: actually move clifford gates to observable
                op.queue()
            for obs in tape.observables:
                qml.expval(obs)

        def postprocess(res):
            return res[0]

        return [tape_new], postprocess

    @staticmethod
    def commute_all_cliffords_to_the_end(
        operations: List[Operator],
    ) -> Tuple[List[Operator], List[Operator]]:

        first_clifford = -1
        for i in range(len(operations) - 2, -1, -1):
            j = i
            while (
                j + 1
                < len(operations)  # Clifford has not alredy reached the end
                and PauliCircuit._is_clifford(operations[j])
                and PauliCircuit._is_pauli_rotation(operations[j + 1])
            ):
                pauli, clifford = PauliCircuit._evolve_clifford(
                    operations[j], operations[j + 1]
                )
                operations[j] = pauli
                operations[j + 1] = clifford
                j += 1
                first_clifford = j

        pauli_rotations = operations[:first_clifford]
        clifford_gates = operations[first_clifford:]

        return pauli_rotations, clifford_gates

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

    @staticmethod
    def _evolve_clifford(
        clifford: Operator, pauli: Operator
    ) -> Tuple[Operator, Operator]:
        if not isinstance(clifford, qml.CNOT):
            raise NotImplementedError(
                "Currently we only support CNOT as clifford gate."
            )

        gen = pauli.generator()
        param = pauli.parameters[0]

        evolved_gen = clifford @ gen @ qml.adjoint(clifford)
        qubits = evolved_gen.wires
        evolved_gen = qml.pauli_decompose(evolved_gen.matrix())
        pauli_str, param_factor = PauliCircuit._get_paulistring_from_generator(
            evolved_gen
        )
        pauli = qml.PauliRot(param * param_factor, pauli_str, qubits)

        return pauli, clifford

    @staticmethod
    def _get_paulistring_from_generator(
        gen: qml.ops.op_math.LinearCombination,
    ) -> Tuple[str, float]:
        factor, term = gen.terms()
        assert factor[0] in [-0.5, 0.5]
        param_factor = -2 * factor  # Rotation is defined as exp(-0.5 theta G)

        pauli_term = term[0]
        pauli_str_list = ["I"] * len(pauli_term)
        for p in pauli_term:
            if "Pauli" in p.name:
                q = p.wires[0]
                pauli_str_list[q] = p.name[-1]
        pauli_str = "".join(pauli_str_list)
        return pauli_str, param_factor
