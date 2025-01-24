from __future__ import annotations
from typing import List, Tuple, Optional
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.typing import PostprocessingFn
import pennylane.ops.op_math as qml_op

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

    @staticmethod
    def from_parameterised_circuit(
        tape: QuantumScript,
    ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
        """
        Transformation function (see also qml.transforms) to convert an ansatz
        into a Pauli-Clifford circuit.


        **Usage** (without using Model, Model provides a boolean argument
               "as_pauli_circuit" that internally uses the Pauli-Clifford):
        ```
        # initialise some QNode
        circuit = qml.QNode(
            circuit_fkt,  # function for your circuit definition
            qml.device("default.qubit", wires=5),
        )
        pauli_circuit = PauliCircuit.from_parameterised_circuit(circuit)

        # Call exactly the same as circuit
        some_input = [0.1, 0.2]

        circuit(some_input)
        pauli_circuit(some_input)

        # Both results should be equal!
        ```

        Args:
            tape (QuantumScript): The quantum tape for the operations in the
                ansatz. This is automatically passed, when initialising the
                transform function with a QNode. Note: directly calling
                `PauliCircuit.from_parameterised_circuit(circuit)` for a QNode
                circuit will fail, see usage above.

        Returns:
            tuple[QuantumScriptBatch, PostprocessingFn]:
                - A new quantum tape, containing the operations of the
                  Pauli-Clifford Circuit.
                - A postprocessing function that does nothing.
        """

        operations = PauliCircuit.get_clifford_pauli_gates(tape)

        pauli_gates, final_cliffords = (
            PauliCircuit.commute_all_cliffords_to_the_end(operations)
        )

        observables = PauliCircuit.cliffords_in_observable(
            final_cliffords, tape.observables
        )

        with QuantumTape() as tape_new:
            for op in pauli_gates:
                op.queue()
            for obs in observables:
                qml.expval(obs)

        def postprocess(res):
            return res[0]

        return [tape_new], postprocess

    @staticmethod
    def commute_all_cliffords_to_the_end(
        operations: List[Operator],
    ) -> Tuple[List[Operator], List[Operator]]:
        """
        This function moves all clifford gates to the end of the circuit,
        accounting for commutation rules.

        Args:
            operations (List[Operator]): The operations in the tape of the
                circuit

        Returns:
            Tuple[List[Operator], List[Operator]]:
                - List of the resulting Pauli-rotations
                - List of the resulting Clifford gates
        """

        first_clifford = -1
        for i in range(len(operations) - 2, -1, -1):
            j = i
            while (
                j + 1
                < len(operations)  # Clifford has not alredy reached the end
                and PauliCircuit._is_clifford(operations[j])
                and PauliCircuit._is_pauli_rotation(operations[j + 1])
            ):
                pauli, clifford = PauliCircuit._evolve_clifford_rotation(
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
        This function decomposes all gates in the circuit to clifford and
        pauli-rotation gates

        Args:
            tape (QuantumScript): The tape of the circuit containing all
                operations.

        Returns:
            List[Operator]: A list of operations consisting only of clifford
                and Pauli-rotation gates.
        """
        operations = []
        for operation in tape.operations:
            if PauliCircuit._is_clifford(
                operation
            ) or PauliCircuit._is_pauli_rotation(operation):
                operations.append(operation)
            elif PauliCircuit._is_skippable(operation):
                continue
            else:
                # TODO: Maybe there is a prettier way to decompose a gate
                tape = QuantumScript([operation])
                decomposed_tape = qml.transforms.decompose(
                    tape, gate_set=PAULI_ROTATION_GATES + CLIFFORD_GATES
                )
                decomposed_ops = decomposed_tape[0][0].operations
                operations.extend(decomposed_ops)

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
    def _evolve_clifford_rotation(
        clifford: Operator, pauli: Operator
    ) -> Tuple[Operator, Operator]:
        """
        This function computes the resulting operations, when switching a
        Cifford gate and a Pauli rotation in the circuit.

        **Example**:
        Consider a circuit consisting of the gate sequence
        ... --- H --- R_z --- ...
        This function computes the evolved Pauli Rotation, and moves the
        clifford (Hadamard) gate to the end:
        ... --- R_x --- H --- ...

        Args:
            clifford (Operator): Clifford gate to move.
            pauli (Operator): Pauli rotation gate to move the clifford past.

        Returns:
            Tuple[Operator, Operator]:
                - Resulting Clifford operator (should be the same as the input)
                - Evolved Pauli rotation operator
        """

        if not any(p_c in clifford.wires for p_c in pauli.wires):
            return pauli, clifford

        gen = pauli.generator()
        param = pauli.parameters[0]

        evolved_gen, _ = PauliCircuit._evolve_clifford_pauli(
            clifford, gen, adjoint_left=False
        )
        qubits = evolved_gen.wires
        evolved_gen = qml.pauli_decompose(evolved_gen.matrix())
        pauli_str, param_factor = PauliCircuit._get_paulistring_from_generator(
            evolved_gen
        )
        pauli = qml.PauliRot(param * param_factor, pauli_str, qubits)

        return pauli, clifford

    @staticmethod
    def _evolve_clifford_pauli(
        clifford: Operator, pauli: Operator, adjoint_left: bool = True
    ) -> Tuple[Operator, Operator]:
        """
        This function computes the resulting operation, when evolving a Pauli
        Operation with a Clifford operation.
        For a Clifford operator C and a Pauli operator P, this functin computes:
            P' = C* P C

        Args:
            clifford (Operator): Clifford gate
            pauli (Operator): Pauli gate
            adjoint_left (bool, optional): If adjoint of the clifford gate is
                applied to the left. If this is set to True C* P C is computed,
                else C P C*. Defaults to True.

        Returns:
            Tuple[Operator, Operator]:
                - Resulting Clifford operator (should be the same as the input)
                - Evolved Pauli operator
        """
        if not any(p_c in clifford.wires for p_c in pauli.wires):
            return pauli, clifford

        if adjoint_left:
            evolved_pauli = (
                qml.adjoint(clifford) @ pauli @ qml.adjoint(clifford)
            )
        else:
            evolved_pauli = clifford @ pauli @ qml.adjoint(clifford)

        return evolved_pauli, clifford

    @staticmethod
    def _evolve_cliffords_list(
        cliffords: List[Operator], pauli: Operator
    ) -> Operator:
        """
        This function evolves a Pauli operation according to a sequence of cliffords.

        Args:
            clifford (Operator): Clifford gate
            pauli (Operator): Pauli gate

        Returns:
            Operator: Evolved Pauli operator
        """
        for clifford in cliffords[::-1]:
            pauli, _ = PauliCircuit._evolve_clifford_pauli(clifford, pauli)
            qubits = pauli.wires
            pauli = qml.pauli_decompose(pauli.matrix(), wire_order=qubits)

        return pauli

    @staticmethod
    def _get_paulistring_from_generator(
        gen: qml_op.LinearCombination,
    ) -> Tuple[str, float]:
        """
        Compute a Paulistring, consisting of "X", "Y", "Z" and "I" from a
        generator.

        Args:
            gen (qml_op.LinearCombination): The generator operation created by
                Pennylane

        Returns:
            Tuple[str, float]:
                - The Paulistring
                - A factor with which to multiply a parameter to the rotation
                  gate.
        """
        factor, term = gen.terms()
        param_factor = -2 * factor  # Rotation is defined as exp(-0.5 theta G)
        pauli_term = term[0] if isinstance(term[0], qml_op.Prod) else [term[0]]
        pauli_str_list = ["I"] * len(pauli_term)
        for p in pauli_term:
            if "Pauli" in p.name:
                q = p.wires[0]
                pauli_str_list[q] = p.name[-1]
        pauli_str = "".join(pauli_str_list)
        return pauli_str, param_factor

    @staticmethod
    def cliffords_in_observable(
        operations: List[Operator], original_obs: List[Operator]
    ) -> List[Operator]:
        """
        Integrates Clifford gates in the observables of the original ansatz.

        Args:
            operations (List[Operator]): Clifford gates
            original_obs (List[Operator]): Original observables from the
                circuit

        Returns:
            List[Operator]: Observables with Clifford operations
        """
        observables = []
        for ob in original_obs:
            clifford_obs = PauliCircuit._evolve_cliffords_list(operations, ob)
            observables.append(clifford_obs)
        return observables


class CoefficientsTreeNode:
    def __init__(
        self,
        parameter: Optional[float],
        parameter_idx: int, # TODO: use or remove
        observable: Operator,
        pauli_rotations: List[Operator],
        is_sine_factor: bool,
        is_cosine_factor: bool,
        left: Optional[CoefficientsTreeNode] = None,
        right: Optional[CoefficientsTreeNode] = None,
    ):
        self.parameter = parameter

        assert not (
            is_sine_factor and is_cosine_factor
        ), "Cannot be sine and cosine at the same time"
        self.is_sine_factor = is_sine_factor
        self.is_cosine_factor = is_sine_factor
        self.parameter_idx = parameter_idx
        self.observable = observable
        self.pauli_rotations = pauli_rotations

        self.left = left
        self.right = right


class FourierTree:

    def __init__(self, quantum_tape: QuantumScript):
        self.parameters = quantum_tape.get_parameters()
        self.observables = quantum_tape.observables
        self.pauli_rotations = quantum_tape.operations
        self.tree_roots = self.build_tree()

    # TODO currently wrapped in transform
    @staticmethod
    def build_coefficients_tree(
        quantum_tape: QuantumScript,
    ) -> tuple[QuantumScriptBatch, PostprocessingFn]:

        tree = FourierTree(quantum_tape)

        def postprocess(res):
            return res[0]

        return [quantum_tape], postprocess

    def build_tree(self) -> List[CoefficientsTreeNode]:
        tree_roots = []
        for obs in self.observables:
            root = self.create_tree_node(obs, self.pauli_rotations)
            tree_roots.append(root)
        return tree_roots

    def create_tree_node(
        self,
        observable: Operator,
        pauli_rotations: List[Operation],
        parameter: Optional[float] = None,
        is_sine: bool = False,
        is_cosine: bool = False,
    ) -> CoefficientsTreeNode:

        # remove commuting paulis
        last_pauli = pauli_rotations[-1] if len(pauli_rotations) > 0 else None
        while last_pauli is not None and qml.is_commuting(
            last_pauli.generator(), observable
        ):
            pauli_rotations = pauli_rotations[:-1]
            last_pauli = (
                pauli_rotations[-1] if len(pauli_rotations) > 0 else None
            )

        if last_pauli is None:  # leaf
            return CoefficientsTreeNode(
                parameter, 0, observable, [], is_sine, is_cosine
            )

        next_pauli_rotations = pauli_rotations[:-1]
        left = self.create_tree_node(
            observable,
            next_pauli_rotations,
            last_pauli.parameters[0],
            is_cosine=True,
        )

        next_observable = self._create_new_observable(
            last_pauli.generator(), observable
        )
        right = self.create_tree_node(
            next_observable,
            next_pauli_rotations,
            last_pauli.parameters[0],
            is_sine=True,
        )

        return CoefficientsTreeNode(
            parameter,
            0,
            observable,
            pauli_rotations,
            is_sine,
            is_cosine,
            left,
            right,
        )

    def _create_new_observable(
        self, pauli: Operator, observable: Operator, adjoint_left: bool = True
    ) -> Operator:
        if adjoint_left:
            obs = qml.adjoint(pauli) @ observable @ qml.adjoint(pauli)
        else:
            obs = pauli @ observable @ qml.adjoint(pauli)

        qubits = obs.wires
        obs = qml.pauli_decompose(obs.matrix(), wire_order=qubits)

        return obs
