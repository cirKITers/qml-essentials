from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.typing import PostprocessingFn
import numpy as np
import math
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

        # No Clifford gates are in the circuit
        if not PauliCircuit._is_clifford(operations[-1]):
            return operations, []

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
        requires_grad = pauli.parameters[0].requires_grad

        evolved_gen, _ = PauliCircuit._evolve_clifford_pauli(
            clifford, gen, adjoint_left=False
        )
        qubits = evolved_gen.wires
        evolved_gen = qml.pauli_decompose(evolved_gen.matrix())
        pauli_str, param_factor = PauliCircuit._get_paulistring_from_generator(
            evolved_gen
        )
        pauli_str, qubits = PauliCircuit._remove_identities_from_paulistr(
            pauli_str, qubits
        )
        pauli = qml.PauliRot(param * param_factor, pauli_str, qubits)
        pauli.parameters[0].requires_grad = requires_grad

        return pauli, clifford

    @staticmethod
    def _remove_identities_from_paulistr(
        pauli_str: str, qubits: List[int]
    ) -> Tuple[str, List[int]]:
        """
        Removes identities from Pauli string and its corresponding qubits.

        Args:
            pauli_str (str): Pauli string
            qubits (List[int]): Corresponding qubit indices.

        Returns:
            Tuple[str, List[int]]:
                - Pauli string without identities
                - Qubits indices without the identities
        """

        reduced_qubits = []
        reduced_pauli_str = ""
        for i, p in enumerate(pauli_str):
            if p != "I":
                reduced_pauli_str += p
                reduced_qubits.append(qubits[i])

        return reduced_pauli_str, reduced_qubits

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
                - Evolved Pauli operator
                - Resulting Clifford operator (should be the same as the input)
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

        pauli = qml.simplify(pauli)

        # remove coefficients
        pauli = (
            pauli.terms()[1][0]
            if isinstance(pauli, (qml_op.Prod, qml_op.LinearCombination))
            else pauli
        )

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
        parameter_idx: Optional[int],
        observable: Operator,
        is_sine_factor: bool,
        is_cosine_factor: bool,
        left: Optional[CoefficientsTreeNode] = None,
        right: Optional[CoefficientsTreeNode] = None,
    ):
        self.parameter_idx = parameter_idx

        assert not (
            is_sine_factor and is_cosine_factor
        ), "Cannot be sine and cosine at the same time"
        self.is_sine_factor = is_sine_factor
        self.is_cosine_factor = is_cosine_factor

        if isinstance(observable, qml_op.SProd):
            term = observable.terms()[0][0]
            observable = observable.terms()[1][0]
        else:
            term = 1.0

        # If the observable does not constist of only Z and I, the
        # expectation (and therefore the constant node term) is zero
        if (
            isinstance(observable, qml_op.Prod)
            and any([isinstance(p, (qml.X, qml.Y)) for p in observable])
            or isinstance(observable, (qml.PauliX, qml.PauliY))
        ):
            self.term = 0.0
        else:
            self.term = term

        self.observable = observable

        self.left = left
        self.right = right

    def evaluate(self, parameters: list[float]) -> float:
        factor = (
            parameters[self.parameter_idx]
            if self.parameter_idx is not None
            else 1.0
        )
        if self.is_sine_factor:
            factor = 1j * np.sin(factor)
        elif self.is_cosine_factor:
            factor = np.cos(factor)
        if not (self.left or self.right):  # leaf
            return factor * self.term

        sum_children = 0.0
        if self.left:
            left = self.left.evaluate(parameters)
            sum_children = sum_children + left
        if self.right:
            right = self.right.evaluate(parameters)
            sum_children = sum_children + right

        return factor * sum_children

    def get_leafs(
        self, sin_list, cos_list, existing_leafs=[]
    ) -> List[TreeLeaf]:

        if self.is_sine_factor:
            sin_list[self.parameter_idx] += 1
        if self.is_cosine_factor:
            cos_list[self.parameter_idx] += 1

        if not (self.left or self.right):  # leaf
            if self.term != 0.0:
                return [TreeLeaf(sin_list, cos_list, self.term)]
            else:
                return []

        if self.left:
            leafs_left = self.left.get_leafs(
                sin_list.copy(), cos_list.copy(), existing_leafs.copy()
            )
        else:
            leafs_left = []

        if self.right:
            leafs_right = self.right.get_leafs(
                sin_list.copy(), cos_list.copy(), existing_leafs.copy()
            )
        else:
            leafs_right = []

        existing_leafs.extend(leafs_left)
        existing_leafs.extend(leafs_right)
        return existing_leafs


@dataclass
class TreeLeaf:
    sin_indices: np.ndarray
    cos_indices: np.ndarray
    term: np.complex128


class FourierTree:

    def __init__(
        self,
        quantum_tape: QuantumScript,
        force_mean: bool = False,
    ):
        self.parameters = [np.squeeze(p) for p in quantum_tape.get_parameters()]
        self.input_indices = [
            i for (i, p) in enumerate(self.parameters) if not p.requires_grad
        ]
        self.observables = quantum_tape.observables
        self.pauli_rotations = quantum_tape.operations
        self.force_mean = force_mean
        self.tree_roots = self.build_tree()
        self.leafs: List[List[TreeLeaf]] = self._get_tree_leafs()

    def build_tree(self) -> List[CoefficientsTreeNode]:
        tree_roots = []
        for obs in self.observables:
            pauli_rotation_indices = np.arange(
                len(self.pauli_rotations), dtype=np.int16
            )
            root = self.create_tree_node(obs, pauli_rotation_indices)
            tree_roots.append(root)
        return tree_roots

    def _get_tree_leafs(self) -> List[List[TreeLeaf]]:
        leafs = []
        for root in self.tree_roots:
            sin_list = np.zeros(len(self.parameters), dtype=np.int32)
            cos_list = np.zeros(len(self.parameters), dtype=np.int32)
            leafs.append(root.get_leafs(sin_list, cos_list, []))
        return leafs

    def get_spectrum(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        parameter_indices = [
            i
            for i in range(len(self.parameters))
            if i not in self.input_indices
        ]

        coeffs = []
        for leafs in self.leafs:
            freq_terms = defaultdict(np.complex128)
            for leaf in leafs:
                leaf_factor, s, c = self._compute_leaf_factors(
                    leaf, parameter_indices
                )

                for a in range(s + 1):
                    for b in range(c + 1):
                        comb = (
                            math.comb(s, a) * math.comb(c, b) * (-1) ** (s - a)
                        )
                        freq_terms[2 * a + 2 * b - s - c] += comb * leaf_factor

            coeffs.append(freq_terms)

        frequencies, coefficients = self._freq_terms_to_coeffs(coeffs)
        return frequencies, coefficients

    def _freq_terms_to_coeffs(
        self, coeffs: List[Dict[int, np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        frequencies = []
        coefficients = []
        if self.force_mean:
            all_freqs = sorted(set([f for c in coeffs for f in c.keys()]))
            coefficients.append(
                np.array(
                    [
                        np.mean([c.get(f, 0.0) for c in coeffs])
                        for f in all_freqs
                    ]
                )
            )
            frequencies.append(np.array(all_freqs))
        else:
            for freq_terms in coeffs:
                freq_terms = dict(sorted(freq_terms.items()))
                frequencies.append(np.array(list(freq_terms.keys())))
                coefficients.append(np.array(list(freq_terms.values())))
        return frequencies, coefficients

    def _compute_leaf_factors(
        self, leaf: TreeLeaf, parameter_indices: List[int]
    ):
        leaf_factor = 1.0
        for i in parameter_indices:
            interm_factor = (
                np.cos(self.parameters[i]) ** leaf.cos_indices[i]
                * (1j * np.sin(self.parameters[i])) ** leaf.sin_indices[i]
            )
            leaf_factor = leaf_factor * interm_factor

        c = np.sum([leaf.cos_indices[k] for k in self.input_indices])
        s = np.sum([leaf.sin_indices[k] for k in self.input_indices])

        leaf_factor = leaf.term * leaf_factor * 0.5 ** (s + c)

        return leaf_factor, s, c

    def evaluate(self) -> np.ndarray:
        results = np.zeros(len(self.tree_roots))
        for i, root in enumerate(self.tree_roots):
            results[i] = np.real_if_close(root.evaluate(self.parameters))

        if self.force_mean:
            return np.mean(results)
        else:
            return results

    def create_tree_node(
        self,
        observable: Operator,
        pauli_rotation_indices: List[int],
        parameter_idx: Optional[int] = None,
        is_sine: bool = False,
        is_cosine: bool = False,
    ) -> CoefficientsTreeNode:

        # remove commuting paulis
        idx = len(pauli_rotation_indices) - 1
        while idx >= 0:
            last_pauli = self.pauli_rotations[pauli_rotation_indices[idx]]
            if not qml.is_commuting(last_pauli.generator(), observable):
                break
            idx -= 1

        if idx < 0:  # leaf
            return CoefficientsTreeNode(
                parameter_idx, observable, is_sine, is_cosine
            )

        next_pauli_rotation_indices = pauli_rotation_indices[:idx]
        last_pauli_idx = pauli_rotation_indices[idx]
        last_pauli = self.pauli_rotations[last_pauli_idx]

        left = self.create_tree_node(
            observable,
            next_pauli_rotation_indices,
            last_pauli_idx,
            is_cosine=True,
        )

        next_observable = self._create_new_observable(
            last_pauli.generator(), observable
        )
        right = self.create_tree_node(
            next_observable,
            next_pauli_rotation_indices,
            last_pauli_idx,
            is_sine=True,
        )

        return CoefficientsTreeNode(
            parameter_idx,
            observable,
            is_sine,
            is_cosine,
            left,
            right,
        )

    def _create_new_observable(
        self, pauli: Operator, observable: Operator
    ) -> Operator:

        pauli = pauli[0] / pauli.coeffs[0]  # ignore coefficients of generator
        obs = pauli @ observable
        obs = qml.simplify(obs)

        return obs
