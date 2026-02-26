from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
from qml_essentials.operations import (
    _cdtype,
    Operation,
    PauliX,
    PauliY,
    PauliZ,
    H,
    S,
    CX,
    CZ,
    RX,
    RY,
    RZ,
    PauliRot,
    Barrier,
    evolve_pauli_with_clifford,
    pauli_decompose,
    pauli_string_from_operation,
)
from scipy.linalg import logm
from collections import defaultdict


def safe_random_split(random_key: jax.random.PRNGKey, *args, **kwargs):
    if random_key is None:
        return None, None
    else:
        return jax.random.split(random_key, *args, **kwargs)


class PauliTape:
    """Simple tape wrapper with ``operations``, ``observables``, and
    ``get_parameters`` — replacing PennyLane's ``Script`` for the
    Fourier-tree algorithm.
    """

    def __init__(
        self,
        operations: List[Operation],
        observables: List[Operation],
    ) -> None:
        self.operations = operations
        self.observables = observables

    def get_parameters(self) -> list:
        """Return the list of all parameter values from the operations."""
        params = []
        for op in self.operations:
            params.extend(op.parameters)
        return params

    def get_input_indices(self) -> list:
        indices = defaultdict(list)
        all_indices = []
        ops_w_params = [o for o in self.operations if len(o.parameters) > 0]
        for i, op in enumerate(ops_w_params):
            if op.input_idx >= 0:
                indices[op.input_idx].append(i)
                all_indices.append(i)
        return indices, all_indices


class PauliCircuit:
    """
    Wrapper for Pauli-Clifford Circuits described by Nemkov et al.
    (https://doi.org/10.1103/PhysRevA.108.032406). The code is inspired
    by the corresponding implementation: https://github.com/idnm/FourierVQA.

    A Pauli Circuit only consists of parameterised Pauli-rotations and Clifford
    gates, which is the default for the most common VQCs.
    """

    CLIFFORD_GATES = (
        PauliX,
        PauliY,
        PauliZ,
        H,
        S,
        CX,
    )

    PAULI_ROTATION_GATES = (
        RX,
        RY,
        RZ,
        PauliRot,
    )

    SKIPPABLE_OPERATIONS = (Barrier,)

    @staticmethod
    def from_parameterised_circuit(
        tape: List[Operation],
        observables: Optional[List[Operation]] = None,
    ) -> PauliTape:
        """
        Transforms a list of operations into a Pauli-Clifford circuit.

        Args:
            tape: List of operations recorded from the circuit.
            observables: List of observable operations.  If ``None``, defaults
                to ``[PauliZ(0)]``.

        Returns:
            PauliTape:
                A new tape containing the operations of the Pauli-Clifford
                circuit and the (possibly Clifford-evolved) observables.
        """
        if observables is None:
            observables = []

        operations = PauliCircuit.get_clifford_pauli_gates(tape)

        pauli_gates, final_cliffords = PauliCircuit.commute_all_cliffords_to_the_end(
            operations
        )

        observables = PauliCircuit.cliffords_in_observable(final_cliffords, observables)

        return PauliTape(operations=pauli_gates, observables=observables)

    @staticmethod
    def commute_all_cliffords_to_the_end(
        operations: List[Operation],
    ) -> Tuple[List[Operation], List[Operation]]:
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
                j + 1 < len(operations)  # Clifford has not alredy reached the end
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
    def get_clifford_pauli_gates(tape: List[Operation]) -> List[Operation]:
        """
        This function decomposes all gates in the circuit to clifford and
        pauli-rotation gates.

        Args:
            tape: List of operations recorded from the circuit.

        Returns:
            List[Operation]: A list of operations consisting only of clifford
                and Pauli-rotation gates.
        """
        from qml_essentials.operations import Rot, CRX, CRY, CRZ

        operations = []
        for operation in tape:
            if PauliCircuit._is_clifford(operation) or PauliCircuit._is_pauli_rotation(
                operation
            ):
                operations.append(operation)
            elif PauliCircuit._is_skippable(operation):
                continue
            elif isinstance(operation, Rot):
                w = operation.wires[0]
                operations.append(RZ(operation.phi, wires=w))
                operations.append(RY(operation.theta, wires=w))
                operations.append(RZ(operation.omega, wires=w))
            elif isinstance(operation, CRZ):
                c, t = operation.wires
                theta = operation.theta
                operations.append(RZ(theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RZ(-theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
            elif isinstance(operation, CRX):
                c, t = operation.wires
                theta = operation.theta
                operations.append(H(wires=t))
                operations.append(RZ(theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RZ(-theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(H(wires=t))
            elif isinstance(operation, CRY):
                c, t = operation.wires
                theta = operation.theta
                operations.append(RX(-jnp.pi / 2, wires=t))
                operations.append(RZ(theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RZ(-theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RX(jnp.pi / 2, wires=t))
            elif isinstance(operation, CZ):
                c, t = operation.wires
                operations.append(H(wires=c))
                operations.append(CX(wires=[c, t]))
                operations.append(H(wires=c))
            else:
                raise NotImplementedError(
                    f"Gate {operation.name} cannot be decomposed into "
                    "Pauli rotations and Clifford gates. Consider using a "
                    "circuit ansatz that only uses RX, RY, RZ, PauliRot, "
                    "Rot, and standard Clifford gates."
                )

        return operations

    @staticmethod
    def _is_skippable(operation: Operation) -> bool:
        """
        Determines is an operator can be ignored when building the Pauli
        Clifford circuit. Currently this only contains barriers.

        Args:
            operation (Operation): Gate operation

        Returns:
            bool: Whether the operation can be skipped.
        """
        return isinstance(operation, PauliCircuit.SKIPPABLE_OPERATIONS)

    @staticmethod
    def _is_clifford(operation: Operation) -> bool:
        """
        Determines is an operator is a Clifford gate.

        Args:
            operation (Operation): Gate operation

        Returns:
            bool: Whether the operation is Clifford.
        """
        return isinstance(operation, PauliCircuit.CLIFFORD_GATES)

    @staticmethod
    def _is_pauli_rotation(operation: Operation) -> bool:
        """
        Determines is an operator is a Pauli rotation gate.

        Args:
            operation (Operation): Gate operation

        Returns:
            bool: Whether the operation is a Pauli operation.
        """
        return isinstance(operation, PauliCircuit.PAULI_ROTATION_GATES)

    @staticmethod
    def _evolve_clifford_rotation(
        clifford: Operation, pauli: Operation
    ) -> Tuple[Operation, Operation]:
        """
        This function computes the resulting operations, when switching a
        Clifford gate and a Pauli rotation in the circuit.

        Example:
        Consider a circuit consisting of the gate sequence
        ... --- H --- R_z --- ...
        This function computes the evolved Pauli Rotation, and moves the
        clifford (Hadamard) gate to the end:
        ... --- R_x --- H --- ...

        Args:
            clifford (Operation): Clifford gate to move.
            pauli (Operation): Pauli rotation gate to move the clifford past.

        Returns:
            Tuple[Operation, Operation]:
                - Evolved Pauli rotation operator
                - Resulting Clifford operator (should be the same as the input)
        """

        if not any(p_c in clifford.wires for p_c in pauli.wires):
            return pauli, clifford

        gen = pauli.generator()
        param = pauli.parameters[0]

        evolved_gen = evolve_pauli_with_clifford(clifford, gen, adjoint_left=False)
        qubits = evolved_gen.wires
        _coeff, evolved_pauli_op = pauli_decompose(
            evolved_gen.matrix, wire_order=qubits
        )

        pauli_str = pauli_string_from_operation(evolved_pauli_op)
        # The coefficient from the decomposition determines if there's a sign
        # flip (param_factor).  For Pauli evolution the coefficient is ±1.
        param_factor = float(jnp.real(_coeff))

        pauli_str, qubits = PauliCircuit._remove_identities_from_paulistr(
            pauli_str, evolved_pauli_op.wires
        )
        new_pauli = PauliRot(param * param_factor, pauli_str, qubits)

        if pauli.input_idx >= 0:
            new_pauli.input_idx = pauli.input_idx

        return new_pauli, clifford

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
        clifford: Operation, pauli: Operation, adjoint_left: bool = True
    ) -> Tuple[Operation, Operation]:
        """
        This function computes the resulting operation, when evolving a Pauli
        Operation with a Clifford operation.
        For a Clifford operator C and a Pauli operator P, this function computes:
            P' = C† P C   (adjoint_left=True)
            P' = C P C†   (adjoint_left=False)

        Args:
            clifford (Operation): Clifford gate
            pauli (Operation): Pauli gate
            adjoint_left (bool, optional): If adjoint of the clifford gate is
                applied to the left. Defaults to True.

        Returns:
            Tuple[Operation, Operation]:
                - Evolved Pauli operator
                - Resulting Clifford operator (same as input)
        """
        if not any(p_c in clifford.wires for p_c in pauli.wires):
            return pauli, clifford

        evolved = evolve_pauli_with_clifford(clifford, pauli, adjoint_left=adjoint_left)
        return evolved, clifford

    @staticmethod
    def _evolve_cliffords_list(
        cliffords: List[Operation], pauli: Operation
    ) -> Operation:
        """
        This function evolves a Pauli operation according to a sequence of
        cliffords.

        Args:
            cliffords (List[Operation]): Clifford gates
            pauli (Operation): Pauli gate

        Returns:
            Operation: Evolved Pauli operator
        """
        for clifford in cliffords[::-1]:
            pauli, _ = PauliCircuit._evolve_clifford_pauli(clifford, pauli)
            qubits = pauli.wires
            _coeff, pauli = pauli_decompose(pauli.matrix, wire_order=qubits)

        return pauli

    @staticmethod
    def cliffords_in_observable(
        operations: List[Operation], original_obs: List[Operation]
    ) -> List[Operation]:
        """
        Integrates Clifford gates in the observables of the original ansatz.

        Args:
            operations (List[Operation]): Clifford gates
            original_obs (List[Operation]): Original observables from the
                circuit

        Returns:
            List[Operation]: Observables with Clifford operations
        """
        observables = []
        for ob in original_obs:
            clifford_obs = PauliCircuit._evolve_cliffords_list(operations, ob)
            observables.append(clifford_obs)
        return observables
