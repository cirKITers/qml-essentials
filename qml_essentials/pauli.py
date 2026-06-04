"""Pauli-Clifford circuit transform for the Fourier-tree algorithm.

This module hosts :class:`PauliCircuit` (and the lightweight :class:`PauliTape`),
which transpile a circuit into the *canonical Pauli-Clifford normal form* used by
the Nemkov et al. algorithm: all Clifford gates are commuted to the end and
absorbed into the observable, leaving a sequence of Pauli rotations.

The Clifford conjugation that drives this transform is done **symbolically** via
:class:`~qml_essentials.operations.PauliWord` (stabilizer-tableau updates, O(n)),
replacing the previous matrix-based path
(:func:`~qml_essentials.operations.evolve_pauli_with_clifford` +
:func:`~qml_essentials.operations.pauli_decompose`, which was O(2^n)+O(4^n)).
"""

from __future__ import annotations

from functools import reduce
from typing import List, Optional, Tuple
from collections import defaultdict

import numpy as np
import jax.numpy as jnp

from qml_essentials.operations import (
    Operation,
    PauliWord,
    Hermitian,
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
    _cdtype,
)


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
        n_qubits: Optional[int] = None,
    ) -> PauliTape:
        """
        Transforms a list of operations into a Pauli-Clifford circuit.

        Args:
            tape: List of operations recorded from the circuit.
            observables: List of observable operations.  If ``None``, defaults
                to an empty list.
            n_qubits: Total number of qubits.  Inferred from the maximum wire
                index if ``None``.

        Returns:
            PauliTape:
                A new tape containing the operations of the Pauli-Clifford
                circuit and the (possibly Clifford-evolved) observables.
        """
        if observables is None:
            observables = []

        operations = PauliCircuit.get_clifford_pauli_gates(tape)

        if n_qubits is None:
            n_qubits = PauliCircuit._infer_n_qubits(operations, observables)

        pauli_gates, final_cliffords = PauliCircuit.commute_all_cliffords_to_the_end(
            operations, n_qubits
        )

        observables = PauliCircuit.cliffords_in_observable(
            final_cliffords, observables, n_qubits
        )

        return PauliTape(operations=pauli_gates, observables=observables)

    @staticmethod
    def _infer_n_qubits(
        operations: List[Operation], observables: List[Operation]
    ) -> int:
        """Infer the register size from the maximum wire index used."""
        max_wire = -1
        for op in list(operations) + list(observables):
            if op.wires:
                max_wire = max(max_wire, max(op.wires))
        return max_wire + 1

    @staticmethod
    def commute_all_cliffords_to_the_end(
        operations: List[Operation],
        n_qubits: int,
    ) -> Tuple[List[Operation], List[Operation]]:
        """
        This function moves all clifford gates to the end of the circuit,
        accounting for commutation rules.

        Args:
            operations (List[Operation]): The operations in the tape of the
                circuit
            n_qubits (int): Total number of qubits.

        Returns:
            Tuple[List[Operation], List[Operation]]:
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
                    operations[j], operations[j + 1], n_qubits
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
        """Whether an operation can be ignored (currently only barriers)."""
        return isinstance(operation, PauliCircuit.SKIPPABLE_OPERATIONS)

    @staticmethod
    def _is_clifford(operation: Operation) -> bool:
        """Whether an operation is a Clifford gate."""
        return isinstance(operation, PauliCircuit.CLIFFORD_GATES)

    @staticmethod
    def _is_pauli_rotation(operation: Operation) -> bool:
        """Whether an operation is a Pauli rotation gate."""
        return isinstance(operation, PauliCircuit.PAULI_ROTATION_GATES)

    @staticmethod
    def _evolve_clifford_rotation(
        clifford: Operation, pauli: Operation, n_qubits: int
    ) -> Tuple[Operation, Operation]:
        """
        Compute the resulting operations when switching a Clifford gate and a
        Pauli rotation in the circuit, i.e. move the Clifford past the rotation:

        ``... C R_P(phi) ...  ->  ... R_{C P C^dagger}(phi) C ...``

        The evolved Pauli rotation is obtained by **symbolic** Clifford
        conjugation of the rotation generator (no matrices).

        Args:
            clifford (Operation): Clifford gate to move.
            pauli (Operation): Pauli rotation gate to move the clifford past.
            n_qubits (int): Total number of qubits.

        Returns:
            Tuple[Operation, Operation]:
                - Evolved Pauli rotation operator
                - The (unchanged) Clifford operator
        """
        if not any(p_c in clifford.wires for p_c in pauli.wires):
            return pauli, clifford

        param = pauli.parameters[0]

        gen_word = PauliWord.from_operation(pauli, n_qubits)
        evolved = gen_word.conjugate_by_clifford(clifford, adjoint_left=False)
        bare, phase = evolved.to_pauli_string_and_phase()

        # Clifford conjugation of a (Hermitian) Pauli generator yields +-1.
        param_factor = float(np.real(phase))

        pauli_str, qubits = PauliCircuit._remove_identities_from_paulistr(
            bare, list(range(n_qubits))
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
    def cliffords_in_observable(
        operations: List[Operation],
        original_obs: List[Operation],
        n_qubits: int,
    ) -> List[Operation]:
        """
        Integrates Clifford gates into the observables of the original ansatz,
        by symbolically conjugating each observable through the final Clifford
        sequence (``O -> C^dagger O C`` for each Clifford, applied in reverse).

        Args:
            operations (List[Operation]): Clifford gates
            original_obs (List[Operation]): Original observables from the
                circuit
            n_qubits (int): Total number of qubits.

        Returns:
            List[Operation]: Observables with Clifford operations absorbed.
                Each carries a cached symbolic ``_pauli_word`` for the
                Fourier-tree algorithm and a matrix for simulation.
        """
        observables = []
        for ob in original_obs:
            word = PauliWord.from_operation(ob, n_qubits)
            for clifford in operations[::-1]:
                word = word.conjugate_by_clifford(clifford, adjoint_left=True)
            observables.append(PauliCircuit._pauli_operation_from_word(word))
        return observables

    @staticmethod
    def _pauli_operation_from_word(word: PauliWord) -> Operation:
        """Build an observable :class:`Operation` from a symbolic Pauli word.

        The returned operation carries both a dense ``matrix`` (for the
        statevector simulator) and a cached ``_pauli_word`` / ``_pauli_label``
        (for symbolic consumers such as the Fourier tree).
        """
        bare, phase = word.to_pauli_string_and_phase()
        reduced_str, reduced_wires = PauliCircuit._remove_identities_from_paulistr(
            bare, list(range(word.n_qubits))
        )

        if not reduced_str:
            obs = Hermitian(
                matrix=phase * jnp.eye(2, dtype=_cdtype()), wires=[0], record=False
            )
            obs._pauli_label = "I"
        else:
            mats = [PauliRot._PAULI_MAP[c] for c in reduced_str]
            mat = reduce(jnp.kron, mats) * phase
            obs = Hermitian(matrix=mat, wires=reduced_wires, record=False)
            obs._pauli_label = reduced_str

        obs._pauli_word = word
        return obs
