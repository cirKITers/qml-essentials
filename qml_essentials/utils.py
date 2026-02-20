from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
from qml_essentials.operations import (
    Operation,
    PauliX,
    PauliY,
    PauliZ,
    H,
    S,
    CX,
    RX,
    RY,
    RZ,
    PauliRot,
    Barrier,
    evolve_pauli_with_clifford,
    pauli_decompose,
    pauli_string_from_operation,
)
from fractions import Fraction
from itertools import cycle
from scipy.linalg import logm


def safe_random_split(random_key: jax.random.PRNGKey, *args, **kwargs):
    if random_key is None:
        return None, None
    else:
        return jax.random.split(random_key, *args, **kwargs)


def logm_v(A: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Compute the logarithm of a matrix. If the provided matrix has an additional
    batch dimension, the logarithm of each matrix is computed.

    Args:
        A (jnp.ndarray): The (potentially batched) matrices of which to compute
        the logarithm.

    Returns:
        jnp.ndarray: The log matrices
    """
    # TODO: check warnings
    if len(A.shape) == 2:
        return logm(A, **kwargs)
    elif len(A.shape) == 3:
        AV = jnp.zeros(A.shape, dtype=jnp.complex128)
        for i in range(A.shape[0]):
            AV = AV.at[i].set(logm(A[i], **kwargs))
        return AV
    else:
        raise NotImplementedError("Unsupported shape of input matrix")


class PauliTape:
    """Simple tape wrapper with ``operations``, ``observables``, and
    ``get_parameters`` — replacing PennyLane's ``QuantumScript`` for the
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
                # Rot(φ, θ, ω) = RZ(ω) @ RY(θ) @ RZ(φ)
                w = operation.wires[0]
                operations.append(RZ(operation.phi, wires=w))
                operations.append(RY(operation.theta, wires=w))
                operations.append(RZ(operation.omega, wires=w))
            elif isinstance(operation, CRZ):
                # CRZ(θ, [c,t]) = RZ(θ/2, t) · CNOT(c,t) · RZ(-θ/2, t) · CNOT(c,t)
                c, t = operation.wires
                theta = operation.theta
                operations.append(RZ(theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RZ(-theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
            elif isinstance(operation, CRX):
                # CRX(θ, [c,t]) = H(t) · CRZ(θ, [c,t]) · H(t)
                #               = H(t) · RZ(θ/2,t) · CX(c,t) · RZ(-θ/2,t) · CX(c,t) · H(t)
                c, t = operation.wires
                theta = operation.theta
                operations.append(H(wires=t))
                operations.append(RZ(theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RZ(-theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(H(wires=t))
            elif isinstance(operation, CRY):
                # CRY(θ, [c,t]) = RX(-π/2, t) · CRZ(θ, [c,t]) · RX(π/2, t)
                #               = RX(-π/2,t) · RZ(θ/2,t) · CX(c,t) · RZ(-θ/2,t) · CX(c,t) · RX(π/2,t)
                c, t = operation.wires
                theta = operation.theta
                operations.append(RX(-jnp.pi / 2, wires=t))
                operations.append(RZ(theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RZ(-theta / 2, wires=t))
                operations.append(CX(wires=[c, t]))
                operations.append(RX(jnp.pi / 2, wires=t))
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

        **Example**:
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
            pauli_str, qubits
        )
        new_pauli = PauliRot(param * param_factor, pauli_str, qubits)

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


class QuanTikz:
    class TikzFigure:
        def __init__(self, quantikz_str: str):
            self.quantikz_str = quantikz_str

        def __repr__(self):
            return self.quantikz_str

        def __str__(self):
            return self.quantikz_str

        def wrap_figure(self):
            """
            Wraps the quantikz string in a LaTeX figure environment.

            Returns:
                str: A formatted LaTeX string representing the TikZ figure containing
                the quantum circuit diagram.
            """
            return f"""
\\begin{{figure}}
    \\centering
    \\begin{{tikzpicture}}
        \\node[scale=0.85] {{
            \\begin{{quantikz}}
                {self.quantikz_str}
            \\end{{quantikz}}
        }};
    \\end{{tikzpicture}}
\\end{{figure}}"""

        def export(self, destination: str, full_document=False, mode="w") -> None:
            """
            Export a LaTeX document with a quantum circuit in stick notation.

            Parameters
            ----------
            quantikz_strs : str or list[str]
                LaTeX string for the quantum circuit or a list of LaTeX strings.
            destination : str
                Path to the destination file.
            """
            if full_document:
                latex_code = f"""
\\documentclass{{article}}
\\usepackage{{quantikz}}
\\usepackage{{tikz}}
\\usetikzlibrary{{quantikz2}}
\\usepackage{{quantikz}}
\\usepackage[a3paper, landscape, margin=0.5cm]{{geometry}}
\\begin{{document}}
{self.wrap_figure()}
\\end{{document}}"""
            else:
                latex_code = self.quantikz_str + "\n"

            with open(destination, mode) as f:
                f.write(latex_code)

    @staticmethod
    def ground_state() -> str:
        """
        Generate the LaTeX representation of the |0⟩ ground state in stick notation.

        Returns
        -------
        str
            LaTeX string for the |0⟩ state.
        """
        return "\\lstick{\\ket{0}}"

    @staticmethod
    def measure(op):
        if len(op.wires) > 1:
            raise NotImplementedError("Multi-wire measurements are not supported yet")
        else:
            return "\\meter{}"

    @staticmethod
    def search_pi_fraction(w, op_name):
        w_pi = Fraction(w / jnp.pi).limit_denominator(100)
        # Not a small nice Fraction
        if w_pi.denominator > 12:
            return f"\\gate{{{op_name}({w:.2f})}}"
        # Pi
        elif w_pi.denominator == 1 and w_pi.numerator == 1:
            return f"\\gate{{{op_name}(\\pi)}}"
        # 0
        elif w_pi.numerator == 0:
            return f"\\gate{{{op_name}(0)}}"
        # Multiple of Pi
        elif w_pi.denominator == 1:
            return f"\\gate{{{op_name}({w_pi.numerator}\\pi)}}"
        # Nice Fraction of pi
        elif w_pi.numerator == 1:
            return (
                f"\\gate{{{op_name}\\left("
                f"\\frac{{\\pi}}{{{w_pi.denominator}}}\\right)}}"
            )
        # Small nice Fraction
        else:
            return (
                f"\\gate{{{op_name}\\left("
                f"\\frac{{{w_pi.numerator}\\pi}}{{{w_pi.denominator}}}"
                f"\\right)}}"
            )

    @staticmethod
    def gate(op, index=None, gate_values=False, inputs_symbols="x") -> str:
        """
        Generate LaTeX for a quantum gate in stick notation.

        Parameters
        ----------
        op : Operation
            The quantum gate to represent.
        index : int, optional
            Gate index in the circuit.
        gate_values : bool, optional
            Include gate values in the representation.
        inputs_symbols : str, optional
            Symbols for the inputs in the representation.

        Returns
        -------
        str
            LaTeX string for the gate.
        """
        op_name = op.name
        match op.name:
            case "H":
                op_name = "H"
            case "RX" | "RY" | "RZ":
                pass
            case "Rot":
                op_name = "R"

        if gate_values and len(op.parameters) > 0:
            w = float(op.parameters[0].item())
            return QuanTikz.search_pi_fraction(w, op_name)
        else:
            # Is gate with parameter
            if op.parameters == [] or op.parameters[0].shape == ():
                if index is None:
                    return f"\\gate{{{op_name}}}"
                else:
                    return f"\\gate{{{op_name}(\\theta_{{{index}}})}}"
            # Is gate with input
            elif op.parameters[0].shape == (1,):
                return f"\\gate{{{op_name}({inputs_symbols})}}"

    @staticmethod
    def cgate(op, index=None, gate_values=False, inputs_symbols="x") -> Tuple[str, str]:
        """
        Generate LaTeX for a controlled quantum gate in stick notation.

        Parameters
        ----------
        op : Operation
            The quantum gate operation to represent.
        index : int, optional
            Gate index in the circuit.
        gate_values : bool, optional
            Include gate values in the representation.
        inputs_symbols : str, optional
            Symbols for the inputs in the representation.

        Returns
        -------
        Tuple[str, str]
            - LaTeX string for the control gate
            - LaTeX string for the target gate
        """
        match op.name:
            case "CRX" | "CRY" | "CRZ" | "CX" | "CY" | "CZ":
                op_name = op.name[1:]
            case _:
                pass
        targ = "\\targ{}"
        if op.name in ["CRX", "CRY", "CRZ"]:
            if gate_values and len(op.parameters) > 0:
                w = float(op.parameters[0].item())
                targ = QuanTikz.search_pi_fraction(w, op_name)
            else:
                # Is gate with parameter
                if op.parameters[0].shape == ():
                    if index is None:
                        targ = f"\\gate{{{op_name}}}"
                    else:
                        targ = f"\\gate{{{op_name}(\\theta_{{{index}}})}}"
                # Is gate with input
                elif op.parameters[0].shape == (1,):
                    targ = f"\\gate{{{op_name}({inputs_symbols})}}"
        elif op.name in ["CX", "CY", "CZ"]:
            targ = "\\control{}"

        distance = op.wires[1] - op.wires[0]
        return f"\\ctrl{{{distance}}}", targ

    @staticmethod
    def barrier(op) -> str:
        """
        Generate LaTeX for a barrier in stick notation.

        Parameters
        ----------
        op : Operation
            The barrier operation to represent.

        Returns
        -------
        str
            LaTeX string for the barrier.
        """
        return (
            "\\slice[style={{draw=black, solid, double distance=2pt, "
            "line width=0.5pt}}]{{}}"
        )

    @staticmethod
    def _build_tikz_circuit(
        tape: List[Operation],
        n_qubits: int,
        gate_values=False,
        inputs_symbols="x",
    ):
        """
        Builds a LaTeX representation of a quantum circuit in TikZ format.

        This static method constructs a TikZ circuit diagram from a given list
        of operations.  It processes gates, controlled gates, and barriers.
        The resulting structure is a list of LaTeX strings, each representing a
        wire in the circuit.

        Parameters
        ----------
        tape : List[Operation]
            The list of operations in the circuit.
        n_qubits : int
            The number of qubits in the circuit.
        gate_values : bool, optional
            If True, include gate parameter values in the representation.
        inputs_symbols : str, optional
            Symbols to represent the inputs in the circuit.

        Returns
        -------
        circuit_tikz : list of list of str
            A nested list where each inner list contains LaTeX strings representing
            the operations on a single wire of the circuit.
        """

        circuit_tikz = [[QuanTikz.ground_state()] for _ in range(n_qubits)]

        index = iter(range(len(tape)))
        for op in tape:
            # catch barriers
            if op.name == "Barrier":
                # get the maximum length of all wires
                max_len = max(len(circuit_tikz[cw]) for cw in range(len(circuit_tikz)))

                # extend the wires by the number of missing operations
                for ow in range(len(circuit_tikz)):
                    circuit_tikz[ow].extend(
                        "" for _ in range(max_len - len(circuit_tikz[ow]))
                    )

                circuit_tikz[op.wires[0]][-1] += QuanTikz.barrier(op)
            # single qubit gate?
            elif len(op.wires) == 1:
                # build and append standard gate
                circuit_tikz[op.wires[0]].append(
                    QuanTikz.gate(
                        op,
                        index=next(index),
                        gate_values=gate_values,
                        inputs_symbols=next(inputs_symbols),
                    )
                )
            # controlled gate?
            elif len(op.wires) == 2:
                # build the controlled gate
                if op.name in ["CRX", "CRY", "CRZ"]:
                    ctrl, targ = QuanTikz.cgate(
                        op,
                        index=next(index),
                        gate_values=gate_values,
                        inputs_symbols=next(inputs_symbols),
                    )
                else:
                    ctrl, targ = QuanTikz.cgate(op)

                # get the wires that this cgate spans over
                crossing_wires = [i for i in range(min(op.wires), max(op.wires) + 1)]
                # get the maximum length of all operations currently on this wire
                max_len = max([len(circuit_tikz[cw]) for cw in crossing_wires])

                # extend the affected wires by the number of missing operations
                for ow in range(min(op.wires), max(op.wires) + 1):
                    circuit_tikz[ow].extend(
                        "" for _ in range(max_len - len(circuit_tikz[ow]))
                    )

                # finally append the cgate operation
                circuit_tikz[op.wires[0]].append(ctrl)
                circuit_tikz[op.wires[1]].append(targ)

                # extend the non-affected wires by the number of missing operations
                non_gate_wires = [w for w in crossing_wires if w not in op.wires]
                for cw in non_gate_wires:
                    circuit_tikz[cw].append("")
            else:
                raise NotImplementedError(">2-wire gates are not supported yet")

        return circuit_tikz

    @staticmethod
    def build(
        script,
        params,
        inputs,
        enc_params=None,
        gate_values=False,
        inputs_symbols="x",
    ) -> str:
        """
        Generate LaTeX for a quantum circuit in stick notation.

        Parameters
        ----------
        script : QuantumScript
            A yaqsi QuantumScript wrapping the circuit function.
        params : array
            Weight parameters for the circuit.
        inputs : array
            Inputs for the circuit.
        enc_params : array
            Encoding weight parameters for the circuit.
        gate_values : bool, optional
            Toggle for gate values or theta variables in the representation.
        inputs_symbols : str, optional
            Symbols for the inputs in the representation.

        Returns
        -------
        str
            LaTeX string for the circuit.
        """
        if enc_params is not None:
            tape = script._record(params=params, inputs=inputs, enc_params=enc_params)
        else:
            tape = script._record(params=params, inputs=inputs)

        # Infer n_qubits from the tape
        n_qubits = max((max(op.wires) + 1 for op in tape if op.wires), default=1)

        if isinstance(inputs_symbols, str) and inputs.size > 1:
            inputs_symbols = cycle(
                [f"{inputs_symbols}_{i}" for i in range(inputs.size)]
            )
        elif isinstance(inputs_symbols, list):
            assert (
                len(inputs_symbols) == inputs.size
            ), f"The number of input symbols {len(inputs_symbols)} \
                must match the number of inputs {inputs.size}."
            inputs_symbols = cycle(inputs_symbols)
        else:
            inputs_symbols = cycle([inputs_symbols])

        circuit_tikz = QuanTikz._build_tikz_circuit(
            tape, n_qubits, gate_values=gate_values, inputs_symbols=inputs_symbols
        )
        quantikz_str = ""

        # get the maximum length of all wires
        max_len = max(len(circuit_tikz[cw]) for cw in range(len(circuit_tikz)))

        # extend the wires by the number of missing operations
        for ow in range(len(circuit_tikz)):
            circuit_tikz[ow].extend("" for _ in range(max_len - len(circuit_tikz[ow])))

        for wire_idx, wire_ops in enumerate(circuit_tikz):
            for op_idx, op in enumerate(wire_ops):
                # if not last operation on wire
                if op_idx < len(wire_ops) - 1:
                    quantikz_str += f"{op} & "
                else:
                    quantikz_str += f"{op}"
                    # if not last wire
                    if wire_idx < len(circuit_tikz) - 1:
                        quantikz_str += " \\\\\n"

        return QuanTikz.TikzFigure(quantikz_str)
