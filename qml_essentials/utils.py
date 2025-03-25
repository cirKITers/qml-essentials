from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.typing import PostprocessingFn
import pennylane.numpy as pnp
import pennylane.ops.op_math as qml_op
from pennylane.drawer import drawable_layers, tape_text
from fractions import Fraction

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

        pauli_gates, final_cliffords = PauliCircuit.commute_all_cliffords_to_the_end(
            operations
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
            if PauliCircuit._is_clifford(operation) or PauliCircuit._is_pauli_rotation(
                operation
            ):
                operations.append(operation)
            elif PauliCircuit._is_skippable(operation):
                continue
            else:
                # TODO: Maybe there is a prettier way to decompose a gate
                # We currently can not handle parametrised input gates, that
                # are not plain pauli rotations
                tape = QuantumScript([operation])
                decomposed_tape = qml.transforms.decompose(
                    tape, gate_set=PAULI_ROTATION_GATES + CLIFFORD_GATES
                )
                decomposed_ops = decomposed_tape[0][0].operations
                decomposed_ops = [
                    (
                        op
                        if PauliCircuit._is_clifford(op)
                        else op.__class__(pnp.tensor(op.parameters), op.wires)
                    )
                    for op in decomposed_ops
                ]
                operations.extend(decomposed_ops)

        return operations

    @staticmethod
    def _is_skippable(operation: Operator) -> bool:
        """
        Determines is an operator can be ignored when building the Pauli
        Clifford circuit. Currently this only contains barriers.

        Args:
            operation (Operator): Gate operation

        Returns:
            bool: Whether the operation can be skipped.
        """
        return isinstance(operation, SKIPPABLE_OPERATIONS)

    @staticmethod
    def _is_clifford(operation: Operator) -> bool:
        """
        Determines is an operator is a Clifford gate.

        Args:
            operation (Operator): Gate operation

        Returns:
            bool: Whether the operation is Clifford.
        """
        return isinstance(operation, CLIFFORD_GATES)

    @staticmethod
    def _is_pauli_rotation(operation: Operator) -> bool:
        """
        Determines is an operator is a Pauli rotation gate.

        Args:
            operation (Operator): Gate operation

        Returns:
            bool: Whether the operation is a Pauli operation.
        """
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
        requires_grad = param.requires_grad if isinstance(param, pnp.tensor) else False
        param = pnp.tensor(param)

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
            evolved_pauli = qml.adjoint(clifford) @ pauli @ qml.adjoint(clifford)
        else:
            evolved_pauli = clifford @ pauli @ qml.adjoint(clifford)

        return evolved_pauli, clifford

    @staticmethod
    def _evolve_cliffords_list(cliffords: List[Operator], pauli: Operator) -> Operator:
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
        w_pi = Fraction(w / np.pi).limit_denominator(100)
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
    def gate(op, index=None, gate_values=False) -> str:
        """
        Generate LaTeX for a quantum gate in stick notation.

        Parameters
        ----------
        op : qml.Operation
            The quantum gate to represent.
        index : int, optional
            Gate index in the circuit.
        gate_values : bool, optional
            Include gate values in the representation.

        Returns
        -------
        str
            LaTeX string for the gate.
        """
        op_name = op.name
        match op.name:
            case "Hadamard":
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
                return f"\\gate{{{op_name}(x)}}"

    @staticmethod
    def cgate(op, index=None, gate_values=False) -> Tuple[str, str]:
        """
        Generate LaTeX for a controlled quantum gate in stick notation.

        Parameters
        ----------
        op : qml.Operation
            The quantum gate operation to represent.
        index : int, optional
            Gate index in the circuit.
        gate_values : bool, optional
            Include gate values in the representation.

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
                    targ = f"\\gate{{{op_name}(x)}}"
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
        op : qml.Operation
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
    def build(circuit: qml.QNode, params, inputs, gate_values=False) -> str:
        """
        Generate LaTeX for a quantum circuit in stick notation.

        Parameters
        ----------
        circuit : qml.QNode
            The quantum circuit to represent.
        params : array
            Weight parameters for the circuit.
        inputs : array
            Inputs for the circuit.
        gate_values : bool, optional
            Toggle for gate values or theta variables in the representation.

        Returns
        -------
        str
            LaTeX string for the circuit.
        """
        quantum_tape = qml.workflow.construct_tape(circuit)(
            params=params, inputs=inputs
        )
        print(quantum_tape.circuit, "\n")
        circuit_tikz = [
            [QuanTikz.ground_state()] for _ in range(quantum_tape.num_wires)
        ]

        index = iter(range(10 * quantum_tape.num_params))
        for op in quantum_tape.circuit:
            # catch measurement operations
            if op._queue_category == "_measurements":
                # get the maximum length of all wires
                max_len = max(len(circuit_tikz[cw]) for cw in range(len(circuit_tikz)))
                if op.wires[0] != 0:
                    max_len -= 1
                # extend the wire by the number of missing operations
                circuit_tikz[op.wires[0]].extend(
                    "" for _ in range(max_len - len(circuit_tikz[op.wires[0]]))
                )
                circuit_tikz[op.wires[0]].append(QuanTikz.measure(op))
            # process all gates
            elif op._queue_category == "_ops":
                # catch barriers
                if op.name == "Barrier":

                    # get the maximum length of all wires
                    max_len = max(
                        len(circuit_tikz[cw]) for cw in range(len(circuit_tikz))
                    )

                    # extend the wires by the number of missing operations
                    for ow in [i for i in range(len(circuit_tikz))]:
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
                        )
                    )
                # controlled gate?
                elif len(op.wires) == 2:
                    # build the controlled gate
                    if op.name in ["CRX", "CRY", "CRZ"]:
                        ctrl, targ = QuanTikz.cgate(
                            op, index=next(index), gate_values=gate_values
                        )
                    else:
                        ctrl, targ = QuanTikz.cgate(op)

                    # get the wires that this cgate spans over
                    crossing_wires = [
                        i for i in range(min(op.wires), max(op.wires) + 1)
                    ]
                    # get the maximum length of all operations currently on this wire
                    max_len = max([len(circuit_tikz[cw]) for cw in crossing_wires])

                    # extend the affected wires by the number of missing operations
                    for ow in [i for i in range(min(op.wires), max(op.wires) + 1)]:
                        circuit_tikz[ow].extend(
                            "" for _ in range(max_len - len(circuit_tikz[ow]))
                        )

                    # finally append the cgate operation
                    circuit_tikz[op.wires[0]].append(ctrl)
                    circuit_tikz[op.wires[1]].append(targ)

                    # extend the non-affected wires by the number of missing operations
                    for cw in crossing_wires - op.wires:
                        circuit_tikz[cw].append("")
                else:
                    raise NotImplementedError(">2-wire gates are not supported yet")

        quantikz_str = ""

        # get the maximum length of all wires
        max_len = max(len(circuit_tikz[cw]) for cw in range(len(circuit_tikz)))

        # extend the wires by the number of missing operations
        for ow in [i for i in range(len(circuit_tikz))]:
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
        # get number of layers
        # iterate layers and get wires
