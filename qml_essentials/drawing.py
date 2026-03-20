from fractions import Fraction
from itertools import cycle
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from qml_essentials.operations import (
    Operation,
)


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
        script : Script
            A yaqsi Script wrapping the circuit function.
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


def _ctrl_target_name(name: str) -> str:
    """Strip the leading 'C' from a controlled gate name to get the target name."""
    # CRX -> RX, CX -> X, etc.
    return name.replace("C", "")


# Text drawing


def _format_param(val: float) -> str:
    """Format a numeric parameter for text display.

    Shows nice π-fractions when possible, otherwise 2 decimal places.
    """
    try:
        frac = Fraction(float(val) / float(jnp.pi)).limit_denominator(100)
        if frac.numerator == 0:
            return "0"
        if frac.denominator <= 12:
            if frac == Fraction(1, 1):
                return "π"
            if frac.numerator == 1:
                return f"π/{frac.denominator}"
            if frac.denominator == 1:
                return f"{frac.numerator}π"
            return f"{frac.numerator}π/{frac.denominator}"
    except (ValueError, ZeroDivisionError):
        pass
    return f"{float(val):.2f}"


def _gate_label(op: Operation) -> str:
    """Build a short label like ``RX(π/2)`` or ``H`` for a gate."""
    name = op.name
    params = op.parameters
    if params:
        param_str = ", ".join(_format_param(p) for p in params)
        return f"{name}({param_str})"
    return name


def draw_text(ops: List[Operation], n_qubits: int) -> str:
    """Render a circuit tape as an ASCII-art string.

    Args:
        ops: Ordered list of gate operations (noise channels excluded).
        n_qubits: Total number of qubits.

    Returns:
        Multi-line string with one row per qubit.
    """
    if not ops:
        lines = [f" q{q}: ───" for q in range(n_qubits)]
        return "\n".join(lines)

    # Schedule operations into time-step columns.
    # Each column is a dict mapping qubit -> display string.
    columns: List[Dict[int, str]] = []
    wire_busy: Dict[int, int] = {}  # qubit -> next free column index

    for op in ops:
        start = max((wire_busy.get(w, 0) for w in op.wires), default=0)

        # Ensure enough columns exist
        while len(columns) <= start:
            columns.append({})

        if op.is_controlled and len(op.wires) >= 2:
            ctrl_wires = op.wires[:-1]
            target_wire = op.wires[-1]
            target_name = _ctrl_target_name(op.name)

            # Build target label with parameters
            if op.parameters:
                param_str = ", ".join(_format_param(p) for p in op.parameters)
                target_label = f"{target_name}({param_str})"
            else:
                target_label = target_name

            for cw in ctrl_wires:
                columns[start][cw] = "●"
            columns[start][target_wire] = target_label

            # Mark all wires in the span as busy (for crossing wires)
            all_spanned = range(min(op.wires), max(op.wires) + 1)
            for w in all_spanned:
                wire_busy[w] = start + 1
        else:
            label = _gate_label(op)
            for w in op.wires:
                columns[start][w] = label
            for w in op.wires:
                wire_busy[w] = start + 1

    # Render the grid
    # Determine column widths
    col_widths = []
    for col in columns:
        max_w = max((len(v) for v in col.values()), default=1)
        col_widths.append(max(max_w, 1))

    lines = []
    for q in range(n_qubits):
        parts = [f" q{q}: "]
        for ci, col in enumerate(columns):
            w = col_widths[ci]
            if q in col:
                cell = col[q].center(w)
            else:
                cell = "─" * w
            parts.append(f"─┤{cell}├")
        parts.append("─")
        lines.append("".join(parts))

    return "\n".join(lines)


# Matplotlib drawing


def draw_mpl(
    ops: List[Operation],
    n_qubits: int,
    **kwargs: Any,
) -> Tuple:
    """Render a circuit tape as a Matplotlib figure.

    Args:
        ops: Ordered list of gate operations (noise channels excluded).
        n_qubits: Total number of qubits.
        **kwargs: Reserved for future options.

    Returns:
        Tuple ``(fig, ax)`` — a Matplotlib ``Figure`` and ``Axes``.
    """

    # Schedule into columns (same logic as text)
    columns: List[Dict[int, str]] = []
    wire_busy: Dict[int, int] = {}
    ctrl_info: List[Dict[str, Any]] = []  # per-column control gate metadata

    for op in ops:
        start = max((wire_busy.get(w, 0) for w in op.wires), default=0)
        while len(columns) <= start:
            columns.append({})
            ctrl_info.append({})

        if op.is_controlled and len(op.wires) >= 2:
            ctrl_wires = op.wires[:-1]
            target_wire = op.wires[-1]
            target_name = _ctrl_target_name(op.name)
            if op.parameters:
                param_str = ", ".join(_format_param(p) for p in op.parameters)
                target_label = f"{target_name}({param_str})"
            else:
                target_label = target_name

            for cw in ctrl_wires:
                columns[start][cw] = "●"
            columns[start][target_wire] = target_label

            ctrl_info[start] = {
                "ctrl": ctrl_wires,
                "target": target_wire,
            }

            all_spanned = range(min(op.wires), max(op.wires) + 1)
            for w in all_spanned:
                wire_busy[w] = start + 1
        else:
            label = _gate_label(op)
            for w in op.wires:
                columns[start][w] = label
                wire_busy[w] = start + 1

    n_cols = len(columns) if columns else 1
    fig_width = max(3.0, 1.2 * (n_cols + 2))
    fig_height = max(2.0, 0.8 * n_qubits)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.5, n_cols + 0.5)
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw qubit wires
    for q in range(n_qubits):
        ax.plot([-0.3, n_cols + 0.3], [q, q], color="black", linewidth=0.8, zorder=0)
        ax.text(
            -0.5,
            q,
            "|0⟩",
            ha="right",
            va="center",
            fontsize=10,
            fontfamily="monospace",
        )

    # Draw gates
    gate_box_h = 0.6
    gate_box_w = 0.6

    for ci, col in enumerate(columns):
        x = ci + 0.5

        # Draw control lines
        ci_meta = ctrl_info[ci] if ci < len(ctrl_info) else {}
        if ci_meta:
            all_wires = list(ci_meta["ctrl"]) + [ci_meta["target"]]
            y_min = min(all_wires)
            y_max = max(all_wires)
            ax.plot([x, x], [y_min, y_max], color="black", linewidth=1.0, zorder=1)

        for q, label in col.items():
            if label == "●":
                # Control dot
                ax.plot(x, q, "o", color="black", markersize=6, zorder=3)
            else:
                # Gate box
                fontsize = 9 if len(label) <= 6 else 7
                bw = max(gate_box_w, len(label) * 0.09 + 0.2)
                rect = mpatches.FancyBboxPatch(
                    (x - bw / 2, q - gate_box_h / 2),
                    bw,
                    gate_box_h,
                    boxstyle="round,pad=0.05",
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=2,
                )
                ax.add_patch(rect)
                ax.text(
                    x,
                    q,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    zorder=4,
                )

    fig.tight_layout()
    return fig, ax


# TikZ drawing


def _tikz_param_str(val: float, op_name: str) -> str:
    """Format a rotation angle as a LaTeX string for quantikz gates."""
    try:
        frac = Fraction(float(val) / float(jnp.pi)).limit_denominator(100)
        if frac.denominator > 12:
            return f"\\gate{{{op_name}({float(val):.2f})}}"
        if frac.denominator == 1 and frac.numerator == 1:
            return f"\\gate{{{op_name}(\\pi)}}"
        if frac.numerator == 0:
            return f"\\gate{{{op_name}(0)}}"
        if frac.denominator == 1:
            return f"\\gate{{{op_name}({frac.numerator}\\pi)}}"
        if frac.numerator == 1:
            return (
                f"\\gate{{{op_name}\\left("
                f"\\frac{{\\pi}}{{{frac.denominator}}}\\right)}}"
            )
        return (
            f"\\gate{{{op_name}\\left("
            f"\\frac{{{frac.numerator}\\pi}}{{{frac.denominator}}}"
            f"\\right)}}"
        )
    except (ValueError, ZeroDivisionError):
        return f"\\gate{{{op_name}({float(val):.2f})}}"


def _tikz_align_wires(circuit_tikz: List[List[str]], wires: List[int]) -> None:
    """Pad all *wires* to the same column length in-place."""
    max_len = max(len(circuit_tikz[w]) for w in wires)
    for w in wires:
        circuit_tikz[w].extend("" for _ in range(max_len - len(circuit_tikz[w])))


def _tikz_cell_controlled(
    op: Operation,
    circuit_tikz: List[List[str]],
    param_index: int,
    gate_values: bool,
) -> int:
    """Append cells for a 2-wire controlled gate; return updated param_index."""
    ctrl_wire = op.wires[0]
    targ_wire = op.wires[1]
    distance = targ_wire - ctrl_wire
    target_name = _ctrl_target_name(op.name)

    # Build target cell
    if op.parameters and target_name in ("RX", "RY", "RZ"):
        if gate_values:
            targ_cell = _tikz_param_str(float(op.parameters[0]), target_name)
        else:
            targ_cell = f"\\gate{{{target_name}(\\theta_{{{param_index}}})}}"
        param_index += 1
    elif target_name in ("X", "Y", "Z"):
        targ_cell = "\\targ{}" if target_name == "X" else "\\control{}"
    else:
        targ_cell = f"\\gate{{{target_name}}}"

    crossing = list(range(min(op.wires), max(op.wires) + 1))
    _tikz_align_wires(circuit_tikz, crossing)

    circuit_tikz[ctrl_wire].append(f"\\ctrl{{{distance}}}")
    circuit_tikz[targ_wire].append(targ_cell)

    # Pad intermediate wires
    for w in crossing:
        if w != ctrl_wire and w != targ_wire:
            circuit_tikz[w].append("")

    return param_index


def _tikz_cell_single(
    op: Operation,
    circuit_tikz: List[List[str]],
    param_index: int,
    gate_values: bool,
) -> int:
    """Append a cell for a single-qubit gate; return updated param_index."""
    w = op.wires[0]
    name = op.name
    if name == "Hadamard":
        name = "H"

    if gate_values and op.parameters:
        cell = _tikz_param_str(float(op.parameters[0]), name)
    elif op.parameters:
        cell = f"\\gate{{{name}(\\theta_{{{param_index}}})}}"
        param_index += 1
    else:
        cell = f"\\gate{{{name}}}"

    circuit_tikz[w].append(cell)
    return param_index


def _tikz_cell_multiqubit(
    op: Operation,
    circuit_tikz: List[List[str]],
) -> None:
    """Append cells for a multi-qubit (>2 wire) gate."""
    _tikz_align_wires(circuit_tikz, list(op.wires))
    label = _gate_label(op)
    for w in op.wires:
        circuit_tikz[w].append(f"\\gate{{{label}}}")


def _tikz_build_string(circuit_tikz: List[List[str]], n_qubits: int) -> str:
    """Render the column grid to a quantikz LaTeX string."""
    # Equalise wire lengths
    max_len = max(len(wire) for wire in circuit_tikz)
    for wire in circuit_tikz:
        wire.extend("" for _ in range(max_len - len(wire)))

    quantikz_str = ""
    for wire_idx, wire_ops in enumerate(circuit_tikz):
        for op_idx, cell in enumerate(wire_ops):
            if op_idx < len(wire_ops) - 1:
                quantikz_str += f"{cell} & "
            else:
                quantikz_str += f"{cell}"
                if wire_idx < n_qubits - 1:
                    quantikz_str += " \\\\\n"

    return quantikz_str


def draw_tikz(
    ops: List[Operation],
    n_qubits: int,
    gate_values: bool = False,
    inputs_symbols: Union[str, List[str]] = "x",
    **kwargs: Any,
) -> Any:
    """Render a circuit tape as LaTeX/TikZ ``quantikz`` code.

    Args:
        ops: Ordered list of gate operations (noise channels excluded).
        n_qubits: Total number of qubits.
        gate_values: If ``True``, show numeric angles; otherwise use
            symbolic θ_i labels.
        inputs_symbols: Symbol(s) for input-encoding gates.

    Returns:
        A :class:`~qml_essentials.utils.QuanTikz.TikzFigure` object.
    """
    circuit_tikz: List[List[str]] = [["\\lstick{\\ket{0}}"] for _ in range(n_qubits)]
    param_index = 0

    for op in ops:
        if op.is_controlled and len(op.wires) == 2:
            param_index = _tikz_cell_controlled(
                op, circuit_tikz, param_index, gate_values
            )
        elif len(op.wires) == 1:
            param_index = _tikz_cell_single(op, circuit_tikz, param_index, gate_values)
        else:
            _tikz_cell_multiqubit(op, circuit_tikz)

    return QuanTikz.TikzFigure(_tikz_build_string(circuit_tikz, n_qubits))


@dataclass
class PulseEvent:
    """Single pulse applied to one or more wires.

    Attributes:
        gate: Gate label, e.g. ``"RX"``, ``"CZ"``.
        wires: Target qubit wire(s).
        envelope_fn: Pure envelope function ``(p, t, t_c) -> amplitude``.
        envelope_params: Envelope-shape parameters (excluding ``w`` and ``t``).
        w: Rotation angle passed to the gate.
        duration: Pulse duration (evolution time).
        carrier_phase: Phase offset for the carrier cosine.
        parent: Optional high-level gate name that decomposed into this event.
    """

    gate: str
    wires: List[int]
    envelope_fn: Any  # (p, t, t_c) -> scalar
    envelope_params: Any  # jnp array of envelope shape params
    w: float  # rotation angle
    duration: float  # evolution time
    carrier_phase: float = 0.0  # phi_c in cos(omega_c * t + phi_c)
    parent: Optional[str] = None  # composite gate that owns this pulse


# Leaf gate metadata for pulse schedule drawing.
# ``physical`` gates have an envelope; virtual gates (RZ, CZ) do not.
LEAF_META = {
    "RX": {"carrier_phase": float(jnp.pi), "physical": True},
    "RY": {"carrier_phase": float(-jnp.pi / 2), "physical": True},
    "RZ": {"carrier_phase": 0.0, "physical": False},
    "CZ": {"carrier_phase": 0.0, "physical": False},
}


def _resolve_wires_for_drawing(wire_fn, wires_list):
    """Resolve a ``wire_fn`` string to concrete wire(s) for drawing."""
    if wire_fn == "all":
        return wires_list
    if wire_fn == "target":
        return [wires_list[-1]] if len(wires_list) > 1 else wires_list
    if wire_fn == "control":
        return [wires_list[0]]
    raise ValueError(f"Unknown wire_fn: {wire_fn!r}")


def collect_pulse_events(
    gate_name: str,
    w: Union[float, List[float]],
    wires: Union[int, List[int]],
    pulse_params: Any = None,
    parent: Optional[str] = None,
) -> List[PulseEvent]:
    """Decompose a (possibly composite) pulse gate into leaf PulseEvents.

    Walks the :class:`DecompositionStep` tree stored on :class:`PulseParams`
    and collects timing / envelope information for drawing — no quantum
    operations are applied.

    Args:
        gate_name: Name of the gate (``"RX"``, ``"H"``, ``"CX"``, etc.).
        w: Rotation angle(s).
        wires: Qubit index or ``[control, target]``.
        pulse_params: Pulse parameters or ``None`` for defaults.
        parent: Label of the enclosing composite gate.

    Returns:
        Ordered list of :class:`PulseEvent` objects.
    """
    from qml_essentials.gates import PulseEnvelope, PulseInformation

    pp_obj = PulseInformation.gate_by_name(gate_name)
    if pp_obj is None:
        raise ValueError(f"Unknown pulse gate: {gate_name!r}")

    wires_list = [wires] if isinstance(wires, int) else list(wires)
    parent_label = parent or gate_name

    # --- Leaf gate ---
    if pp_obj.is_leaf:
        meta = LEAF_META.get(gate_name)
        if meta is None:
            raise ValueError(f"Unknown pulse gate: {gate_name!r}")

        info = PulseEnvelope.get(PulseInformation.get_envelope())
        pp = pp_obj.split_params(pulse_params)

        if meta["physical"]:
            env_p = pp[:-1]
            dur = float(pp[-1])
            return [
                PulseEvent(
                    gate=gate_name,
                    wires=wires_list,
                    envelope_fn=info["fn"],
                    envelope_params=jnp.array(env_p),
                    w=float(w),
                    duration=dur,
                    carrier_phase=meta["carrier_phase"],
                    parent=parent_label,
                )
            ]
        else:
            # Virtual gate (RZ, CZ) — no physical envelope
            return [
                PulseEvent(
                    gate=gate_name,
                    wires=wires_list,
                    envelope_fn=None,
                    envelope_params=jnp.ravel(jnp.asarray(pp)),
                    w=float(w) if not isinstance(w, list) else 0.0,
                    duration=1.0,
                    carrier_phase=0.0,
                    parent=parent_label,
                )
            ]

    # --- Composite gate ---
    parts = pp_obj.split_params(pulse_params)
    events = []
    for step, child_params in zip(pp_obj.decomposition, parts):
        child_wires = _resolve_wires_for_drawing(step.wire_fn, wires_list)
        child_w = step.angle_fn(w) if step.angle_fn is not None else w
        events += collect_pulse_events(
            step.gate.name,
            child_w,
            child_wires,
            child_params,
            parent=parent_label,
        )
    return events


def _make_event_label(gate: str, parent: Optional[str]) -> str:
    """Build a display label, appending the parent gate if different."""
    if parent and parent != gate:
        return f"{gate} ({parent})"
    return gate


def _compute_display_window(
    ev: PulseEvent,
    n_samples: int,
    threshold: float = 0.05,
    max_display_mult: float = 6.0,
) -> Tuple[float, float, float]:
    """Compute the (t_lo, t_hi, amp_max) display window for a physical pulse.

    Uses binary search to find the half-width where the envelope drops
    to *threshold* of its peak.  The result is capped at
    ``max_display_mult * duration`` so that very broad envelopes
    (sigma >> duration) still produce a compact plot while showing enough
    of the bell curve to be visually distinguishable from a rectangle.

    Returns:
        ``(t_lo, t_hi, amp_max)`` — local time bounds and peak amplitude.
    """
    t_c = ev.duration / 2
    val_center = float(ev.envelope_fn(ev.envelope_params, t_c, t_c))

    if abs(val_center) < 1e-12:
        t_lo, t_hi = 0.0, ev.duration
    else:
        val_edge = float(ev.envelope_fn(ev.envelope_params, 0.0, t_c))

        if abs(val_edge / val_center) <= threshold:
            # Envelope already decays visibly within the evolution window
            t_lo, t_hi = 0.0, ev.duration
        else:
            # Binary-search for the half-width where the envelope drops
            # to `threshold` of its peak.
            lo, hi = ev.duration / 2, ev.duration * 200
            for _ in range(40):
                mid = (lo + hi) / 2
                val = float(ev.envelope_fn(ev.envelope_params, t_c + mid, t_c))
                if abs(val / val_center) > threshold:
                    lo = mid
                else:
                    hi = mid
            # Cap: show enough to see the shape, but stay compact.
            natural_half = hi * 1.1
            max_half = max_display_mult * ev.duration
            half_width = min(natural_half, max_half)
            t_lo = t_c - half_width
            t_hi = t_c + half_width

    t_arr = jnp.linspace(t_lo, t_hi, n_samples)
    env = jnp.array(
        [float(ev.envelope_fn(ev.envelope_params, ti, t_c)) for ti in t_arr]
    )
    amp = float(jnp.max(jnp.abs(env)) * abs(ev.w) * 1.1)
    return t_lo, t_hi, amp


def _draw_physical_pulse(
    ev: PulseEvent,
    t_start: float,
    t_lo: float,
    t_hi: float,
    axes,
    color: str,
    n_samples: int,
    omega_c: float,
    show_carrier: bool,
) -> None:
    """Draw a physical (RX/RY) pulse envelope on the given axes."""
    t_c = ev.duration / 2
    t_arr = jnp.linspace(t_lo, t_hi, n_samples)
    env = jnp.array(
        [float(ev.envelope_fn(ev.envelope_params, ti, t_c)) for ti in t_arr]
    )
    signal = env * ev.w
    t_display = t_arr - t_c + t_start + ev.duration / 2

    for wire in ev.wires:
        ax = axes[wire]
        ax.fill_between(t_display, signal, alpha=0.12, color=color, zorder=2)
        ax.plot(t_display, signal, color=color, linewidth=1.4, alpha=0.85, zorder=3)

        # Mark evolution window boundaries with visible dashed lines
        for t_edge in (t_start, t_start + ev.duration):
            ax.axvline(
                t_edge,
                color=color,
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                zorder=4,
            )

        if show_carrier:
            modulated = signal * jnp.cos(omega_c * t_arr + ev.carrier_phase)
            ax.plot(
                t_display, modulated, color=color, linewidth=0.5, alpha=0.4, zorder=2
            )

        peak_idx = jnp.argmax(jnp.abs(signal))
        ax.annotate(
            _make_event_label(ev.gate, ev.parent),
            xy=(float(t_display[peak_idx]), float(signal[peak_idx])),
            fontsize=7,
            ha="center",
            va="bottom" if signal[peak_idx] >= 0 else "top",
            color=color,
            fontweight="bold",
            zorder=5,
        )


def _draw_virtual_z(
    ev: PulseEvent, t_start: float, axes, color: str, amp_max: float
) -> None:
    """Draw a virtual-Z gate as a dashed vertical line."""
    t_mid = t_start + ev.duration / 2
    for wire in ev.wires:
        ax = axes[wire]
        ax.axvline(
            t_mid, color=color, linestyle="--", linewidth=1.0, alpha=0.7, zorder=2
        )
        ax.annotate(
            _make_event_label(ev.gate, ev.parent),
            xy=(t_mid, amp_max * 0.85),
            fontsize=6,
            ha="center",
            va="bottom",
            color=color,
            fontstyle="italic",
            zorder=5,
        )


def _draw_cz(ev: PulseEvent, t_start: float, axes, color: str, amp_max: float) -> None:
    """Draw a CZ gate as a shaded rectangle spanning its wires."""
    for wire in ev.wires:
        ax = axes[wire]
        rect = mpatches.Rectangle(
            (t_start, -amp_max * 0.6),
            ev.duration,
            amp_max * 1.2,
            alpha=0.2,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            zorder=1,
        )
        ax.add_patch(rect)

    ax = axes[ev.wires[0]]
    ax.annotate(
        _make_event_label(ev.gate, ev.parent),
        xy=(t_start + ev.duration / 2, amp_max * 0.7),
        fontsize=7,
        ha="center",
        va="bottom",
        color=color,
        fontweight="bold",
        zorder=5,
    )


def draw_pulse_schedule(
    events: List[PulseEvent],
    n_qubits: int,
    n_samples: int = 200,
    show_carrier: bool = True,
    **kwargs: Any,
) -> Tuple:
    """Render a pulse schedule as a Matplotlib figure.

    Each qubit gets its own subplot row.  Physical pulses (RX, RY) are
    drawn as filled envelope shapes; virtual-Z gates are shown as thin
    vertical lines; CZ gates appear as shaded rectangles spanning both
    wires.

    Args:
        events: Ordered list of :class:`PulseEvent` from
            :func:`collect_pulse_events`.
        n_qubits: Total number of qubits.
        n_samples: Number of time samples per pulse envelope.
        show_carrier: If ``True``, overlay the carrier-modulated waveform
            (envelope x cos) as a thin line.
        **kwargs: Forwarded to ``plt.subplots``.

    Returns:
        ``(fig, axes)`` — Matplotlib Figure and array of Axes.
    """
    from qml_essentials.gates import PulseGates, PulseInformation

    omega_c = float(PulseGates.omega_c)

    # Assign start times per wire (sequential, no parallelism)
    wire_cursor: Dict[int, float] = {q: 0.0 for q in range(n_qubits)}
    scheduled: List[Tuple[PulseEvent, float]] = []  # (event, t_start)

    for ev in events:
        t_start = max(wire_cursor[w] for w in ev.wires)
        scheduled.append((ev, t_start))
        for w in ev.wires:
            wire_cursor[w] = t_start + ev.duration

    t_total = max(wire_cursor.values()) if wire_cursor else 1.0

    gate_colors = {
        "RX": "#4e79a7",
        "RY": "#f28e2b",
        "RZ": "#76b7b2",
        "CZ": "#e15759",
        "H": "#59a14f",
    }

    fig, axes = plt.subplots(
        n_qubits,
        1,
        figsize=kwargs.pop("figsize", (max(8, t_total * 2.5), 1.8 * n_qubits)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for q in range(n_qubits):
        ax = axes[q]
        ax.set_ylabel(f"q{q}", rotation=0, labelpad=20, fontsize=11, va="center")
        ax.axhline(0, color="grey", linewidth=0.4, zorder=0)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time", fontsize=11)

    # Pre-compute display windows and amplitude range for physical pulses
    display_windows: Dict[int, Tuple[float, float]] = {}
    amp_max = 1.0
    for idx, (ev, _) in enumerate(scheduled):
        if ev.envelope_fn is not None and ev.gate in ("RX", "RY"):
            t_lo, t_hi, amp = _compute_display_window(ev, n_samples)
            display_windows[idx] = (t_lo, t_hi)
            amp_max = max(amp_max, amp)

    # Compute effective x-limits accounting for widened display windows
    x_lo, x_hi = 0.0, t_total
    for idx, (ev, t_start) in enumerate(scheduled):
        if idx in display_windows:
            t_c = ev.duration / 2
            dw_lo, dw_hi = display_windows[idx]
            x_lo = min(x_lo, dw_lo - t_c + t_start + ev.duration / 2)
            x_hi = max(x_hi, dw_hi - t_c + t_start + ev.duration / 2)
    x_margin = (x_hi - x_lo) * 0.05
    for q in range(n_qubits):
        axes[q].set_xlim(x_lo - x_margin, x_hi + x_margin)
        axes[q].set_ylim(-amp_max, amp_max)

    # Draw events
    for idx, (ev, t_start) in enumerate(scheduled):
        color = gate_colors.get(ev.gate, "#bab0ac")

        if ev.gate in ("RX", "RY") and ev.envelope_fn is not None:
            t_lo, t_hi = display_windows.get(idx, (0.0, ev.duration))
            _draw_physical_pulse(
                ev,
                t_start,
                t_lo,
                t_hi,
                axes,
                color,
                n_samples,
                omega_c,
                show_carrier,
            )
        elif ev.gate == "RZ":
            _draw_virtual_z(ev, t_start, axes, color, amp_max)
        elif ev.gate == "CZ":
            _draw_cz(ev, t_start, axes, color, amp_max)

    # Legend
    handles = []
    used_gates = {ev.gate for ev, _ in scheduled}
    for gate, color in gate_colors.items():
        if gate in used_gates:
            handles.append(mpatches.Patch(color=color, alpha=0.5, label=gate))
    if handles:
        axes[0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)

    fig.suptitle(
        f"Pulse Schedule ({PulseInformation.get_envelope()} envelope)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig, axes
