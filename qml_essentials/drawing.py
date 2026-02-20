from fractions import Fraction
from itertools import cycle
from typing import Any, Dict, List, Tuple, Union

import jax.numpy as jnp

from qml_essentials.operations import (
    Operation,
)


def _ctrl_target_name(name: str) -> str:
    """Strip the leading 'C' from a controlled gate name to get the target name."""
    # CRX → RX, CX → X, etc.
    return name.replace("C", "")


# -- Text drawing -------------------------------------------------------


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
    # Each column is a dict mapping qubit → display string.
    columns: List[Dict[int, str]] = []
    wire_busy: Dict[int, int] = {}  # qubit → next free column index

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


# -- Matplotlib drawing -------------------------------------------------


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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

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


# -- TikZ drawing -------------------------------------------------------


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
    from qml_essentials.utils import QuanTikz

    # Build the per-wire column structure
    circuit_tikz: List[List[str]] = [["\\lstick{\\ket{0}}"] for _ in range(n_qubits)]

    # # Prepare an input symbol iterator
    # if isinstance(inputs_symbols, str):
    #     sym_iter = cycle([inputs_symbols])
    # else:
    #     sym_iter = cycle(inputs_symbols)

    param_index = 0

    for op in ops:
        if op.is_controlled and len(op.wires) == 2:
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
                if target_name == "X":
                    targ_cell = "\\targ{}"
                else:
                    targ_cell = "\\control{}"
            else:
                targ_cell = f"\\gate{{{target_name}}}"

            ctrl_cell = f"\\ctrl{{{distance}}}"

            # Align columns for crossing wires
            crossing = range(min(op.wires), max(op.wires) + 1)
            max_len = max(len(circuit_tikz[w]) for w in crossing)
            for w in crossing:
                circuit_tikz[w].extend(
                    "" for _ in range(max_len - len(circuit_tikz[w]))
                )

            circuit_tikz[ctrl_wire].append(ctrl_cell)
            circuit_tikz[targ_wire].append(targ_cell)

            # Pad intermediate wires
            for w in crossing:
                if w != ctrl_wire and w != targ_wire:
                    circuit_tikz[w].append("")

        elif len(op.wires) == 1:
            w = op.wires[0]
            name = op.name
            # Rename for display
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
        else:
            # Multi-qubit gate (>2 wires) — render as a generic box
            max_len = max(len(circuit_tikz[w]) for w in op.wires)
            for w in op.wires:
                circuit_tikz[w].extend(
                    "" for _ in range(max_len - len(circuit_tikz[w]))
                )
            label = _gate_label(op)
            for w in op.wires:
                circuit_tikz[w].append(f"\\gate{{{label}}}")

    # Equalise wire lengths
    max_len = max(len(wire) for wire in circuit_tikz)
    for wire in circuit_tikz:
        wire.extend("" for _ in range(max_len - len(wire)))

    # Build the quantikz string
    quantikz_str = ""
    for wire_idx, wire_ops in enumerate(circuit_tikz):
        for op_idx, cell in enumerate(wire_ops):
            if op_idx < len(wire_ops) - 1:
                quantikz_str += f"{cell} & "
            else:
                quantikz_str += f"{cell}"
                if wire_idx < n_qubits - 1:
                    quantikz_str += " \\\\\n"

    return QuanTikz.TikzFigure(quantikz_str)
