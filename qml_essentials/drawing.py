from fractions import Fraction
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from qml_essentials.operations import (
    Operation,
)


class TikzFigure:
    """Wrapper around a ``quantikz`` LaTeX string with export helpers."""

    def __init__(self, quantikz_str: str):
        self.quantikz_str = quantikz_str

    def __repr__(self):
        return self.quantikz_str

    def __str__(self):
        return self.quantikz_str

    def wrap_figure(self) -> str:
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

    def export(
        self, destination: str, full_document: bool = False, mode: str = "w"
    ) -> None:
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


# Backwards-compatible alias so existing ``QuanTikz.TikzFigure`` references
# keep working without changes in downstream code.
class QuanTikz:
    TikzFigure = TikzFigure


def _ctrl_target_name(name: str) -> str:
    """Strip the leading 'C' from a controlled gate name to get the target name."""
    # CRX -> RX, CX -> X, etc.
    return name.replace("C", "")


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


def _tikz_cell_barrier(
    op: Operation,
    circuit_tikz: List[List[str]],
) -> None:
    """Align all wires so that subsequent gates start in the same column.

    The barrier is a no-op visually — it only pads shorter wires so that
    every wire has the same number of cells at this point.
    """
    all_wires = list(range(len(circuit_tikz)))
    _tikz_align_wires(circuit_tikz, all_wires)


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
    **kwargs: Any,
) -> Any:
    """Render a circuit tape as LaTeX/TikZ ``quantikz`` code.

    Args:
        ops: Ordered list of gate operations (noise channels excluded).
        n_qubits: Total number of qubits.
        gate_values: If ``True``, show numeric angles; otherwise use
            symbolic \\theta_i labels.

    Returns:
        A :class:`~qml_essentials.drawing.TikzFigure` object.
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
        elif op.name == "Barrier":
            _tikz_cell_barrier(op, circuit_tikz)
        else:
            _tikz_cell_multiqubit(op, circuit_tikz)

    return TikzFigure(_tikz_build_string(circuit_tikz, n_qubits))


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


def _sample_envelope(ev: PulseEvent, t_lo: float, t_hi: float, n_samples: int):
    """Sample the envelope over [t_lo, t_hi] and return (t_arr, signal).

    Uses vectorised JAX operations instead of a Python loop.
    """
    t_c = ev.duration / 2
    t_arr = jnp.linspace(t_lo, t_hi, n_samples)
    env = ev.envelope_fn(ev.envelope_params, t_arr, t_c)
    signal = env * ev.w
    return t_arr, signal


def _compute_display_window(
    ev: PulseEvent,
    n_samples: int,
    envelope_width: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute the (t_lo, t_hi, amp_max) display window for a physical pulse.

    The display window is chosen adaptively based on how much the envelope
    decays within the evolution window ``[0, duration]``.  If the envelope
    is essentially zero at the edges, the evolution window is used as-is.
    Otherwise the window is widened until the envelope drops to
    ``edge_ratio ** 10`` of its peak, where ``edge_ratio`` is the
    amplitude at the window edge relative to the center.

    The ``envelope_width`` parameter scales the resulting extension beyond
    the evolution window.  ``1.0`` gives the default adaptive width,
    values ``> 1`` widen further, values ``< 1`` tighten, and ``0``
    clamps the display exactly to the evolution window ``[0, duration]``.

    Returns:
        ``(t_lo, t_hi, amp_max)`` — local time bounds and peak amplitude.
    """
    t_c = ev.duration / 2

    if envelope_width == 0:
        t_lo, t_hi = 0.0, ev.duration
    else:
        val_center = float(ev.envelope_fn(ev.envelope_params, t_c, t_c))

        if abs(val_center) < 1e-12:
            t_lo, t_hi = 0.0, ev.duration
        else:
            val_edge = float(ev.envelope_fn(ev.envelope_params, 0.0, t_c))
            edge_ratio = abs(val_edge / val_center)

            if edge_ratio < 0.01:
                t_lo, t_hi = 0.0, ev.duration
            else:
                target = edge_ratio**10
                lo, hi = ev.duration / 2, ev.duration * 50
                for _ in range(30):
                    mid = (lo + hi) / 2
                    val = float(ev.envelope_fn(ev.envelope_params, t_c + mid, t_c))
                    if abs(val / val_center) > target:
                        lo = mid
                    else:
                        hi = mid
                half_width = ev.duration / 2 + (hi - ev.duration / 2) * envelope_width
                t_lo = t_c - half_width
                t_hi = t_c + half_width

    _, signal = _sample_envelope(ev, t_lo, t_hi, n_samples)
    amp = float(jnp.max(jnp.abs(signal))) * 1.1
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
    t_arr, signal = _sample_envelope(ev, t_lo, t_hi, n_samples)
    t_display = t_arr + t_start

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
                alpha=0.7,
                zorder=4,
            )

        if show_carrier:
            modulated = signal * jnp.cos(omega_c * t_arr + ev.carrier_phase)
            ax.plot(
                t_display, modulated, color=color, linewidth=0.8, alpha=0.8, zorder=2
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
        ax.vlines(
            t_mid,
            -amp_max * 0.6,
            amp_max * 0.6,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )
        ax.annotate(
            _make_event_label(ev.gate, ev.parent),
            xy=(t_mid, amp_max * 0.85),
            fontsize=7,
            ha="center",
            va="bottom",
            color=color,
            # fontstyle="italic",
            zorder=5,
        )


def _draw_block(
    ev: PulseEvent, t_start: float, axes, color: str, amp_max: float
) -> None:
    """Draw a gate as a labelled rectangular block on each of its wires."""
    label = _make_event_label(ev.gate, ev.parent)
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

    axes[ev.wires[0]].annotate(
        label,
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
    show_envelope: bool = True,
    envelope_width: float = 0.0,
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
        show_envelope: If ``True``, draw the full envelope shape for
            physical pulses.  If ``False``, show them as simple
            rectangular blocks indicating duration only.
        envelope_width: Scales how far the displayed envelope extends
            beyond the evolution window.  ``1.0`` (default) uses the
            adaptive width, ``> 1`` widens further, ``< 1`` tightens,
            and ``0`` clamps the envelope exactly to the pulse duration.
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
        "RX": "#1F78B4",
        "RY": "#E69F00",
        "RZ": "#009371",
        "CZ": "#ED665A",
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

    # Pre-compute display windows, amplitude range, and x-limits
    display_windows: Dict[int, Tuple[float, float]] = {}
    amp_max = 1.0
    x_lo, x_hi = 0.0, t_total

    if show_envelope:
        for idx, (ev, t_start) in enumerate(scheduled):
            if ev.envelope_fn is None or ev.gate not in ("RX", "RY"):
                continue
            t_lo, t_hi, amp = _compute_display_window(ev, n_samples, envelope_width)
            display_windows[idx] = (t_lo, t_hi)
            amp_max = max(amp_max, amp)
            # Map local display bounds to global time coordinates
            x_lo = min(x_lo, t_lo + t_start)
            x_hi = max(x_hi, t_hi + t_start)

    x_margin = (x_hi - x_lo) * 0.05
    for q in range(n_qubits):
        axes[q].set_xlim(x_lo - x_margin, x_hi + x_margin)
        axes[q].set_ylim(-amp_max, amp_max)

    # Draw events
    for idx, (ev, t_start) in enumerate(scheduled):
        color = gate_colors.get(ev.gate, "#bab0ac")

        if ev.gate in ("RX", "RY") and ev.envelope_fn is not None and show_envelope:
            t_lo, t_hi = display_windows[idx]
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
        else:
            _draw_block(ev, t_start, axes, color, amp_max)

    # Legend
    used_gates = {ev.gate for ev, _ in scheduled}
    handles = [
        mpatches.Patch(color=c, alpha=0.7, label=g)
        for g, c in gate_colors.items()
        if g in used_gates
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower right",
            ncol=len(handles),
            fontsize=8,
            framealpha=0.8,
        )

    fig.suptitle(
        f"Pulse Schedule ({PulseInformation.get_envelope()} envelope)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig, axes
