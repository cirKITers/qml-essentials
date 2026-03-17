from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, List, Optional

if TYPE_CHECKING:
    from qml_essentials.operations import Operation

_local = threading.local()


def _tape_stack() -> List[List["Operation"]]:
    """Return the per-thread tape stack, creating it on first access.

    Returns:
        The tape stack for the current thread (a list of tape lists).
    """
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def active_tape() -> Optional[List["Operation"]]:
    """Return the innermost active tape, or ``None`` if not recording.

    This is called from :meth:`Operation.__init__` to decide whether an
    operation should be appended to a tape.

    Returns:
        The currently active tape list, or ``None``.
    """
    stack = _tape_stack()
    return stack[-1] if stack else None


@contextmanager
def recording() -> Iterator[List["Operation"]]:
    """Context manager that creates a fresh tape for recording operations.

    Operations instantiated inside this block will be appended to the
    returned tape list (via :func:`active_tape`).  Nesting is supported:
    each ``with recording()`` pushes a new tape onto the per-thread stack,
    and the previous tape is restored on exit.

    Yields:
        A new empty list that will be populated with ``Operation`` instances.
    """
    stack = _tape_stack()
    tape: List["Operation"] = []
    stack.append(tape)
    try:
        yield tape
    finally:
        stack.pop()


def _pulse_tape_stack() -> List[list]:
    """Return the per-thread pulse-event tape stack."""
    if not hasattr(_local, "pulse_stack"):
        _local.pulse_stack = []
    return _local.pulse_stack


def active_pulse_tape() -> Optional[list]:
    """Return the innermost active pulse-event tape, or ``None``.

    Called from :class:`~qml_essentials.gates.PulseGates` leaf methods
    to record :class:`~qml_essentials.drawing.PulseEvent` objects.
    """
    stack = _pulse_tape_stack()
    return stack[-1] if stack else None


@contextmanager
def pulse_recording() -> Iterator[list]:
    """Context manager that collects pulse events emitted by PulseGates.

    Yields:
        A list that will be populated with
        :class:`~qml_essentials.drawing.PulseEvent` instances.
    """
    stack = _pulse_tape_stack()
    tape: list = []
    stack.append(tape)
    try:
        yield tape
    finally:
        stack.pop()
