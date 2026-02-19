"""Thread-safe, re-entrant tape recording context for quantum operations.

The tape is the mechanism by which :class:`~qml_essentials.operations.Operation`
instances are collected when a circuit function is executed.  This module
provides a clean context-manager interface that:

* is **thread-safe** — each thread has its own tape stack via
  ``threading.local()``, so concurrent ``QuantumScript.execute`` calls in
  different threads never interfere;
* is **re-entrant** — nested ``recording()`` blocks each get their own tape
  (implemented as a stack), which is important because
  ``QuantumScript._execute_batched`` re-records inside a ``jax.vmap``-traced
  function;
* is **multiprocessing-safe** by construction — each process has its own
  address space, so there is nothing to share.

Usage (inside ``QuantumScript._record``)::

    with recording() as tape:
        circuit_fn(*args, **kwargs)
    # tape is now a list[Operation]

Usage (inside ``Operation.__init__``)::

    tape = active_tape()
    if tape is not None:
        tape.append(self)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, List, Optional

if TYPE_CHECKING:
    from qml_essentials.operations import Operation

# ---------------------------------------------------------------------------
# Thread-local tape stack
# ---------------------------------------------------------------------------
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

    Example::

        with recording() as tape:
            RX(0.5, wires=0)
            H(wires=1)
        assert len(tape) == 2
    """
    stack = _tape_stack()
    tape: List["Operation"] = []
    stack.append(tape)
    try:
        yield tape
    finally:
        stack.pop()
