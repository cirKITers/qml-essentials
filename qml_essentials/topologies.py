from typing import List, Callable, Union
import jax
import logging

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


class Topology:
    """
    Generates [control, target] wire-pair lists for two-qubit gates.

    All public methods are static and share a small set of private
    helpers so that related topologies (e.g. ``linear`` / ``circular``,
    ``brick_layer`` / ``brick_layer_wrap``) re-use the same core logic.

    Raises
    ------
    ValueError
        If ``n_qubits < 2`` is passed to any topology method.
    """

    @classmethod
    def stairs(
        cls,
        n_qubits: int,
        offset: Union[int, Callable] = 0,
        wrap=False,
        reverse: bool = True,
        mirror: bool = True,
        span: Union[int, Callable] = 1,
        stride: int = 1,
        modulo: bool = True,
    ) -> List[List[int]]:
        """
        Unified generator for nearest-neighbour and spand pair topologies.
        Produces ``[control, target]`` pairs of qubits.

        The default values, produce an "upstairs" entangling sequence
        without wrapping around the last gate.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        offset : Union[int, Callable]
            Offset for starting the entangling sequence.
            Can either be a integer or a callable that takes n_qubits as input.
        wrap : bool
            Wraps around the entangling gates.
        reverse : bool
            Reverses both the iteration direction (upstairs/ downstairs)
        mirror: bool
            Flip target/ control qubit
        span : int
            Offset between control and target qubit. Defaults to 1
        stride : int
            Step size for entangling gates. Defaults to 1, meaning a stair
            pattern will be generated.
        modulo : bool
            If a gate should be placed when the iterator decreases below 0
            or exceeds n_qubits. Defaults to True

        Returns
        -------
        List[List[int]]
        """
        ctrls = []
        targets = []

        n_gates = n_qubits if wrap else n_qubits - 1
        _offset = offset(n_qubits) if callable(offset) else offset
        _span = span(n_qubits) if callable(span) else span

        for q in range(0, n_gates, stride):
            _target = q + _offset + _span
            if _target >= n_qubits and not modulo:
                continue
            _control = q + _offset
            if _control < 0 and not modulo:
                continue

            if _target == _control:
                continue

            targets += [_target % n_qubits]
            ctrls += [_control % n_qubits]

        if reverse:
            ctrls = reversed(ctrls)
            targets = reversed(targets)

        if mirror:
            ctrls, targets = targets, ctrls

        pairs = list(zip(ctrls, targets, strict=True))

        return pairs

    @classmethod
    def bricks(cls, n_qubits: int, **kwargs) -> List[List[int]]:
        kwargs.setdefault("stride", 2)
        kwargs.setdefault("modulo", False)
        kwargs.setdefault("reverse", False)
        return cls.stairs(n_qubits=n_qubits, **kwargs)

    @classmethod
    def all_to_all(cls, n_qubits: int) -> List[List[int]]:
        """Every ordered pair ``(i, j)`` with ``i ≠ j``."""
        pairs: List[List[int]] = []
        for ql in range(n_qubits):
            for q in range(n_qubits):
                if q != ql:
                    pairs.append(
                        [
                            n_qubits - ql - 1,
                            (n_qubits - q - 1) % n_qubits,
                        ]
                    )
        return pairs
