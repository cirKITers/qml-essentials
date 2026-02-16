from typing import List
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
    def _consecutive(
        cls,
        n_qubits: int,
        wrap: bool = False,
        reverse: bool = False,
        target_forward: bool = False,
        stride: int = 1,
    ) -> List[List[int]]:
        """
        Unified generator for nearest-neighbour and strided pair topologies.
        Produces ``[control, target]`` pairs of qubits.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        wrap : bool
            If *target_forward* is True the wrap extends the
            iteration count so that every qubit appears as control
            once (chain / circular behaviour).
            If *target_forward* is False an extra ``[n-1, 0]`` (or
            ``[0, n-1]`` when reversed) pair is appended when
            ``n_qubits > 2`` (ring behaviour).
        reverse : bool
            Reverses both the iteration direction and the
            control→target offset.
        target_forward : bool
            If True the target qubit is ``(control + stride) % n``
            (chain-like); if False it is ``(control - stride) % n``
            (ring-like).  The sign is flipped when *reverse* is
            True.
        stride : int
            Offset between control and target qubit. Defaults to 1
            (nearest-neighbour). For example ``stride=2`` pairs each
            qubit with the one two positions away.

        Returns
        -------
        List[List[int]]
        """
        n = n_qubits
        # XOR: forward offset when exactly one of the flags is set
        delta = stride if (target_forward != reverse) else -stride

        # Number of pairs produced by the main loop
        if target_forward:
            count = n if wrap else n - 1
        else:
            count = n - 1

        # Control-qubit sequence
        if reverse:
            if target_forward:
                ctrls = [(q - 1) % n for q in range(count)]
            else:
                ctrls = [(q + 1) % n for q in range(count)]
        else:
            ctrls = [n - q - 1 for q in range(count)]

        pairs = [[c, (c + delta) % n] for c in ctrls]

        # Ring-style wrap: append one extra closing pair
        if not target_forward and wrap and n > 2:
            pairs.append([0, n - 1] if reverse else [n - 1, 0])

        return pairs

    @classmethod
    def _brick(
        cls,
        n_qubits: int,
        wrap: bool = False,
        reverse_pairs: bool = False,
    ) -> List[List[int]]:
        """
        Brick-layer (even/odd) pairing.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        wrap : bool
            If True an extra ``[n-1, 0]`` pair is appended when
            ``n_qubits > 2``.
        reverse_pairs : bool
            If True the control/target order inside every pair is
            swapped (``[1,0]`` instead of ``[0,1]``).

        Returns
        -------
        List[List[int]]
        """
        pairs: List[List[int]] = []
        for q in range(n_qubits // 2):
            a, b = 2 * q, 2 * q + 1
            pairs.append([b, a] if reverse_pairs else [a, b])
        for q in range((n_qubits - 1) // 2):
            a, b = 2 * q + 1, 2 * q + 2
            pairs.append([b, a] if reverse_pairs else [a, b])
        if wrap and n_qubits > 2:
            pairs.append([n_qubits - 1, 0])
        return pairs

    # ── public topology methods ────────────────────────────────
    @classmethod
    def downstairs(cls, n_qubits: int) -> List[List[int]]:
        """Descending consecutive pairs without wrapping."""
        return cls._consecutive(n_qubits, reverse=True)

    @classmethod
    def upstairs(cls, n_qubits: int) -> List[List[int]]:
        """Descending consecutive pairs without wrapping."""
        return cls._consecutive(n_qubits)

    @classmethod
    def upstairs_wraped(cls, n_qubits: int) -> List[List[int]]:
        """Descending consecutive pairs with wrapping ``[n-1, 0]``."""
        return cls._consecutive(n_qubits, wrap=True)

    @classmethod
    def wraped_upstairs(cls, n_qubits: int) -> List[List[int]]:
        """Wrapping chain high→low (every qubit is control once)."""
        return cls._consecutive(n_qubits, wrap=True, target_forward=True)

    @classmethod
    def wraped_downstairs(cls, n_qubits: int) -> List[List[int]]:
        """Wrapping chain low→high."""
        return cls._consecutive(n_qubits, wrap=True, reverse=True, target_forward=True)

    @classmethod
    def brick(cls, n_qubits: int) -> List[List[int]]:
        """Even pairs then odd pairs: ``[0,1],[2,3],…,[1,2],[3,4],…``."""
        return cls._brick(n_qubits)

    @classmethod
    def brick_wraped(cls, n_qubits: int) -> List[List[int]]:
        """Brick-layer with an extra ``[n-1, 0]`` wrapping pair."""
        return cls._brick(n_qubits, wrap=True)

    @classmethod
    def brick_wraped_mirrored(cls, n_qubits: int) -> List[List[int]]:
        """Brick-layer with swapped control/target inside each pair."""
        return cls._brick(n_qubits, reverse_pairs=True)

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

    @classmethod
    def strongly_ent(cls, n_qubits: int, stride: int = 1) -> List[List[int]]:
        """
        Circular stride-*k* pairing: ``q → (q + stride) % n``.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        stride : int
            Offset between control and target qubit.
        """
        return cls._consecutive(n_qubits, wrap=True, target_forward=True, stride=stride)
