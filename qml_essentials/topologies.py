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
    def _chain(
        cls,
        n_qubits: int,
        wrap: bool = False,
        reverse: bool = False,
    ) -> List[List[int]]:
        """
        Nearest-neighbour chain.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        wrap : bool
            If True the chain wraps around (circular), adding one
            extra pair so every qubit appears as both control and
            target.
        reverse : bool
            If False pairs run high→low ``[n-1→n-2, …, 1→0]``;
            if True pairs run low→high ``[0→1, 1→2, …]``.

        Returns
        -------
        List[List[int]]
        """
        count = n_qubits if wrap else n_qubits - 1
        if reverse:
            return [[(q - 1) % n_qubits, (q - 2) % n_qubits] for q in range(count)]
        return [[n_qubits - q - 1, (n_qubits - q) % n_qubits] for q in range(count)]

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

    @classmethod
    def _ring(
        cls,
        n_qubits: int,
        wrap: bool = False,
    ) -> List[List[int]]:
        """
        Ring pairing (used by CZ topologies).

        Pairs run ``[n-2→n-1, n-3→n-2, …, 0→1]``, i.e. each
        consecutive pair in descending control-qubit order.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        wrap : bool
            If True an extra ``[n-1, 0]`` pair is appended when
            ``n_qubits > 2``.

        Returns
        -------
        List[List[int]]
        """
        pairs = [
            [(n_qubits - q - 2) % n_qubits, (n_qubits - q - 1) % n_qubits]
            for q in range(n_qubits - 1)
        ]
        if wrap and n_qubits > 2:
            pairs.append([n_qubits - 1, 0])
        return pairs

    # ── public topology methods ────────────────────────────────

    @classmethod
    def linear(cls, n_qubits: int) -> List[List[int]]:
        """Chain running high→low: ``[n-1→n-2, …, 1→0]``."""
        return cls._chain(n_qubits)

    @classmethod
    def linear_reversed(cls, n_qubits: int) -> List[List[int]]:
        """Chain running low→high: ``[0→1, 1→2, …]``."""
        return [[q, q + 1] for q in range(n_qubits - 1)]

    @classmethod
    def circular(cls, n_qubits: int) -> List[List[int]]:
        """Wrapping chain high→low (every qubit is control once)."""
        return cls._chain(n_qubits, wrap=True)

    @classmethod
    def circular_reversed(cls, n_qubits: int) -> List[List[int]]:
        """Wrapping chain low→high."""
        return cls._chain(n_qubits, wrap=True, reverse=True)

    @classmethod
    def brick_layer(cls, n_qubits: int) -> List[List[int]]:
        """Even pairs then odd pairs: ``[0,1],[2,3],…,[1,2],[3,4],…``."""
        return cls._brick(n_qubits)

    @classmethod
    def brick_layer_wrap(cls, n_qubits: int) -> List[List[int]]:
        """Brick-layer with an extra ``[n-1, 0]`` wrapping pair."""
        return cls._brick(n_qubits, wrap=True)

    @classmethod
    def brick_layer_reversed(cls, n_qubits: int) -> List[List[int]]:
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
        return [[q, (q + stride) % n_qubits] for q in range(n_qubits)]

    @classmethod
    def ring_cz(cls, n_qubits: int) -> List[List[int]]:
        """Descending consecutive pairs without wrapping."""
        return cls._ring(n_qubits)

    @classmethod
    def ring_cz_wrap(cls, n_qubits: int) -> List[List[int]]:
        """Descending consecutive pairs with wrapping ``[n-1, 0]``."""
        return cls._ring(n_qubits, wrap=True)
