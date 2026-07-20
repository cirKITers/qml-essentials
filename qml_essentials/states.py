r"""State-preparation utilities.

Constructors for input statevectors used in trainability and barren-plateau
analysis.  All functions return a dense statevector of shape :math:`(2^n,)`
with qubit 0 leftmost (most significant), consistent with
:mod:`qml_essentials.algebra` and ``g_purity_from_basis``.  The arrays are plain
numpy and auto-convert to the jnp ``initial_state`` accepted by
:meth:`qml_essentials.script.Script.execute`.

- :func:`dicke_state` builds the permutation-symmetric Dicke state
  :math:`|D_{n,k}\rangle`.
- :func:`haar_state` builds a Haar-random pure state.
- :func:`graph_state_vector` builds the graph state :math:`\prod_{(i,j)} CZ_{ij}
  H^{\otimes n}|0\rangle` for a given edge set, with :func:`matching_edges`,
  :func:`path_edges`, and :func:`complete_edges` as standard edge-set
  constructors.
"""

from typing import Iterable, List, Tuple, Union

import numpy as np


def dicke_state(n: int, k: int) -> np.ndarray:
    r"""Return the Dicke state :math:`|D_{n,k}\rangle`.

    The permutation-symmetric equal superposition of all computational basis
    states of Hamming weight :math:`k`.

    Args:
        n: Number of qubits.
        k: Hamming weight, with :math:`0 \leq k \leq n`.

    Returns:
        The normalised statevector of shape :math:`(2^n,)` (qubit 0 leftmost).

    Raises:
        ValueError: If ``n < 1`` or ``k`` is outside ``[0, n]``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k} for n={n}")
    psi = np.zeros(2**n, dtype=complex)
    idx = [x for x in range(2**n) if bin(x).count("1") == k]
    psi[idx] = 1.0
    return psi / np.linalg.norm(psi)


def haar_state(
    n: int, seed: Union[int, np.random.Generator, None] = None
) -> np.ndarray:
    r"""Return a Haar-random pure state.

    Drawn as a normalised complex Gaussian vector, which is uniform with respect
    to the Haar measure on pure states.

    Args:
        n: Number of qubits.
        seed: Source of randomness.  An ``int`` seed, a
            :class:`numpy.random.Generator`, or ``None`` for fresh entropy.

    Returns:
        The normalised statevector of shape :math:`(2^n,)`.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return v / np.linalg.norm(v)


def graph_state_vector(n: int, edges: Iterable[Tuple[int, int]]) -> np.ndarray:
    r"""Return the explicit graph-state statevector for the given edges.

    The graph state is :math:`\prod_{(i,j) \in E} CZ_{ij} H^{\otimes n}|0\rangle`.
    Each :math:`CZ` flips the sign of the amplitudes whose two qubits are both
    one.

    Args:
        n: Number of qubits.
        edges: Iterable of qubit-index pairs :math:`(i, j)`.

    Returns:
        The statevector of shape :math:`(2^n,)` (qubit 0 leftmost).
    """
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)  # H^n|0>
    idx = np.arange(2**n)
    for i, j in edges:
        bi = (idx >> (n - 1 - i)) & 1  # qubit 0 is most significant
        bj = (idx >> (n - 1 - j)) & 1
        psi = psi * np.where(bi & bj, -1.0, 1.0)
    return psi


def matching_edges(n: int) -> List[Tuple[int, int]]:
    r"""Return the perfect-matching edges :math:`(0,1),(2,3),\dots`.

    Yields a disconnected graph state (a product of two-qubit graph states) with
    low entanglement.

    Args:
        n: Number of qubits.

    Returns:
        The list of edge pairs.
    """
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def path_edges(n: int) -> List[Tuple[int, int]]:
    r"""Return the path edges :math:`0\text{-}1\text{-}\dots\text{-}(n-1)`.

    Yields a connected one-dimensional cluster state.

    Args:
        n: Number of qubits.

    Returns:
        The list of edge pairs.
    """
    return [(i, i + 1) for i in range(n - 1)]


def complete_edges(n: int) -> List[Tuple[int, int]]:
    r"""Return the complete-graph edges of :math:`K_n`.

    Yields a connected, permutation-symmetric graph state.

    Args:
        n: Number of qubits.

    Returns:
        The list of edge pairs :math:`(i, j)` with :math:`i < j`.
    """
    return [(i, j) for i in range(n) for j in range(i + 1, n)]
