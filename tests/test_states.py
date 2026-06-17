"""Tests for the state-preparation utilities.

Closed forms are validated directly (Dicke support and norm, Haar normalisation
and determinism) and the explicit graph-state vector is cross-checked against a
JAQSI H + CZ simulation.
"""

import numpy as np
import jax
import pytest

from qml_essentials.states import (
    dicke_state,
    haar_state,
    graph_state_vector,
    matching_edges,
    path_edges,
    complete_edges,
)
from qml_essentials.jaqsi import Script
from qml_essentials.gates import Gates

jax.config.update("jax_enable_x64", True)


class TestDickeState:
    @pytest.mark.unittest
    def test_closed_form_2_1(self):
        # |D_{2,1}> = (|01> + |10>)/sqrt2; qubit 0 leftmost -> indices 1 and 2.
        expected = np.zeros(4, dtype=complex)
        expected[[1, 2]] = 1.0 / np.sqrt(2.0)
        assert np.allclose(dicke_state(2, 1), expected)

    @pytest.mark.unittest
    @pytest.mark.parametrize("n,k", [(2, 1), (3, 2), (4, 0), (4, 4), (4, 2)])
    def test_support(self, n, k):
        psi = dicke_state(n, k)
        support = set(np.flatnonzero(np.abs(psi) > 1e-12).tolist())
        assert support == {x for x in range(2**n) if bin(x).count("1") == k}

    @pytest.mark.unittest
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_norm(self, n):
        for k in range(n + 1):
            assert np.isclose(np.linalg.norm(dicke_state(n, k)), 1.0)

    @pytest.mark.unittest
    @pytest.mark.parametrize("n", [3, 4])
    def test_permutation_symmetric(self, n):
        # Dicke states are invariant under any qubit permutation; check a swap.
        psi = dicke_state(n, 2).reshape((2,) * n)
        assert np.allclose(psi, np.swapaxes(psi, 0, 1))

    @pytest.mark.unittest
    @pytest.mark.parametrize("k", [-1, 5])
    def test_invalid_k(self, k):
        with pytest.raises(ValueError):
            dicke_state(4, k)


class TestHaarState:
    @pytest.mark.unittest
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_norm(self, n):
        assert np.isclose(np.linalg.norm(haar_state(n, seed=0)), 1.0)

    @pytest.mark.unittest
    def test_deterministic_seed(self):
        assert np.allclose(haar_state(3, seed=7), haar_state(3, seed=7))

    @pytest.mark.unittest
    def test_accepts_generator(self):
        # A freshly seeded Generator reproduces the int-seed result.
        rng = np.random.default_rng(7)
        assert np.allclose(haar_state(3, seed=rng), haar_state(3, seed=7))

    @pytest.mark.unittest
    def test_different_seeds_differ(self):
        assert not np.allclose(haar_state(3, seed=1), haar_state(3, seed=2))


class TestGraphStateVector:
    @pytest.mark.unittest
    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("edge_fn", [matching_edges, path_edges, complete_edges])
    def test_matches_script(self, n, edge_fn):
        edges = edge_fn(n)

        def circ():
            for q in range(n):
                Gates.H(wires=q)
            for i, j in edges:
                Gates.CZ(wires=[i, j])

        state = np.asarray(Script(circ, n_qubits=n).execute(type="state"))
        gv = graph_state_vector(n, edges)
        assert np.isclose(np.abs(np.vdot(gv, state)), 1.0)

    @pytest.mark.unittest
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_norm(self, n):
        assert np.isclose(np.linalg.norm(graph_state_vector(n, path_edges(n))), 1.0)


class TestEdgeConstructors:
    @pytest.mark.unittest
    def test_matching_edges(self):
        assert matching_edges(4) == [(0, 1), (2, 3)]
        assert matching_edges(5) == [(0, 1), (2, 3)]

    @pytest.mark.unittest
    def test_path_edges(self):
        assert path_edges(4) == [(0, 1), (1, 2), (2, 3)]

    @pytest.mark.unittest
    def test_complete_edges(self):
        assert complete_edges(3) == [(0, 1), (0, 2), (1, 2)]
