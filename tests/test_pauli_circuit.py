"""Tests for the Pauli-Clifford circuit transform (PauliCircuit).

The transform rewrites a parameterised circuit into the Pauli-Clifford normal
form used by the analytical Fourier tree.  Correctness is validated against
the simulator's own expectation values.
"""

import numpy as np
import pytest

from jaqsi.paulis import (
    PauliWord,
)
from jaqsi.gateset import (
    H,
    CX,
    CZ,
    RX,
    RY,
    RZ,
    PauliZ,
)
from jaqsi import simulation

from qml_essentials.pauli import PauliCircuit


class TestPauliCircuit:
    """Behaviour of the symbolic Pauli-Clifford circuit transform."""

    @pytest.mark.unittest
    def test_h_rz_commutes_to_rx(self):
        # Moving H past RZ(theta) yields RX(theta): H Z H = X.
        ops = [H(wires=0), RZ(0.7, wires=0)]
        pauli_gates, cliffords = PauliCircuit.commute_all_cliffords_to_the_end(ops, 1)

        assert len(pauli_gates) == 1
        assert PauliWord.from_operation(pauli_gates[0], 1).to_pauli_string() == "X"
        assert np.isclose(float(pauli_gates[0].parameters[0]), 0.7)
        assert len(cliffords) == 1 and cliffords[0].name == "H"

    @pytest.mark.unittest
    def test_evolved_observable_carries_symbolic_word(self):
        # CX then measure Z on the control: Z_0 evolves to Z_0 (commutes),
        # while the symbolic word is cached on the returned observable.
        obs = PauliCircuit.cliffords_in_observable([CX(wires=[0, 1])], [PauliZ(0)], 2)
        assert len(obs) == 1
        word = obs[0]._pauli_word
        assert isinstance(word, PauliWord)
        # H X H on the measured qubit would differ; CX leaves Z_0 invariant.
        assert word.to_pauli_string() == "ZI"

    @pytest.mark.unittest
    def test_cz_is_clifford_and_commuted(self):
        """A bare CZ is now classified Clifford and commuted (not decomposed),
        and the canonical form reproduces the original circuit's expectation."""
        ops = [RX(0.3, wires=0), CZ(wires=[0, 1]), RY(0.5, wires=1)]
        obs = [PauliZ(0)]

        ref = simulation.simulate_and_measure(ops, 2, "expval", obs, False)
        operations, observables = PauliCircuit.from_parameterised_circuit(
            ops, obs, n_qubits=2
        )

        # CZ commuted to the end, not split into H/CX/H -> only Pauli rotations.
        assert all(PauliCircuit._is_pauli_rotation(o) for o in operations)
        got = simulation.simulate_and_measure(
            operations, 2, "expval", observables, False
        )
        assert np.allclose(np.asarray(ref), np.asarray(got), atol=1e-6)
