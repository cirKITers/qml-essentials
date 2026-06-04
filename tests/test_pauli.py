"""Tests for the symbolic Pauli/Clifford layer (PauliWord) and the
Pauli-Clifford circuit transform (PauliCircuit).

Correctness of the symbolic algebra is validated against dense-matrix ground
truth, so these tests do not depend on the matrix helpers they are meant to
replace.
"""

import itertools

import numpy as np
import pytest

from qml_essentials.operations import (
    PauliWord,
    H,
    S,
    CX,
    CZ,
    PauliX,
    PauliY,
    PauliZ,
    PauliRot,
    _embed_matrix,
)

# Single-qubit Pauli matrices in the X^x Z^z convention.
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _xz_matrix(xb: int, zb: int) -> np.ndarray:
    """Single-qubit X^xb Z^zb."""
    m = _X if xb else _I
    return m @ (_Z if zb else _I)


def pw_to_matrix(pw: PauliWord) -> np.ndarray:
    """Dense matrix of a PauliWord:  i^phase * kron_q X^x_q Z^z_q."""
    mat = np.array([[1.0 + 0.0j]])
    for q in range(pw.n_qubits):
        mat = np.kron(mat, _xz_matrix(int(pw.x[q]), int(pw.z[q])))
    return (1j**pw.phase) * mat


def clifford_matrix(gate, n: int) -> np.ndarray:
    """Dense matrix of a Clifford gate embedded on *n* qubits."""
    return np.asarray(_embed_matrix(gate.matrix, gate.wires, list(range(n)), n))


def all_pauli_strings(n: int):
    return ("".join(p) for p in itertools.product("IXYZ", repeat=n))


class TestPauliWordAlgebra:
    """Symbolic Pauli algebra vs dense-matrix ground truth."""

    @pytest.mark.unittest
    @pytest.mark.parametrize("n", [1, 2])
    def test_compose_matches_matrix_product(self, n):
        for s1 in all_pauli_strings(n):
            for s2 in all_pauli_strings(n):
                p1 = PauliWord.from_pauli_string(s1, list(range(n)), n)
                p2 = PauliWord.from_pauli_string(s2, list(range(n)), n)
                got = pw_to_matrix(p1.compose(p2))
                expected = pw_to_matrix(p1) @ pw_to_matrix(p2)
                assert np.allclose(got, expected), f"{s1}*{s2}"

    @pytest.mark.unittest
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_commutes_with_matches_matrix(self, n):
        for s1 in all_pauli_strings(n):
            for s2 in all_pauli_strings(n):
                p1 = PauliWord.from_pauli_string(s1, list(range(n)), n)
                p2 = PauliWord.from_pauli_string(s2, list(range(n)), n)
                m1, m2 = pw_to_matrix(p1), pw_to_matrix(p2)
                commute = np.allclose(m1 @ m2, m2 @ m1)
                assert p1.commutes_with(p2) == commute, f"[{s1},{s2}]"

    @pytest.mark.unittest
    def test_roundtrip_string_and_phase(self):
        for s in all_pauli_strings(3):
            pw = PauliWord.from_pauli_string(s, [0, 1, 2], 3)
            bare, phase = pw.to_pauli_string_and_phase()
            assert bare == s
            assert np.isclose(phase, 1.0)  # Hermitian Pauli string => +1

    @pytest.mark.unittest
    def test_zero_expectation(self):
        # Diagonal (I/Z only) -> non-zero; otherwise zero.
        assert PauliWord.from_pauli_string("ZZ", [0, 1], 2).zero_expectation() == 1.0
        assert PauliWord.from_pauli_string("IZ", [0, 1], 2).zero_expectation() == 1.0
        assert PauliWord.from_pauli_string("XZ", [0, 1], 2).zero_expectation() == 0.0
        # -Z has phase 2 -> expectation -1
        zneg = PauliWord.from_pauli_string("Z", [0], 1).compose(
            PauliWord(np.zeros(1, np.int8), np.zeros(1, np.int8), 2)
        )
        assert zneg.zero_expectation() == -1.0

    @pytest.mark.unittest
    def test_from_operation_paulirot_generator(self):
        rot = PauliRot(0.3, "XY", wires=[0, 1])
        pw = PauliWord.from_operation(rot, 2)
        assert pw.to_pauli_string() == "XY"


class TestCliffordConjugation:
    """Symbolic C P C^dagger vs dense-matrix conjugation."""

    def _check(self, gate, n, adjoint_left):
        c = clifford_matrix(gate, n)
        cdg = c.conj().T
        for s in all_pauli_strings(n):
            pw = PauliWord.from_pauli_string(s, list(range(n)), n)
            sym = pw_to_matrix(pw.conjugate_by_clifford(gate, adjoint_left))
            ref = (
                (cdg @ pw_to_matrix(pw) @ c)
                if adjoint_left
                else (c @ pw_to_matrix(pw) @ cdg)
            )
            assert np.allclose(sym, ref), f"gate={gate.name} adj={adjoint_left} P={s}"

    @pytest.mark.unittest
    @pytest.mark.parametrize("adjoint_left", [False, True])
    @pytest.mark.parametrize(
        "gate_factory",
        [
            lambda: H(wires=0),
            lambda: S(wires=0),
            lambda: PauliX(wires=0),
            lambda: PauliY(wires=0),
            lambda: PauliZ(wires=0),
        ],
    )
    def test_single_qubit_cliffords(self, gate_factory, adjoint_left):
        self._check(gate_factory(), 1, adjoint_left)

    @pytest.mark.unittest
    @pytest.mark.parametrize("adjoint_left", [False, True])
    @pytest.mark.parametrize(
        "gate_factory",
        [
            lambda: CX(wires=[0, 1]),
            lambda: CX(wires=[1, 0]),
            lambda: CZ(wires=[0, 1]),
            lambda: H(wires=1),
            lambda: S(wires=1),
        ],
    )
    def test_two_qubit_cliffords(self, gate_factory, adjoint_left):
        self._check(gate_factory(), 2, adjoint_left)


class TestPauliCircuit:
    """Behaviour of the symbolic Pauli-Clifford circuit transform."""

    @pytest.mark.unittest
    def test_h_rz_commutes_to_rx(self):
        # Moving H past RZ(theta) yields RX(theta): H Z H = X.
        from qml_essentials.pauli import PauliCircuit
        from qml_essentials.operations import RZ

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
        from qml_essentials.pauli import PauliCircuit

        obs = PauliCircuit.cliffords_in_observable([CX(wires=[0, 1])], [PauliZ(0)], 2)
        assert len(obs) == 1
        word = obs[0]._pauli_word
        assert isinstance(word, PauliWord)
        # H X H on the measured qubit would differ; CX leaves Z_0 invariant.
        assert word.to_pauli_string() == "ZI"
