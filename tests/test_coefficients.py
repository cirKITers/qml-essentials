from qml_essentials.model import Model
from qml_essentials.ansaetze import Encoding
from qml_essentials.coefficients import Coefficients, FourierTree, FCC, Datasets
from pennylane.fourier import coefficients as pcoefficients

import traceback
import numpy as np
import jax.numpy as jnp
import jax
import logging
import pytest
from scipy.stats import pearsonr, spearmanr

from functools import partial


logger = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", True)


class TestCoefficients:
    """Tests for Fourier coefficient computation via FFT."""

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "circuit_type, n_qubits, n_layers, output_qubit",
        [
            ("Circuit_1", 3, 1, [0, 1]),
            ("Circuit_9", 4, 1, 0),
            ("Circuit_19", 5, 1, 0),
        ],
        ids=["Circuit_1-3q", "Circuit_9-4q", "Circuit_19-5q"],
    )
    def test_coefficients(self, circuit_type, n_qubits, n_layers, output_qubit) -> None:
        reference_inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

        model = Model(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            output_qubit=output_qubit,
        )

        coeffs, freqs = Coefficients.get_spectrum(model)

        assert coeffs.shape == model.degree, "Wrong number of coefficients"
        assert jnp.isclose(jnp.sum(coeffs).imag, 0.0, rtol=1.0e-5), (
            "Imaginary part is not zero"
        )

        partial_circuit = partial(model, model.params, force_mean=True)
        ref_coeffs = pcoefficients(partial_circuit, 1, model.degree[0] // 2)

        assert jnp.allclose(coeffs, ref_coeffs, rtol=1.0e-5), (
            "Coefficients don't match the pennylane reference"
        )

        for ref_input in reference_inputs:
            exp_model = model(params=None, inputs=ref_input, force_mean=True)

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=coeffs,
                frequencies=freqs,
                inputs=ref_input,
            )

            assert jnp.isclose(exp_model, exp_fourier, atol=1.0e-5), (
                "Fourier series does not match model expectation"
            )

    @pytest.mark.unittest
    def test_dummy_model(self) -> None:
        class Model_Fct:
            def __init__(self, c, f):
                self.c = c
                self.f = f
                self.degree = (2 * max(f) + 1,)
                self.frequencies = f
                self.n_input_feat = 1

            def __call__(self, inputs, **kwargs):
                return np.sum(
                    [c * jnp.exp(-1j * inputs * f) for f, c in zip(self.f, self.c)],
                    axis=0,
                )

        mts = 2
        freqs = [-3, -1.5, 0, 1.5, 3]
        coeffs = [1, 1, 0, 1, 1]

        fs = max(freqs) * 2 + 1
        model_fct = Model_Fct(coeffs, freqs)

        x = jnp.arange(0, mts * 2 * jnp.pi, 2 * jnp.pi / fs)
        out = model_fct(x)

        X = jnp.fft.fft(out) / out.size

        X_freq = jnp.fft.fftfreq(X.size, 1 / fs)

        if X.size % 2 == 0:
            X = jnp.delete(X, len(X) // 2)
            X_freq = jnp.delete(X_freq, len(X_freq) // 2)

        X_shift = jnp.fft.fftshift(X)
        X_freq_shift = jnp.fft.fftshift(X_freq)

        X2_shift, X2_freq_shift = Coefficients.get_spectrum(
            model_fct, mts=mts, shift=True, trim=True
        )

        assert jnp.allclose(X2_shift, X_shift, atol=1.0e-5), (
            "Model and dummy coefficients are not equal."
        )
        assert jnp.allclose(X2_freq_shift, X_freq_shift, atol=1.0e-5), (
            "Model and dummy frequencies are not equal."
        )

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "output_qubit, output_size, force_mean",
        [
            (-1, 1, True),
            ([0, 1], 1, True),
            (-1, 3, False),
            ([0, 1], 2, False),
        ],
        ids=["all-mean", "subset-mean", "all-no_mean", "subset-no_mean"],
    )
    def test_multi_dim_input(self, output_qubit, output_size, force_mean) -> None:
        model = Model(
            n_qubits=3,
            n_layers=1,
            circuit_type="Hardware_Efficient",
            output_qubit=output_qubit,
            encoding=["RX", "RY"],
            data_reupload=[[[1, 0], [1, 0], [1, 1]]],
        )

        coeffs, freqs = Coefficients.get_spectrum(model, force_mean=force_mean)

        assert coeffs.shape == model.degree or coeffs.shape == (
            *model.degree,
            output_size,
        ), f"Wrong shape of coefficients: {coeffs.shape}, expected {model.degree}"

        ref_input = jnp.array([1, 2, 3, 4])
        exp_model = model(params=None, inputs=ref_input, force_mean=force_mean)
        exp_fourier = Coefficients.evaluate_Fourier_series(
            coefficients=coeffs,
            frequencies=freqs,
            inputs=ref_input,
        )

        assert jnp.isclose(exp_model, exp_fourier, atol=1.0e-5).all(), (
            "Fourier series does not match model expectation"
        )

    @pytest.mark.unittest
    def test_evaluate_fourier_series_accepts_grid_and_flat_spectra(self) -> None:
        model = Model(
            3,
            1,
            "Hardware_Efficient",
            encoding=["RX", "RY"],
            output_qubit=0,
        )
        inputs = np.ones(2)
        exp_model = model(inputs=inputs)

        tree = FourierTree(model)
        tree_coeffs, tree_freqs = tree.get_spectrum(force_mean=False)
        exp_tree = Coefficients.evaluate_Fourier_series(
            coefficients=tree_coeffs[0], frequencies=tree_freqs[0], inputs=inputs
        )

        fft_coeffs, fft_freqs = Coefficients.get_spectrum(model, shift=True)
        exp_fft = Coefficients.evaluate_Fourier_series(
            coefficients=fft_coeffs, frequencies=fft_freqs, inputs=inputs
        )

        assert jnp.isclose(exp_tree, exp_model, atol=1.0e-5)
        assert jnp.isclose(exp_fft, exp_model, atol=1.0e-5)

    @pytest.mark.smoketest
    def test_batch(self) -> None:
        n_samples = 3

        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_15",
            output_qubit=-1,
        )

        random_key = jax.random.key(1000)

        model.initialize_params(random_key, repeat=n_samples)
        random_key, _ = jax.random.split(random_key)
        params = model.params
        coeffs_parallel, _ = Coefficients.get_spectrum(model, shift=True, trim=True)

        # TODO: once the code is ready, test frequency vector as well
        for i in range(n_samples):
            model.params = params[i]
            coeffs_single, _ = Coefficients.get_spectrum(
                model, params=params[i], shift=True, trim=True
            )
            assert jnp.allclose(coeffs_parallel[:, i], coeffs_single, rtol=1.0e-5), (
                "MP and SP coefficients don't match for 1D input"
            )

        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            output_qubit=-1,
            encoding=["RX", "RY"],
        )

        model.initialize_params(random_key, repeat=n_samples)
        params = model.params
        coeffs_parallel, _ = Coefficients.get_spectrum(model, shift=True, trim=True)

        for i in range(n_samples):
            coeffs_single, _ = Coefficients.get_spectrum(
                model, params=params[i], shift=True, trim=True
            )
            assert jnp.allclose(coeffs_parallel[:, :, i], coeffs_single, rtol=1.0e-5), (
                "MP and SP coefficients don't match for 2D input"
            )

    @pytest.mark.unittest
    def test_oversampling_time(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )

        assert Coefficients.get_spectrum(model, mts=3)[0].shape[0] == 15, (
            "Oversampling time failed"
        )

    @pytest.mark.unittest
    def test_oversampling_frequency(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )

        assert Coefficients.get_spectrum(model, mfs=3)[0].shape[0] == 15, (
            "Oversampling frequency failed"
        )

    @pytest.mark.unittest
    def test_shift(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )
        coeffs, freqs = Coefficients.get_spectrum(model, shift=True)

        assert (jnp.abs(coeffs) == jnp.abs(coeffs[::-1])).all(), (
            "Shift failed. Spectrum must be symmetric."
        )

    @pytest.mark.unittest
    def test_trim(self) -> None:
        model = Model(
            n_qubits=3,
            n_layers=1,
            circuit_type="Hardware_Efficient",
            output_qubit=-1,
        )

        coeffs, freqs = Coefficients.get_spectrum(model, mts=2, trim=False)
        coeffs_trimmed, freqs = Coefficients.get_spectrum(model, mts=2, trim=True)

        assert coeffs.size - 1 == coeffs_trimmed.size, (
            f"Wrong shape of coefficients: {coeffs_trimmed.size}, \
            expected {coeffs.size - 1}"
        )

    @pytest.mark.unittest
    def test_frequencies(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )
        coeffs, freqs = Coefficients.get_spectrum(model)

        assert freqs.shape == coeffs.shape, (
            f"(1D) Frequencies ({freqs.shape}) and \
            coefficients ({coeffs.shape}) must have the same length."
        )

        # 2d

        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            encoding=["RX", "RY"],
        )
        coeffs, freqs = Coefficients.get_spectrum(model)

        assert (freqs[0].size * freqs[1].size) == coeffs.size, (
            f"(2D) Frequencies ({freqs.shape}) and \
            coefficients ({coeffs.shape}) must add up to the same length."
        )

        # uneven 2d

        model = Model(
            n_qubits=2,
            n_layers=2,
            circuit_type="Circuit_19",
            encoding=["RX", "RY"],
            data_reupload=[
                [[True, True], [False, True]],
                [[False, True], [True, True]],
            ],
        )
        coeffs, freqs = Coefficients.get_spectrum(model)

        assert (freqs[0].size * freqs[1].size) == coeffs.size, (
            f"(2D) Frequencies ({freqs.shape}) and \
            coefficients ({coeffs.shape}) must add up to the same length."
        )

    @pytest.mark.smoketest
    def test_psd(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )
        coeffs, _ = Coefficients.get_spectrum(model, shift=True)
        _ = Coefficients.get_psd(coeffs)

    @pytest.mark.unittest
    def test_numerical_cap_trims_spectrum(self) -> None:
        """
        With `numerical_cap > 0`, frequencies whose coefficients vanish
        entirely after capping must be removed from both `coeffs` and
        `freqs`, so the returned spectrum stays self-consistent.
        """
        model = Model(
            n_qubits=3,
            n_layers=3,
            circuit_type="Strongly_Entangling",
            encoding=Encoding("hamming", "RZ"),
            output_qubit=-1,
        )
        model.initialize_params(jax.random.key(1000), repeat=20)

        coeffs_full, freqs_full = Coefficients.get_spectrum(
            model, shift=True, trim=True, numerical_cap=-1
        )

        # per-frequency maximum magnitude over the sample axis; a frequency
        # is dropped iff this maximum is below the cap
        sample_axes = tuple(range(1, coeffs_full.ndim))
        per_freq_max = jnp.max(jnp.abs(coeffs_full), axis=sample_axes)
        cap = float(jnp.median(per_freq_max))

        coeffs_cap, freqs_cap = Coefficients.get_spectrum(
            model, shift=True, trim=True, numerical_cap=cap
        )

        # coeffs and freqs stay consistent
        assert coeffs_cap.shape[0] == freqs_cap.shape[0], (
            "Capped coeffs and freqs must have matching length."
        )
        # strictly fewer frequencies survive
        assert freqs_cap.shape[0] < freqs_full.shape[0], (
            "Cap must remove at least one frequency."
        )
        # surviving frequencies are exactly those above the cap
        expected = freqs_full[per_freq_max >= cap]
        assert jnp.array_equal(jnp.sort(freqs_cap), jnp.sort(expected)), (
            "Surviving frequencies must match the above-cap selection."
        )
        # no surviving coefficient vector is all-zero
        assert jnp.all(
            jnp.any(coeffs_cap != 0, axis=tuple(range(1, coeffs_cap.ndim)))
        ), "Surviving coefficients must contain a non-zero entry."
        # the spectrum stays symmetric around zero
        assert jnp.array_equal(jnp.sort(freqs_cap), jnp.sort(-freqs_cap)), (
            "Surviving frequencies must remain symmetric around zero."
        )


class TestFourierTree:
    """Tests for analytical Fourier tree coefficient computation."""

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "circuit_type, n_qubits, n_layers, output_qubit",
        [
            ("Circuit_1", 3, 1, [0, 1]),
            ("Circuit_9", 4, 1, 0),
            ("Circuit_19", 3, 1, 0),
        ],
        ids=["Circuit_1-3q", "Circuit_9-4q", "Circuit_19-3q"],
    )
    def test_coefficients_tree(
        self, circuit_type, n_qubits, n_layers, output_qubit
    ) -> None:
        reference_inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

        model = Model(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            output_qubit=output_qubit,
        )

        fft_coeffs, fft_freqs = Coefficients.get_spectrum(
            model, shift=True, force_mean=False
        )

        coeff_tree = FourierTree(model)
        analytical_coeffs, analytical_freqs = coeff_tree.get_spectrum()
        analytical_coeffs = jnp.stack(analytical_coeffs).T

        assert jnp.isclose(jnp.sum(analytical_coeffs).imag, 0.0, rtol=1.0e-5), (
            "Imaginary part is not zero"
        )

        # Filter fft_coeffs for only the frequencies that occur in the spectrum
        greater_zeros = jnp.invert(jnp.isclose(fft_coeffs, 0.0))
        if greater_zeros.any():
            sel_fft_coeffs = fft_coeffs[greater_zeros]
        else:
            sel_fft_coeffs = jnp.zeros(analytical_coeffs.shape).flatten()

        assert all(
            jnp.isclose(sel_fft_coeffs, analytical_coeffs.flatten(), atol=1.0e-5)
        ), "FFT and analytical coefficients are not equal."

        for ref_input in reference_inputs:
            exp_fourier_fft = Coefficients.evaluate_Fourier_series(
                coefficients=fft_coeffs,
                frequencies=fft_freqs,
                inputs=ref_input,
            )

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=analytical_coeffs,
                frequencies=analytical_freqs[0],
                inputs=ref_input,
            )

            exp_tree = coeff_tree(inputs=ref_input)

            assert jnp.isclose(exp_fourier_fft, exp_fourier, atol=1.0e-5).all(), (
                "FFT and analytical Fourier series do not match"
            )

            assert jnp.isclose(exp_tree, exp_fourier, atol=1.0e-5).all(), (
                "Analytic Fourier series evaluation not working"
            )

    @pytest.mark.unittest
    def test_coefficients_tree_mq(self) -> None:
        reference_inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

        model = Model(
            n_qubits=3,
            n_layers=1,
            circuit_type="Hardware_Efficient",
            output_qubit=-1,
        )

        fft_coeffs, fft_freqs = Coefficients.get_spectrum(model, shift=True)

        coeff_tree = FourierTree(model)
        analytical_coeffs, analytical_freqs = coeff_tree.get_spectrum(force_mean=True)
        analytical_coeffs = jnp.stack(analytical_coeffs).T

        assert jnp.isclose(jnp.sum(analytical_coeffs).imag, 0.0, rtol=1.0e-5), (
            "Imaginary part is not zero"
        )

        # Filter fft_coeffs for only the frequencies that occur in the spectrum
        greater_zeros = jnp.invert(jnp.isclose(fft_coeffs, 0.0))
        if greater_zeros.any():
            sel_fft_coeffs = fft_coeffs[greater_zeros]
        else:
            sel_fft_coeffs = jnp.zeros(analytical_coeffs.shape).flatten()

        assert all(
            jnp.isclose(sel_fft_coeffs, analytical_coeffs.flatten(), atol=1.0e-5)
        ), "FFT and analytical coefficients are not equal."

        for ref_input in reference_inputs:
            exp_fourier_fft = Coefficients.evaluate_Fourier_series(
                coefficients=fft_coeffs,
                frequencies=fft_freqs,
                inputs=ref_input,
            )

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=analytical_coeffs,
                frequencies=analytical_freqs[0],
                inputs=ref_input,
            )

            exp_tree = coeff_tree(inputs=ref_input, force_mean=True)

            assert jnp.isclose(exp_fourier_fft, exp_fourier, atol=1.0e-5), (
                "FFT and analytical Fourier series do not match"
            )

            assert jnp.isclose(exp_tree, exp_fourier, atol=1.0e-5), (
                "Analytic Fourier series evaluation not working"
            )

    @pytest.mark.unittest
    def test_coefficients_tree_multi_feature(self) -> None:
        """Multi-dimensional spectrum: the analytical Fourier series and the
        tree expectation must reproduce the circuit for a 2-feature model."""
        import numpy as np

        model = Model(
            n_qubits=3,
            n_layers=1,
            circuit_type="Circuit_19",
            output_qubit=0,
            encoding=["RX", "RY"],
        )
        assert model.n_input_feat == 2

        tree = FourierTree(model)
        coeffs, freqs = tree.get_spectrum(force_mean=True)
        coeffs = coeffs[0]
        freqs = np.asarray(freqs[0])

        # d-dimensional frequency vectors (n_freq, d).
        assert freqs.ndim == 2 and freqs.shape[1] == 2
        assert jnp.isclose(jnp.sum(coeffs).imag, 0.0, atol=1.0e-5)

        rng = np.random.default_rng(0)
        for _ in range(8):
            x = rng.uniform(-np.pi, np.pi, size=2)
            series = jnp.sum(coeffs * jnp.exp(1j * (freqs @ x)))
            tree_val = tree(inputs=x, force_mean=True)
            model_val = jnp.mean(model(model.params, inputs=jnp.array([x])))

            assert jnp.isclose(jnp.real(series), model_val, atol=1.0e-5), (
                "Multi-feature analytical Fourier series does not match circuit"
            )
            assert jnp.isclose(tree_val, model_val, atol=1.0e-5), (
                "Multi-feature tree evaluation does not match circuit"
            )

    @pytest.mark.unittest
    def test_coefficients_tree_input_scaling(self) -> None:
        """A custom Pauli circuit whose encoding rotations carry heterogeneous
        per-gate data scalings (1, 3, 9) must produce the correctly *scaled*
        Fourier frequencies (not unit counts), match the FFT spectrum, and
        reproduce the circuit output.  The encodings carry no tags: feature and
        scaling are auto-detected from the tape."""
        from qml_essentials.gates import Gates as g
        import qml_essentials.jaqsi as js

        def variational(params, inputs, *args, **kwargs):
            params = params.squeeze()
            g.PauliRot(1 * inputs, "Y", wires=[0])
            g.PauliRot(params[0], "XZX", wires=[0, 1, 2])
            g.PauliRot(3 * inputs, "XZ", wires=[0, 1])
            g.PauliRot(params[1], "YY", wires=[0, 1])
            g.PauliRot(9 * inputs, "XY", wires=[0, 1])

        def prep():
            model = Model(n_qubits=3, n_layers=1, output_qubit=0)
            model._params_shape = (2, 1)
            model.initialize_params()
            model.degree = (27,)  # 2 * max_freq + 1: the FFT sampling grid
            model.script = js.Script(f=variational, n_qubits=3)
            return model

        model = prep()
        expected = {-13, -11, -7, -5, 5, 7, 11, 13}

        # Per-gate scaling recovered from the canonical tape: cols 0/2/4 -> 1/3/9.
        tree = FourierTree(model)
        assert tree.input_scaling[tree.all_input_indices].tolist() == [1, 3, 9]

        coeffs, freqs = tree.get_spectrum()
        coeffs, freqs = np.asarray(coeffs[0]), np.asarray(freqs[0])
        nz = np.abs(coeffs) > 1e-8
        assert set(freqs[nz].astype(int).tolist()) == expected
        assert jnp.isclose(jnp.sum(coeffs).imag, 0.0, atol=1.0e-5)

        # The exact symbolic support agrees.
        assert set(np.asarray(model.exact_spectrum()[0]).tolist()) == expected

        # ... and so does the numerical FFT spectrum.
        fft_coeffs, fft_freqs = Coefficients.get_spectrum(prep(), shift=True)
        fft_nz = np.abs(np.asarray(fft_coeffs)) > 1e-8
        assert set(np.asarray(fft_freqs)[fft_nz].astype(int).tolist()) == expected

        # The 'dp' support method cannot represent per-gate scaling.
        with pytest.raises(NotImplementedError, match="scaling"):
            tree.get_exact_support(method="dp")

        # The analytical Fourier series reproduces the circuit output.
        rng = np.random.default_rng(0)
        for _ in range(6):
            x = float(rng.uniform(-np.pi, np.pi))
            series = np.real(np.sum(coeffs * np.exp(1j * freqs * x)))
            circ = np.asarray(model(model.params, inputs=jnp.array([[x]])))
            assert np.isclose(series, circ.reshape(-1)[0], atol=1.0e-5)

    @pytest.mark.unittest
    def test_coefficients_tree_golomb(self) -> None:
        """FourierTree supports the Golomb diagonal-Hamiltonian encoding.

        The encoding gate ``exp(-i x diag(marks))`` decomposes into commuting
        Pauli-Z rotations whose per-string scalings are rational, while the
        model spectrum is the integer set of Golomb-mark differences.  The
        analytical series must match both the FFT spectrum and the circuit
        output, and must not warn about non-integer scalings."""
        import warnings
        from qml_essentials.unitary import golomb_ruler

        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_1",
            encoding=Encoding("golomb", None),
            output_qubit=0,
            remove_zero_encoding=False,
        )

        # Rational per-string scalings must not trigger a warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tree = FourierTree(model)
            coeffs, freqs = tree.get_spectrum()
        assert not any("scaling" in str(w.message).lower() for w in caught), (
            "unexpected non-integer scaling warning for Golomb encoding"
        )

        coeffs = np.asarray(coeffs[0])
        freqs = np.asarray(freqs[0])
        # Integer model spectrum despite rational intermediate scalings.
        assert np.issubdtype(freqs.dtype, np.integer)

        nz = np.abs(coeffs) > 1e-8
        support = {int(v) for v in freqs[nz]}
        marks = golomb_ruler(2**model.n_qubits)  # (0, 1, 3, 7)
        diffs = {a - b for a in marks for b in marks}
        assert support <= diffs, f"support {support} exceeds Golomb diffs {diffs}"
        assert len(support) > 1, "expected a non-trivial Golomb spectrum"

        # The analytical support agrees with the numerical FFT spectrum.
        fft_coeffs, fft_freqs = Coefficients.get_spectrum(
            model, shift=True, force_mean=False
        )
        fft_nz = np.abs(np.asarray(fft_coeffs)) > 1e-8
        fft_support = {int(v) for v in np.asarray(fft_freqs)[fft_nz]}
        assert support == fft_support

        # Real-valued model -> Hermitian spectrum.
        assert np.isclose(np.sum(coeffs).imag, 0.0, atol=1e-5)

        # The analytical Fourier series reproduces the circuit output.
        rng = np.random.default_rng(0)
        for _ in range(8):
            x = float(rng.uniform(-np.pi, np.pi))
            series = np.real(np.sum(coeffs * np.exp(1j * freqs * x)))
            circ = np.asarray(model(model.params, inputs=jnp.array([[x]])))
            assert np.isclose(series, circ.reshape(-1)[0], atol=1e-5)
            assert np.isclose(
                np.asarray(tree(inputs=x)).reshape(-1)[0], series, atol=1e-5
            )

        # The 'dp' support method cannot represent rational per-gate scalings.
        with pytest.raises(NotImplementedError, match="scaling"):
            tree.get_exact_support(method="dp")

    @pytest.mark.unittest
    def test_coefficients_tree_multi_feature_per_gate_rejected(self) -> None:
        """Auto-detection requires each encoding rotation to be linear in a
        single feature; a rotation mixing two features is rejected."""
        from qml_essentials.gates import Gates as g
        import qml_essentials.jaqsi as js

        def variational(params, inputs, *args, **kwargs):
            params = params.squeeze()
            g.PauliRot(inputs[..., 0] + inputs[..., 1], "X", wires=[0])
            g.PauliRot(params[0], "Z", wires=[0])
            g.PauliRot(params[1], "Z", wires=[1])

        model = Model(n_qubits=2, n_layers=1, output_qubit=0, encoding=["RX", "RY"])
        model._params_shape = (2, 1)
        model.initialize_params()
        model.script = js.Script(f=variational, n_qubits=2)

        with pytest.raises(NotImplementedError, match="single feature"):
            FourierTree(model)

    @pytest.mark.unittest
    def test_fourier_tree_batched_params(self) -> None:
        """Models can carry batched parameters (e.g. after FCC sampling with
        ``initialize_params(repeat=...)``); the tree must fall back to the
        first parameter set instead of feeding batched angles into the gates."""
        import numpy as np

        model = Model(n_qubits=2, n_layers=1, circuit_type="Circuit_19", output_qubit=0)
        model.initialize_params(model.random_key, repeat=4)
        assert model.params.ndim == 3 and model.params.shape[0] == 4
        first_params = model.params[0]

        with pytest.warns(UserWarning, match="batched"):
            tree = FourierTree(model)

        coeffs, freqs = tree.get_spectrum()
        assert set(np.asarray(freqs[0]).tolist()).issubset(
            set(int(v) for v in model.frequencies[0])
        )

        # Evaluation must match the circuit run with the first parameter set.
        with pytest.warns(UserWarning, match="batched"):
            val_tree = tree(inputs=0.5)
        val_model = model(first_params, inputs=jnp.array([[0.5]]))
        assert jnp.isclose(val_tree, val_model, atol=1.0e-5).all()

    @pytest.mark.unittest
    def test_exact_support_dp_matches_tree(self) -> None:
        """The merged-state DP support equals the fully exact tree support
        on circuits without cross-path cancellations."""
        import numpy as np

        for circuit_type in ["Circuit_19", "Hardware_Efficient"]:
            model = Model(
                n_qubits=3, n_layers=1, circuit_type=circuit_type, output_qubit=0
            )
            tree = FourierTree(model)
            sup_tree = tree.get_exact_support(method="tree")
            sup_dp = tree.get_exact_support(method="dp")
            for st, sd in zip(sup_tree, sup_dp):
                assert set(np.asarray(st).tolist()) == set(np.asarray(sd).tolist()), (
                    f"DP and tree supports differ for {circuit_type}"
                )

    @pytest.mark.unittest
    def test_exact_support_dp_upper_bound(self) -> None:
        """On directly repeated encodings the tree detects the cross-path
        cancellation at omega=0 while the DP yields the structural superset."""
        import numpy as np

        model = Model(
            n_qubits=1,
            n_layers=2,
            circuit_type="No_Ansatz",
            data_reupload=True,
            encoding="RX",
            output_qubit=0,
        )
        tree = FourierTree(model)
        sup_tree = set(np.asarray(tree.get_exact_support("tree")[0]).tolist())
        sup_dp = set(np.asarray(tree.get_exact_support("dp")[0]).tolist())

        assert sup_tree == {-2, 2}
        assert sup_dp == {-2, 0, 2}
        assert sup_tree.issubset(sup_dp)

    @pytest.mark.unittest
    def test_exact_support_dp_deep_circuit(self) -> None:
        """For deep entangling circuits the explicit tree is combinatorially
        infeasible; the DP support stays cheap and must contain every
        FFT-significant frequency while being a subset of the naive grid."""
        import numpy as np

        model = Model(
            n_qubits=4,
            n_layers=4,
            circuit_type="Strongly_Entangling",
            output_qubit=0,
        )
        tree = FourierTree(model)
        union = set()
        for sup in tree.get_exact_support(method="dp"):
            union |= set(np.asarray(sup).tolist())

        naive = set(int(v) for v in model.frequencies[0])
        assert union.issubset(naive)

        fft_coeffs, fft_freqs = Coefficients.get_spectrum(
            model, shift=True, force_mean=True
        )
        significant = {
            int(f)
            for f, c in zip(
                np.asarray(fft_freqs).ravel(), np.asarray(fft_coeffs).ravel()
            )
            if abs(c) > 1e-4
        }
        assert significant.issubset(union)

    @pytest.mark.unittest
    def test_exact_support_dp_matches_tree_multi_feature(self) -> None:
        """The merged-state DP support equals the fully exact tree support for
        multi-feature models (no cross-path cancellations on these circuits).
        Supports are (n_freq, d) frequency-vector arrays, compared as row sets."""
        import numpy as np

        for circuit_type in ["Circuit_19", "Hardware_Efficient"]:
            model = Model(
                n_qubits=3,
                n_layers=1,
                circuit_type=circuit_type,
                output_qubit=0,
                encoding=["RX", "RY"],
            )
            assert model.n_input_feat == 2

            tree = FourierTree(model)
            sup_tree = tree.get_exact_support(method="tree")
            sup_dp = tree.get_exact_support(method="dp")
            for st, sd in zip(sup_tree, sup_dp):
                st, sd = np.asarray(st), np.asarray(sd)
                assert st.ndim == 2 and st.shape[1] == 2
                assert sd.ndim == 2 and sd.shape[1] == 2
                tree_rows = set(map(tuple, st.tolist()))
                dp_rows = set(map(tuple, sd.tolist()))
                assert tree_rows == dp_rows, (
                    f"DP and tree multi-feature supports differ for {circuit_type}"
                )

            # exact_spectrum threads the multi-feature dp support through to a
            # per-feature tuple, matching the tree method per axis.
            spec_dp = model.exact_spectrum(method="dp")
            spec_tree = model.exact_spectrum(method="tree")
            assert len(spec_dp) == 2
            for sd, st, naive in zip(spec_dp, spec_tree, model.frequencies):
                assert set(np.asarray(sd).tolist()) == set(np.asarray(st).tolist())
                assert set(np.asarray(sd).tolist()).issubset(
                    {int(v) for v in np.asarray(naive)}
                )


class TestFCC:
    """Tests for Fourier Coefficient Correlation (FCC) computation."""

    @pytest.mark.unittest
    def test_pearson_correlation(self) -> None:
        N = 1000
        K = 5
        seed = 1000
        rng = np.random.default_rng(seed)

        # create a random array of shape N, K
        coeffs = rng.normal(size=(N, K))
        pearson = FCC._pearson(coeffs)

        for i in range(coeffs.shape[1]):
            for j in range(coeffs.shape[1]):
                reference = pearsonr(coeffs[:, i], coeffs[:, j]).correlation
                assert jnp.isclose(pearson[i, j], reference, atol=1.0e-5), (
                    f"Pearson correlation does not match reference. "
                    f"For index {i}, {j}, got {pearson[i, j]}, expected {reference}"
                )

    @pytest.mark.unittest
    def test_spearman_correlation(self) -> None:
        N = 1000
        K = 5
        seed = 1000
        rng = np.random.default_rng(seed)

        # create a random array of shape N, K
        coeffs = rng.normal(size=(N, K))
        spearman = FCC._spearman(coeffs)

        for i in range(coeffs.shape[1]):
            for j in range(coeffs.shape[1]):
                reference = spearmanr(coeffs[:, i], coeffs[:, j]).correlation
                assert jnp.isclose(spearman[i, j], reference, atol=1.0e-5), (
                    f"Spearman correlation does not match reference. "
                    f"For index {i}, {j}, got {spearman[i, j]}, expected {reference}"
                )

    @pytest.mark.unittest
    def test_pearson_correlation_complex(self) -> None:
        """Pearson on complex input should match scipy on stacked real/imag."""
        N = 1000
        K = 5
        seed = 42
        rng = np.random.default_rng(seed)

        coeffs = rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K))
        pearson = FCC._pearson(jnp.array(coeffs))

        # Reference: stack real and imag along sample axis, then use scipy
        stacked = np.concatenate([coeffs.real, coeffs.imag], axis=0)
        for i in range(K):
            for j in range(K):
                reference = pearsonr(stacked[:, i], stacked[:, j]).correlation
                assert jnp.isclose(pearson[i, j], reference, atol=1.0e-5), (
                    f"Complex Pearson mismatch at ({i},{j}): "
                    f"got {pearson[i, j]}, expected {reference}"
                )

    @pytest.mark.unittest
    def test_complex_pearson_correlation(self) -> None:
        """Complex Pearson should match Hermitian normalized covariance."""
        N = 1000
        K = 5
        seed = 314
        rng = np.random.default_rng(seed)

        coeffs = rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K))
        coeffs[0, 1] = np.nan + 0.0j
        coeffs[1, 2] = np.inf + 0.0j

        complex_pearson = FCC._complex_pearson(jnp.array(coeffs))

        for i in range(K):
            for j in range(K):
                valid = np.isfinite(coeffs[:, i]) & np.isfinite(coeffs[:, j])
                x = coeffs[valid, i]
                y = coeffs[valid, j]
                x_centered = x - np.mean(x)
                y_centered = y - np.mean(y)
                denom = np.sqrt(
                    np.sum(np.abs(x_centered) ** 2) * np.sum(np.abs(y_centered) ** 2)
                )
                reference = (
                    np.sum(np.conj(x_centered) * y_centered) / denom
                    if denom > 0
                    else np.nan
                )

                assert jnp.isclose(complex_pearson[i, j], reference, atol=1.0e-5), (
                    f"Complex Pearson mismatch at ({i},{j}): "
                    f"got {complex_pearson[i, j]}, expected {reference}"
                )

    @pytest.mark.unittest
    def test_complex_pearson_phase(self) -> None:
        """Complex Pearson magnitude tracks correlation and angle tracks phase."""
        N = 200
        seed = 2718
        phase = 0.37
        rng = np.random.default_rng(seed)

        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        y = np.exp(1j * phase) * x
        coeffs = jnp.stack([jnp.array(x), jnp.array(y)], axis=1)

        corr = FCC._complex_pearson(coeffs)
        corr_from_method = FCC._correlate(coeffs, method="complex_pearson")

        assert jnp.allclose(corr, corr_from_method, atol=1.0e-10)
        assert jnp.isclose(jnp.abs(corr[0, 1]), 1.0, atol=1.0e-10)
        assert jnp.isclose(jnp.angle(corr[0, 1]), phase, atol=1.0e-10)
        assert jnp.isclose(jnp.angle(corr[1, 0]), -phase, atol=1.0e-10)

    @pytest.mark.unittest
    def test_covariance_correlation(self) -> None:
        """Covariance should match the Hermitian pairwise sample covariance and
        be reachable via `method="covariance"`."""
        N = 1000
        K = 5
        seed = 314
        rng = np.random.default_rng(seed)

        coeffs = rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K))
        coeffs[0, 1] = np.nan + 0.0j
        coeffs[1, 2] = np.inf + 0.0j

        covariance = FCC._covariance(jnp.array(coeffs))
        cov_from_method = FCC._correlate(jnp.array(coeffs), method="covariance")

        assert jnp.allclose(
            covariance, cov_from_method, atol=1.0e-10, equal_nan=True
        ), "method='covariance' must dispatch to _covariance."

        for i in range(K):
            for j in range(K):
                valid = np.isfinite(coeffs[:, i]) & np.isfinite(coeffs[:, j])
                x = coeffs[valid, i]
                y = coeffs[valid, j]
                x_centered = x - np.mean(x)
                y_centered = y - np.mean(y)
                reference = np.sum(np.conj(x_centered) * y_centered) / (len(x) - 1)

                assert jnp.isclose(covariance[i, j], reference, atol=1.0e-5), (
                    f"Covariance mismatch at ({i},{j}): "
                    f"got {covariance[i, j]}, expected {reference}"
                )

    @pytest.mark.unittest
    def test_spearman_correlation_complex(self) -> None:
        """Spearman on complex input should match scipy on stacked real/imag."""
        N = 1000
        K = 5
        seed = 42
        rng = np.random.default_rng(seed)

        coeffs = rng.normal(size=(N, K)) + 1j * rng.normal(size=(N, K))
        spearman = FCC._spearman(jnp.array(coeffs))

        # Reference: stack real and imag along sample axis, then use scipy
        stacked = np.concatenate([coeffs.real, coeffs.imag], axis=0)
        for i in range(K):
            for j in range(K):
                reference = spearmanr(stacked[:, i], stacked[:, j]).correlation
                assert jnp.isclose(spearman[i, j], reference, atol=1.0e-5), (
                    f"Complex Spearman mismatch at ({i},{j}): "
                    f"got {spearman[i, j]}, expected {reference}"
                )

    @pytest.mark.unittest
    def test_pearson_complex_preserves_imaginary(self) -> None:
        """Ensure complex correlations differ from real-only correlations,
        i.e. the imaginary part is not silently discarded."""
        N = 200
        K = 4
        seed = 123
        rng = np.random.default_rng(seed)

        real_part = rng.normal(size=(N, K))
        imag_part = rng.normal(size=(N, K))
        coeffs_complex = jnp.array(real_part + 1j * imag_part)
        coeffs_real_only = jnp.array(real_part)

        corr_complex = FCC._pearson(coeffs_complex)
        corr_real = FCC._pearson(coeffs_real_only)

        # They should generally differ (imaginary part contributes)
        assert not jnp.allclose(corr_complex, corr_real, atol=1.0e-3), (
            "Complex and real-only Pearson correlations should differ "
            "when imaginary components carry information."
        )

    @pytest.mark.unittest
    def test_weighting_mean(self) -> None:
        """Mean weighting should match the coefficient order used by correlation."""
        fourier_fingerprint = jnp.arange(16, dtype=float).reshape(4, 4)
        coeffs = jnp.array(
            [
                [[1.0, 3.0], [-2.0, 4.0]],
                [[5.0, 7.0], [8.0, 10.0]],
            ]
        )

        coefficient_means = jnp.abs(jnp.mean(coeffs, axis=-1)).T.reshape(-1)
        expected = fourier_fingerprint * jnp.outer(coefficient_means, coefficient_means)

        assert jnp.allclose(FCC._weighting_mean(fourier_fingerprint, coeffs), expected)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "circuit_type, expected_fcc",
        [
            ("Circuit_20", 0.004),
            ("Circuit_19", 0.010),
            ("Circuit_17", 0.078),
            ("Hardware_Efficient", 0.080),
        ],
        ids=["Circuit_20", "Circuit_19", "Circuit_17", "Hardware_Efficient"],
    )
    def test_fcc(self, circuit_type, expected_fcc) -> None:
        """
        This test replicates the results obtained for the FCC
        as shown in Fig. 3a from the paper
        "Fourier Fingerprints of Ansatzes in Quantum Machine Learning"
        https://doi.org/10.48550/arXiv.2508.20868
        """
        model = Model(
            n_qubits=6,
            n_layers=1,
            circuit_type=circuit_type,
            output_qubit=-1,
            encoding=["RY"],
        )
        fcc = FCC.get_fcc(model=model, n_samples=500, scale=True)

        assert jnp.isclose(fcc, expected_fcc, atol=3.0e-2), (
            f"Wrong FCC for {circuit_type}. Got {fcc}, expected {expected_fcc}."
        )

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "encoding_strategy, circuit_type, n_qubits, n_layers, n_samples",
        [
            ("hamming", "Circuit_2", 2, 2, 5),
            ("binary", "Circuit_2", 2, 2, 5),
            ("ternary", "Circuit_2", 2, 2, 5),
        ],
    )
    def test_fcc_encoding_strategies(
        self, encoding_strategy, circuit_type, n_qubits, n_layers, n_samples
    ) -> None:
        """
        Test that the FCC is bounded in [0, 1] for Hamming, Binary,
        and Ternary encoding strategies.
        """
        enc = Encoding(encoding_strategy, "RX")
        model = Model(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            encoding=enc,
            output_qubit=-1,
        )

        # Verify spectrum computation succeeds
        model.initialize_params(repeat=n_samples)
        coeffs, freqs = Coefficients.get_spectrum(
            model, shift=True, trim=True, force_mean=True, execution_type="expval"
        )
        assert coeffs.shape[0] == model.degree[0], (
            f"Wrong number of coefficients for {encoding_strategy}: "
            f"{coeffs.shape[0]}, expected {model.degree[0]}"
        )

        # Verify correlation values are bounded
        fp_pearson = FCC._correlate(coeffs.transpose(), method="pearson")
        assert np.all(np.abs(fp_pearson[np.isfinite(fp_pearson)]) <= 1.0 + 1e-10), (
            f"Pearson correlation out of [-1, 1] for {encoding_strategy}. "
            f"Max |r| = {np.max(np.abs(fp_pearson[np.isfinite(fp_pearson)]))}"
        )

        fp_spearman = FCC._correlate(coeffs.transpose(), method="spearman")
        assert np.all(np.abs(fp_spearman[np.isfinite(fp_spearman)]) <= 1.0 + 1e-10), (
            f"Spearman correlation out of [-1, 1] for {encoding_strategy}. "
            f"Max |rho| = {np.max(np.abs(fp_spearman[np.isfinite(fp_spearman)]))}"
        )

        # Verify FCC is bounded in [0, 1] with both methods
        for method in ["pearson", "spearman"]:
            fcc = FCC.get_fcc(
                model, n_samples=n_samples, method=method, trim_redundant=True
            )
            assert 0.0 <= float(fcc) <= 1.0, (
                f"FCC out of [0, 1] for {encoding_strategy}/{method}: {float(fcc)}"
            )

            # Also test without trimming
            fcc_untrimmed = FCC.get_fcc(
                model, n_samples=n_samples, method=method, trim_redundant=False
            )
            assert 0.0 <= float(fcc_untrimmed) <= 1.0, (
                f"FCC (untrimmed) out of [0, 1] for "
                f"{encoding_strategy}/{method}: {float(fcc_untrimmed)}"
            )

    @pytest.mark.smoketest
    @pytest.mark.parametrize(
        "circuit_type",
        ["Circuit_20", "Circuit_19", "Circuit_17", "Hardware_Efficient"],
    )
    def test_fourier_fingerprint(self, circuit_type) -> None:
        """
        This test checks if the calculation of the Fourier fingerprint
        returns the expected result by using hashs.
        """
        model = Model(
            n_qubits=4,
            n_layers=1,
            circuit_type=circuit_type,
            output_qubit=-1,
            encoding=["RY"],
        )
        _ = FCC.get_fourier_fingerprint(
            model=model,
            n_samples=500,
            scale=True,
        )

    @pytest.mark.unittest
    def test_fcc_2d(self) -> None:
        """
        This test replicates the results obtained for the FCC
        as shown in Fig. 3b from the paper
        "Fourier Fingerprints of Ansatzes in Quantum Machine Learning"
        https://doi.org/10.48550/arXiv.2508.20868

        Note that we only test one circuit here with and also with a lower
        number of qubits, because it get's computationally too expensive
        otherwise.
        """
        model = Model(
            n_qubits=4,
            n_layers=1,
            circuit_type="Circuit_19",
            output_qubit=-1,
            encoding=["RX", "RY"],
        )
        fcc = FCC.get_fcc(
            model=model,
            n_samples=250,
            scale=True,
        )
        assert jnp.isclose(fcc, 0.016, atol=2.0e-3), (
            f"Wrong FCC for Circuit_19. Got {fcc}, expected 0.020."
        )

    @pytest.mark.unittest
    @pytest.mark.parametrize("weight", [False, True], ids=["fast", "weighted"])
    def test_fingerprint_freqs_match_matrix(self, weight) -> None:
        """
        With `numerical_cap`, the frequencies returned by
        `get_fourier_fingerprint` must label the trimmed matrix axes 1:1,
        i.e. a `(row_freqs, col_freqs)` tuple matching the matrix shape.
        """
        model = Model(
            n_qubits=3,
            n_layers=3,
            circuit_type="Strongly_Entangling",
            encoding=Encoding("hamming", "RZ"),
            output_qubit=-1,
        )
        matrix, freqs = FCC.get_fourier_fingerprint(
            model=model,
            n_samples=50,
            random_key=jax.random.key(1000),
            weight=weight,
            numerical_cap=1e-10,
        )

        assert isinstance(freqs, tuple) and len(freqs) == 2, (
            "Trimmed fingerprint must return a (row_freqs, col_freqs) tuple."
        )
        row_freqs, col_freqs = freqs
        assert row_freqs.shape[0] == matrix.shape[0], (
            f"Row freqs ({row_freqs.shape[0]}) must match matrix rows "
            f"({matrix.shape[0]})."
        )
        assert col_freqs.shape[0] == matrix.shape[1], (
            f"Col freqs ({col_freqs.shape[0]}) must match matrix cols "
            f"({matrix.shape[1]})."
        )

    @pytest.mark.unittest
    def test_fingerprint_freqs_match_matrix_2d(self) -> None:
        """
        The matrix-axis frequency alignment also holds for multi-dimensional
        input, where each axis label is a per-coefficient frequency tuple.
        """
        model = Model(
            n_qubits=3,
            n_layers=2,
            circuit_type="Strongly_Entangling",
            encoding=["RX", "RY"],
            output_qubit=-1,
        )
        matrix, freqs = FCC.get_fourier_fingerprint(
            model=model,
            n_samples=50,
            random_key=jax.random.key(1000),
            numerical_cap=1e-10,
        )

        assert isinstance(freqs, tuple) and len(freqs) == 2
        row_freqs, col_freqs = freqs
        assert row_freqs.shape[0] == matrix.shape[0]
        assert col_freqs.shape[0] == matrix.shape[1]
        # each axis label is a (f_x, f_y) frequency tuple
        assert row_freqs.shape[1] == model.n_input_feat
        assert col_freqs.shape[1] == model.n_input_feat

    @pytest.mark.unittest
    def test_weighting(self) -> None:
        """
        This test replicates the results obtained for the FCC
        as shown in Fig. 3b from the paper
        "Fourier Fingerprints of Ansatzes in Quantum Machine Learning"
        https://doi.org/10.48550/arXiv.2508.20868

        Note that we only test one circuit here with and also with a lower
        number of qubits, because it get's computationally too expensive
        otherwise.
        """
        model = Model(
            n_qubits=3,
            n_layers=1,
            circuit_type="Circuit_19",
            output_qubit=-1,
            encoding=["RY"],
        )
        fcc_weight = FCC.get_fcc(
            model=model,
            n_samples=500,
            scale=True,
            weight=True,
        )
        fcc_no_weight = FCC.get_fcc(
            model=model,
            n_samples=500,
            scale=True,
            weight=False,
        )
        assert fcc_weight < fcc_no_weight, (
            "Weighted FCC should be substantially smaller for degenerate circuits."
        )


class TestDatasets:
    """Tests for Fourier series dataset generation."""

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "n_input_feat, encoding_strategy, coefficients_min, coefficients_max, "
        "zero_centered",
        [
            (2, "hamming", 0.0, 1.0, False),
            (3, "hamming", 0.0, 1.0, False),
            (1, "hamming", 0.1, 0.9, False),
            (1, "hamming", 0.0, 1.0, True),
            (1, "binary", 0.0, 1.0, False),
            (1, "ternary", 0.0, 1.0, False),
        ],
        ids=[
            "2d-hamming",
            "3d-hamming",
            "custom-coeff-range",
            "zero-centered",
            "binary-encoding",
            "ternary-encoding",
        ],
    )
    def test_fourier_series_dataset(
        self,
        n_input_feat,
        encoding_strategy,
        coefficients_min,
        coefficients_max,
        zero_centered,
    ) -> None:
        n_samples = 100
        seed = 1000
        random_key = jax.random.key(seed)

        encoding = Encoding(
            encoding_strategy,
            ["RY" for _ in range(n_input_feat)],
        )

        model = Model(
            n_qubits=2,
            n_layers=1,
            encoding=encoding,
        )

        # Build kwargs for generate_fourier_series, only including
        # non-default values
        generate_kwargs = {}
        if coefficients_min != 0.0:
            generate_kwargs["coefficients_min"] = coefficients_min
        if coefficients_max != 1.0:
            generate_kwargs["coefficients_max"] = coefficients_max
        if zero_centered:
            generate_kwargs["zero_centered"] = zero_centered

        all_domain_samples = []
        all_fourier_samples = []
        all_coefficients = []

        for i in range(n_samples):
            try:
                domain_samples, fourier_samples, coefficients = (
                    Datasets.generate_fourier_series(
                        random_key,
                        model=model,
                        **generate_kwargs,
                    )
                )
                random_key, _ = jax.random.split(random_key)
            except Exception as e:
                tb = traceback.format_exc()
                raise Exception(f"Error in iteration {i}: {e}\n{tb}")

            # Sanity check to ensure the FFT is correct
            coefficients_hat = jnp.fft.fftshift(
                jnp.fft.fftn(
                    fourier_samples,
                    axes=list(range(model.n_input_feat)),
                )
            )
            assert jnp.allclose(
                coefficients,
                coefficients_hat,
                atol=1e-6,
            ), "Frequencies don't match"

            assert jnp.all(
                domain_samples.shape
                == (
                    *model.degree,
                    model.n_input_feat,
                )
            ), f"Wrong shape of domain samples: {domain_samples.shape}"
            assert jnp.all(fourier_samples.shape == model.degree), (
                f"Wrong shape of Fourier values: {fourier_samples.shape}"
            )
            assert jnp.all(coefficients.shape == model.degree), (
                f"Wrong shape of coefficients: {coefficients.shape}"
            )

            all_domain_samples.append(domain_samples)
            all_fourier_samples.append(fourier_samples)
            all_coefficients.append(coefficients)

        all_domain_samples = jnp.array(all_domain_samples)
        all_fourier_samples = jnp.array(all_fourier_samples)
        all_coefficients = jnp.array(all_coefficients)

        if not zero_centered:
            assert jnp.sqrt(coefficients_min) <= jnp.min(jnp.abs(coefficients)), (
                "Coefficients are too small"
            )
            assert jnp.sqrt(coefficients_max) >= jnp.max(jnp.abs(coefficients)), (
                "Coefficients are too large"
            )
        else:
            assert jnp.isclose(fourier_samples.mean(), 0.0, atol=1e-1), (
                "Zero centering failed"
            )
