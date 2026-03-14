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
        assert jnp.isclose(
            jnp.sum(coeffs).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

        partial_circuit = partial(model, model.params, force_mean=True)
        ref_coeffs = pcoefficients(partial_circuit, 1, model.degree[0] // 2)

        assert jnp.allclose(
            coeffs, ref_coeffs, rtol=1.0e-5
        ), "Coefficients don't match the pennylane reference"

        for ref_input in reference_inputs:
            exp_model = model(params=None, inputs=ref_input, force_mean=True)

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=coeffs,
                frequencies=freqs,
                inputs=ref_input,
            )

            assert jnp.isclose(
                exp_model, exp_fourier, atol=1.0e-5
            ), "Fourier series does not match model expectation"

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

        assert jnp.allclose(
            X2_shift, X_shift, atol=1.0e-5
        ), "Model and dummy coefficients are not equal."
        assert jnp.allclose(
            X2_freq_shift, X_freq_shift, atol=1.0e-5
        ), "Model and dummy frequencies are not equal."

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

        assert jnp.isclose(
            exp_model, exp_fourier, atol=1.0e-5
        ).all(), "Fourier series does not match model expectation"

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
            assert jnp.allclose(
                coeffs_parallel[:, i], coeffs_single, rtol=1.0e-5
            ), "MP and SP coefficients don't match for 1D input"

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
            assert jnp.allclose(
                coeffs_parallel[:, :, i], coeffs_single, rtol=1.0e-5
            ), "MP and SP coefficients don't match for 2D input"

    @pytest.mark.unittest
    def test_oversampling_time(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )

        assert (
            Coefficients.get_spectrum(model, mts=3)[0].shape[0] == 15
        ), "Oversampling time failed"

    @pytest.mark.unittest
    def test_oversampling_frequency(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )

        assert (
            Coefficients.get_spectrum(model, mfs=3)[0].shape[0] == 15
        ), "Oversampling frequency failed"

    @pytest.mark.unittest
    def test_shift(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )
        coeffs, freqs = Coefficients.get_spectrum(model, shift=True)

        assert (
            jnp.abs(coeffs) == jnp.abs(coeffs[::-1])
        ).all(), "Shift failed. Spectrum must be symmetric."

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

        assert (
            coeffs.size - 1 == coeffs_trimmed.size
        ), f"Wrong shape of coefficients: {coeffs_trimmed.size}, \
            expected {coeffs.size - 1}"

    @pytest.mark.unittest
    def test_frequencies(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )
        coeffs, freqs = Coefficients.get_spectrum(model)

        assert (
            freqs.shape == coeffs.shape
        ), f"(1D) Frequencies ({freqs.shape}) and \
            coefficients ({coeffs.shape}) must have the same length."

        # 2d

        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            encoding=["RX", "RY"],
        )
        coeffs, freqs = Coefficients.get_spectrum(model)

        assert (
            freqs[0].size * freqs[1].size
        ) == coeffs.size, f"(2D) Frequencies ({freqs.shape}) and \
            coefficients ({coeffs.shape}) must add up to the same length."

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

        assert (
            freqs[0].size * freqs[1].size
        ) == coeffs.size, f"(2D) Frequencies ({freqs.shape}) and \
            coefficients ({coeffs.shape}) must add up to the same length."

    @pytest.mark.smoketest
    def test_psd(self) -> None:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )
        coeffs, _ = Coefficients.get_spectrum(model, shift=True)
        _ = Coefficients.get_psd(coeffs)


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

        assert jnp.isclose(
            jnp.sum(analytical_coeffs).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

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

            assert jnp.isclose(
                exp_fourier_fft, exp_fourier, atol=1.0e-5
            ).all(), "FFT and analytical Fourier series do not match"

            assert jnp.isclose(
                exp_tree, exp_fourier, atol=1.0e-5
            ).all(), "Analytic Fourier series evaluation not working"

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

        assert jnp.isclose(
            jnp.sum(analytical_coeffs).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

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

            assert jnp.isclose(
                exp_fourier_fft, exp_fourier, atol=1.0e-5
            ), "FFT and analytical Fourier series do not match"

            assert jnp.isclose(
                exp_tree, exp_fourier, atol=1.0e-5
            ), "Analytic Fourier series evaluation not working"


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
    @pytest.mark.parametrize(
        "circuit_type, expected_fcc",
        [
            ("Circuit_20", 0.004),
            ("Circuit_19", 0.010),
            ("Circuit_17", 0.530),
            ("Hardware_Efficient", 0.715),
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
        fcc = FCC.get_fcc(
            model=model, n_samples=500, seed=1000, scale=True, numerical_cap=1e-10
        )

        assert jnp.isclose(
            fcc, expected_fcc, atol=3.0e-2
        ), f"Wrong FCC for {circuit_type}. Got {fcc}, expected {expected_fcc}."

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
            seed=1000,
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
            seed=1000,
            scale=True,
        )
        assert jnp.isclose(
            fcc, 0.020, atol=2.0e-3
        ), f"Wrong FCC for Circuit_19. Got {fcc}, expected 0.020."

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
        fcc = FCC.get_fcc(
            model=model,
            n_samples=500,
            seed=1000,
            scale=True,
            weight=True,
        )
        assert jnp.isclose(
            fcc, 0.010, atol=5.0e-3
        ), f"Wrong FCC for Circuit_19. Got {fcc}, expected 0.010."


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
            assert jnp.all(
                fourier_samples.shape == model.degree
            ), f"Wrong shape of Fourier values: {fourier_samples.shape}"
            assert jnp.all(
                coefficients.shape == model.degree
            ), f"Wrong shape of coefficients: {coefficients.shape}"

            all_domain_samples.append(domain_samples)
            all_fourier_samples.append(fourier_samples)
            all_coefficients.append(coefficients)

        all_domain_samples = jnp.array(all_domain_samples)
        all_fourier_samples = jnp.array(all_fourier_samples)
        all_coefficients = jnp.array(all_coefficients)

        if not zero_centered:
            assert jnp.sqrt(coefficients_min) <= jnp.min(
                jnp.abs(coefficients)
            ), "Coefficients are too small"
            assert jnp.sqrt(coefficients_max) >= jnp.max(
                jnp.abs(coefficients)
            ), "Coefficients are too large"
        else:
            assert jnp.isclose(
                fourier_samples.mean(), 0.0, atol=1e-1
            ), "Zero centering failed"
