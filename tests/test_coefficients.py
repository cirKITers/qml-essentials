from qml_essentials.model import Model
from qml_essentials.ansaetze import Encoding
from qml_essentials.coefficients import Coefficients, FourierTree, FCC, Datasets
from pennylane.fourier import coefficients as pcoefficients
import hashlib

import traceback
import numpy as np
import pennylane.numpy as pnp
import logging
import pytest
from scipy.stats import pearsonr, spearmanr

from functools import partial


logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_coefficients() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 3,
            "n_layers": 1,
            "output_qubit": [0, 1],
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_19",
            "n_qubits": 5,
            "n_layers": 1,
            "output_qubit": 0,
        },
    ]
    reference_inputs = np.linspace(-np.pi, np.pi, 10)

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            output_qubit=test_case["output_qubit"],
        )

        coeffs, freqs = Coefficients.get_spectrum(model)

        assert len(coeffs) == model.degree[0], "Wrong number of coefficients"
        assert np.isclose(
            np.sum(coeffs).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

        partial_circuit = partial(model, model.params)
        ref_coeffs = pcoefficients(partial_circuit, 1, model.degree[0] // 2)

        assert np.allclose(
            coeffs, ref_coeffs, rtol=1.0e-5
        ), "Coefficients don't match the pennylane reference"

        for ref_input in reference_inputs:
            exp_model = model(params=None, inputs=ref_input)

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=coeffs,
                frequencies=freqs,
                inputs=ref_input,
            )

            assert np.isclose(
                exp_model, exp_fourier, atol=1.0e-5
            ), "Fourier series does not match model expectation"


@pytest.mark.unittest
def test_multi_dim_input() -> None:
    model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        output_qubit=-1,
        encoding=["RX", "RX"],
        data_reupload=[[[1, 0], [1, 0], [1, 1]]],
    )

    coeffs, freqs = Coefficients.get_spectrum(model)

    assert (
        coeffs.shape == [model.degree[i]] for i in range(model.n_input_feat)
    ), f"Wrong shape of coefficients: {coeffs.shape}, \
        expected {[[model.degee[i]] for i in range(model.n_input_feat)]}"

    ref_input = [1, 2]
    exp_model = model(params=None, inputs=ref_input, force_mean=True)
    exp_fourier = Coefficients.evaluate_Fourier_series(
        coefficients=coeffs,
        frequencies=freqs,
        inputs=ref_input,
    )

    assert np.isclose(
        exp_model, exp_fourier, atol=1.0e-5
    ), "Fourier series does not match model expectation"


@pytest.mark.smoketest
def test_batch() -> None:
    n_samples = 3

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_15",
        output_qubit=-1,
        mp_threshold=100,
        initialization="random",
    )

    model.initialize_params(rng=pnp.random.default_rng(1000), repeat=n_samples)
    params = model.params
    coeffs_parallel, _ = Coefficients.get_spectrum(model, shift=True, trim=True)

    # TODO: once the code is ready, test frequency vector as well
    for i in range(n_samples):
        model.params = params[:, :, i]
        coeffs_single, _ = Coefficients.get_spectrum(model, shift=True, trim=True)
        assert np.allclose(
            coeffs_parallel[:, i], coeffs_single, rtol=1.0e-5
        ), "MP and SP coefficients don't match for 1D input"

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        output_qubit=-1,
        mp_threshold=100,
        encoding=["RX", "RY"],
        initialization="random",
    )

    model.initialize_params(rng=pnp.random.default_rng(1000), repeat=n_samples)
    params = model.params
    coeffs_parallel, _ = Coefficients.get_spectrum(model, shift=True, trim=True)

    for i in range(n_samples):
        model.params = params[:, :, i]
        coeffs_single, _ = Coefficients.get_spectrum(model, shift=True, trim=True)
        assert np.allclose(
            coeffs_parallel[:, :, i], coeffs_single, rtol=1.0e-5
        ), "MP and SP coefficients don't match for 2D input"


@pytest.mark.unittest
def test_coefficients_tree() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 3,
            "n_layers": 1,
            "output_qubit": [0, 1],
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_19",
            "n_qubits": 3,
            "n_layers": 1,
            "output_qubit": 0,
        },
    ]

    reference_inputs = np.linspace(-np.pi, np.pi, 10)
    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            output_qubit=test_case["output_qubit"],
            as_pauli_circuit=False,
        )

        fft_coeffs, fft_freqs = Coefficients.get_spectrum(model, shift=True)

        coeff_tree = FourierTree(model)
        analytical_coeffs, analytical_freqs = coeff_tree.get_spectrum()

        assert len(analytical_freqs[0]) == len(
            analytical_freqs[0]
        ), "Wrong number of frequencies"
        assert np.isclose(
            np.sum(analytical_coeffs[0]).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

        # Filter fft_coeffs for only the frequencies that occur in the spectrum
        sel_fft_coeffs = np.take(fft_coeffs, analytical_freqs[0] + int(fft_freqs.max()))
        assert all(
            np.isclose(sel_fft_coeffs, analytical_coeffs[0], atol=1.0e-5)
        ), "FFT and analytical coefficients are not equal."

        for ref_input in reference_inputs:
            exp_fourier_fft = Coefficients.evaluate_Fourier_series(
                coefficients=fft_coeffs,
                frequencies=fft_freqs,
                inputs=ref_input,
            )

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=analytical_coeffs[0],
                frequencies=analytical_freqs[0],
                inputs=ref_input,
            )

            exp_tree = coeff_tree(inputs=ref_input)

            assert np.isclose(
                exp_fourier_fft, exp_fourier, atol=1.0e-5
            ), "FFT and analytical Fourier series do not match"

            assert np.isclose(
                exp_tree, exp_fourier, atol=1.0e-5
            ), "Analytic Fourier series evaluation not working"


@pytest.mark.unittest
def test_coefficients_tree_mq() -> None:
    reference_inputs = np.linspace(-np.pi, np.pi, 10)

    model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        output_qubit=-1,
        as_pauli_circuit=False,
    )

    fft_coeffs, fft_freqs = Coefficients.get_spectrum(model, shift=True)

    coeff_tree = FourierTree(model)
    analytical_coeffs, analytical_freqs = coeff_tree.get_spectrum(force_mean=True)

    assert len(analytical_freqs[0]) == len(
        analytical_freqs[0]
    ), "Wrong number of frequencies"
    assert np.isclose(
        np.sum(analytical_coeffs[0]).imag, 0.0, rtol=1.0e-5
    ), "Imaginary part is not zero"

    # Filter fft_coeffs for only the frequencies that occur in the spectrum
    sel_fft_coeffs = np.take(fft_coeffs, analytical_freqs[0] + int(fft_freqs.max()))
    assert all(
        np.isclose(sel_fft_coeffs, analytical_coeffs[0], atol=1.0e-5)
    ), "FFT and analytical coefficients are not equal."

    for ref_input in reference_inputs:
        exp_fourier_fft = Coefficients.evaluate_Fourier_series(
            coefficients=fft_coeffs,
            frequencies=fft_freqs,
            inputs=ref_input,
        )

        exp_fourier = Coefficients.evaluate_Fourier_series(
            coefficients=analytical_coeffs[0],
            frequencies=analytical_freqs[0],
            inputs=ref_input,
        )

        exp_tree = coeff_tree(inputs=ref_input, force_mean=True)

        assert np.isclose(
            exp_fourier_fft, exp_fourier, atol=1.0e-5
        ), "FFT and analytical Fourier series do not match"

        assert np.isclose(
            exp_tree, exp_fourier, atol=1.0e-5
        ), "Analytic Fourier series evaluation not working"


@pytest.mark.unittest
def test_oversampling_time() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    assert (
        Coefficients.get_spectrum(model, mts=3)[0].shape[0] == 15
    ), "Oversampling time failed"


@pytest.mark.unittest
def test_oversampling_frequency() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    assert (
        Coefficients.get_spectrum(model, mfs=3)[0].shape[0] == 15
    ), "Oversampling frequency failed"


@pytest.mark.unittest
def test_shift() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )
    coeffs, freqs = Coefficients.get_spectrum(model, shift=True)

    assert (
        np.abs(coeffs) == np.abs(coeffs[::-1])
    ).all(), "Shift failed. Spectrum must be symmetric."


@pytest.mark.unittest
def test_trim() -> None:
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
        expected {coeffs.size-1}"


@pytest.mark.unittest
def test_frequencies() -> None:
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
        data_reupload=[[[True, True], [False, True]], [[False, True], [True, True]]],
    )
    coeffs, freqs = Coefficients.get_spectrum(model)

    assert (
        freqs[0].size * freqs[1].size
    ) == coeffs.size, f"(2D) Frequencies ({freqs.shape}) and \
        coefficients ({coeffs.shape}) must add up to the same length."


@pytest.mark.smoketest
def test_psd() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )
    coeffs, _ = Coefficients.get_spectrum(model, shift=True)
    _ = Coefficients.get_psd(coeffs)


@pytest.mark.unittest
def test_pearson_correlation() -> None:
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
            assert np.isclose(
                pearson[i, j], reference, atol=1.0e-5
            ), f"Pearson correlation does not match reference. \
                For index {i}, {j}, got {pearson[i, j]}, expected {reference}"


@pytest.mark.unittest
def test_spearman_correlation() -> None:
    N = 1000
    K = 5
    seed = 1000
    rng = np.random.default_rng(seed)

    # create a random array of shape N, K
    coeffs = rng.normal(size=(N, K))
    pearson = FCC._spearman(coeffs)

    for i in range(coeffs.shape[1]):
        for j in range(coeffs.shape[1]):
            reference = spearmanr(coeffs[:, i], coeffs[:, j]).correlation
            assert np.isclose(
                pearson[i, j], reference, atol=1.0e-5
            ), f"Pearson correlation does not match reference. \
                For index {i}, {j}, got {pearson[i, j]}, expected {reference}"


@pytest.mark.expensive
@pytest.mark.unittest
def test_fcc() -> None:
    """
    This test replicates the results obtained for the FCC
    as shown in Fig. 3a from the paper
    "Fourier Fingerprints of Ansatzes in Quantum Machine Learning"
    https://doi.org/10.48550/arXiv.2508.20868
    """
    test_cases = [
        {
            "circuit_type": "Circuit_15",
            "fcc": 0.004,
        },
        {
            "circuit_type": "Circuit_19",
            "fcc": 0.010,
        },
        {
            "circuit_type": "Circuit_17",
            "fcc": 0.115,
        },
        {
            "circuit_type": "Hardware_Efficient",
            "fcc": 0.144,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=6,
            n_layers=1,
            circuit_type=test_case["circuit_type"],
            output_qubit=-1,
            encoding=["RY"],
            mp_threshold=3000,
        )
        fcc = FCC.get_fcc(
            model=model,
            n_samples=500,
            seed=1000,
            scale=True,
        )
        # # print(f"FCC for {test_case['circuit_type']}: \t{fcc}")
        assert np.isclose(
            fcc, test_case["fcc"], atol=1.0e-3
        ), f"Wrong FCC for {test_case['circuit_type']}. \
            Got {fcc}, expected {test_case['fcc']}."


@pytest.mark.unittest
def test_fourier_fingerprint() -> None:
    """
    This test checks if the calculation of the Fourier fingerprint
    returns the expected result by using hashs.
    """
    test_cases = [
        {
            "circuit_type": "Circuit_15",
            "hash": "8a1fae4f3afda8c243a847c4e8396d87",
        },
        {
            "circuit_type": "Circuit_19",
            "hash": "b4d3e6f3881f69fe7e778713cbd1c573",
        },
        {
            "circuit_type": "Circuit_17",
            "hash": "422847ebfa133299cb9c654730f753a7",
        },
        {
            "circuit_type": "Hardware_Efficient",
            "hash": "2fa201197e53f04ee53eb40db755bcc9",
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=3,
            n_layers=1,
            circuit_type=test_case["circuit_type"],
            output_qubit=-1,
            encoding=["RY"],
        )
        fp_and_freqs = FCC.get_fourier_fingerprint(
            model=model,
            n_samples=500,
            seed=1000,
            scale=True,
        )
        hs = hashlib.md5(repr(fp_and_freqs).encode("utf-8")).hexdigest()
        # print(hs)
        assert (
            hs == test_case["hash"]
        ), f"Wrong hash for {test_case['circuit_type']}. \
            Got {hs}, expected {test_case['hash']}"


@pytest.mark.expensive
@pytest.mark.unittest
def test_fcc_2d() -> None:
    """
    This test replicates the results obtained for the FCC
    as shown in Fig. 3b from the paper
    "Fourier Fingerprints of Ansatzes in Quantum Machine Learning"
    https://doi.org/10.48550/arXiv.2508.20868

    Note that we only test one circuit here with and also with a lower
    number of qubits, because it get's computationally too expensive otherwise.
    """
    test_cases = [
        {
            "circuit_type": "Circuit_19",
            "fcc": 0.020,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=4,
            n_layers=1,
            circuit_type=test_case["circuit_type"],
            output_qubit=-1,
            encoding=["RX", "RY"],
            mp_threshold=3000,
        )
        fcc = FCC.get_fcc(
            model=model,
            n_samples=250,
            seed=1000,
            scale=True,
        )
        # # print(f"FCC for {test_case['circuit_type']}: \t{fcc}")
        assert np.isclose(
            fcc, test_case["fcc"], atol=1.0e-3
        ), f"Wrong FCC for {test_case['circuit_type']}. \
            Got {fcc}, expected {test_case['fcc']}."


@pytest.mark.expensive
@pytest.mark.unittest
def test_weighting() -> None:
    """
    This test replicates the results obtained for the FCC
    as shown in Fig. 3b from the paper
    "Fourier Fingerprints of Ansatzes in Quantum Machine Learning"
    https://doi.org/10.48550/arXiv.2508.20868

    Note that we only test one circuit here with and also with a lower
    number of qubits, because it get's computationally too expensive otherwise.
    """
    test_cases = [
        {
            "circuit_type": "Circuit_19",
            "fcc": 0.013,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=3,
            n_layers=1,
            circuit_type=test_case["circuit_type"],
            output_qubit=-1,
            encoding=["RY"],
            mp_threshold=3000,
        )
        fcc = FCC.get_fcc(
            model=model,
            n_samples=500,
            seed=1000,
            scale=True,
            weight=True,
        )
        # print(f"FCC for {test_case['circuit_type']}: \t{fcc}")
        assert np.isclose(
            fcc, test_case["fcc"], atol=1.0e-3
        ), f"Wrong FCC for {test_case['circuit_type']}. \
            Got {fcc}, expected {test_case['fcc']}."


@pytest.mark.unittest
def test_fourier_series_dataset() -> None:
    test_cases = [
        {"n_input_feat": 2},
        {"n_input_feat": 3},
        {"coefficients_min": 0.1, "coefficients_max": 0.9},
        {"zero_centered": True},
        {"encoding_strategy": "binary"},
        {"encoding_strategy": "ternary"},
    ]

    n_samples = 100
    seed = 1000

    for test_case in test_cases:
        rng = np.random.default_rng(seed)

        n_input_feat = test_case.pop("n_input_feat", 1)
        coefficients_min = test_case.get("coefficients_min", 0.0)
        coefficients_max = test_case.get("coefficients_max", 1.0)
        zero_centered = test_case.get("zero_centered", False)

        encoding = Encoding(
            test_case.pop("encoding_strategy", "binary"),
            ["RY" for _ in range(n_input_feat)],
        )

        model = Model(
            n_qubits=1,
            n_layers=1,
            encoding=encoding,
        )

        all_domain_samples = []
        all_fourier_samples = []
        all_coefficients = []

        for i in range(n_samples):
            try:
                domain_samples, fourier_samples, coefficients = (
                    Datasets.generate_fourier_series(
                        rng=rng,
                        model=model,
                        **test_case,
                    )
                )
            except Exception as e:
                tb = traceback.format_exc()
                raise Exception(
                    f"Error in iteration {i} for test case {test_case}: {e}\n{tb}"
                )

            # Sanity check to ensure the FFT is correct
            coefficients_hat = np.fft.fftshift(
                np.fft.fftn(
                    fourier_samples,
                    axes=list(range(model.n_input_feat)),
                )
            )
            assert np.allclose(
                coefficients,
                coefficients_hat,
                atol=1e-6,
            ), f"Frequencies don't match for test case {test_case}"

            assert np.all(
                domain_samples.shape
                == (
                    *model.degree,
                    model.n_input_feat,
                )
            ), f"Wrong shape of domain samples for test case {test_case}"
            assert np.all(
                fourier_samples.shape == model.degree
            ), f"Wrong shape of Fourier values for test case {test_case}"
            assert np.all(
                coefficients.shape == model.degree
            ), f"Wrong shape of coefficients for test case {test_case}"

            all_domain_samples.append(domain_samples)
            all_fourier_samples.append(fourier_samples)
            all_coefficients.append(coefficients)

        all_domain_samples = np.array(all_domain_samples)
        all_fourier_samples = np.array(all_fourier_samples)
        all_coefficients = np.array(all_coefficients)

        if not zero_centered:
            assert np.sqrt(coefficients_min) <= np.min(
                np.abs(coefficients)
            ), f"Coefficients are too small for test case {test_case}"
            assert np.sqrt(coefficients_max) >= np.max(
                np.abs(coefficients)
            ), f"Coefficients are too large for test case {test_case}"
        else:
            assert np.isclose(
                fourier_samples.mean(), 0.0, atol=1e-1
            ), f"Zero centering failed for test case {test_case}"

        # assert domain_samples.shape[-1] == n_input_feat, "Wrong shape of domain samples"
