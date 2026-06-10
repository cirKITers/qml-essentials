# Coefficients

A characteristic property of any Fourier model are its coefficients.
Our package can, given a model, calculate the corresponding coefficients.

In the simplest case, this could look as follows:
```python
from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients

model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Hardware_Efficient",
        )

coeffs, freqs = Coefficients.get_spectrum(model)
```

Here, the coefficients are stored in the `coeffs` variable, and the corresponding frequency indices are stored in the `freqs` variable.

But wait! There is much more to this. Let's keep on reading if you're curious :eyes:.

## Detailled Explanation

To visualize what happens, let's create a very simplified Fourier model
```python
class Model_Fct:
    def __init__(self, c, f):
        self.c = c
        self.f = f
        self.degree = (2*max(f)+1,)
        self.frequencies = f
        self.n_input_feat = 1

    def __call__(self, inputs, **kwargs):
        return np.sum([c * np.exp(-1j * inputs * f) for f, c in zip(self.f, self.c)], axis=0)
```

This model takes a vector of coefficients and frequencies on instantiation.
When called, these coefficients and frequencies are used to compute the output of the model, which is the sum of sine functions determined by the length of the vectors.
Let's try that for just two frequencies:

```python
freqs = [-3, -1.5, 0, 1.5, 3]
coeffs = [1, 1, 0, 1, 1]

fs = max(freqs) * 2 + 1
model_fct = Model_Fct(coeffs, freqs)

x = np.arange(0, 2 * np.pi, 2 * np.pi / fs)
out = model_fct(x)
```

We can now calculate the [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) of our model:
```python
X = np.fft.fft(out) / len(out)
X_shift = np.fft.fftshift(X)
X_freq = np.fft.fftfreq(X.size, 1/fs)
X_freq_shift = np.fft.fftshift(X_freq)
```
Note that calling `np.fft.fftshift` is not required from a technical point of view, but makes our spectrum nicely zero-centered and projected correctly.

![Model Fct Spectr](figures/model_fct_spectr_light.png#center#only-light)
![Model Fct Spectr](figures/model_fct_spectr_dark.png#center#only-dark)

You may notice, that something isn't quite right here; we specified the frequencies [1.5,3] earlier, but get frequencies for [0,1,2,3].
This is because, we chose the wrong resolution for the FFT, i.e. the window length was too short.
In our framework we can achieve the same and above, while simultanously applying the fix to our problem, i.e. setting `mts=2`.
This additional variable effectively doubles the window length which gives us then the possibility to obtain frequencies "between" the integer valued frequencies seen above.

```python
X_shift, X_freq_shift = Coefficients.get_spectrum(model_fct, mts=2, shift=True)
```

![Model Fct Spectr Ours](figures/model_fct_spectr_ours_light.png#center#only-light)
![Model Fct Spectr Ours](figures/model_fct_spectr_ours_dark.png#center#only-dark)

Note, that applying the shift can be controlled with the optional `shift` argument.

Another important point is, that the `force_mean` flag is set, and the `execution_type` is is implicitly set to `expval`.
This is mainly because, we require a single expectation value to calculate the coefficients.

## Increasing the Resolution

You might have noticed that we choose our sampling frequency `fs` in such a way, that it just fulfills the [Nyquist criterium](https://en.wikipedia.org/wiki/Nyquist_frequency).
Also the number of samples `x` are just enough to sufficiently represent our function.
In such a simplified scenario, this is fine, but there are cases, where we want to have more information both in the time and frequency domain.
Therefore, two additional arguments exist in the `get_spectrum` method:
- `mfs`: The multiplier for the highest frequency. Increasing this will increase the width of the spectrum
- `mts`: The multiplier for the number of time samples. Increasing this will increase the resolution of the time domain and therefore "add" frequencies in between our original frequencies.
- `trim`: Whether to remove the Nyquist frequency if spectrum is even. This will result in a symmetric spectrum

```python
X_shift, X_freq_shift = Coefficients.get_spectrum(model_fct, mfs=2, mts=3, shift=True)
```

![Model Fct Spectr OS](figures/model_fct_spectr_os_light.png#center#only-light)
![Model Fct Spectr OS](figures/model_fct_spectr_os_dark.png#center#only-dark)

Note that, as the frequencies change with the `mts` argument, we have to take that into account when calculating the frequencies with the last call.

Feel free to checkout our [jupyter notebook](https://github.com/cirKITers/qml-essentials/blob/main/docs/coefficients.ipynb) if you would like to play around with this.

A sidenote on the performance; Increasing the `mts` value effectively increases the input lenght that goes into the model.
This means that `mts=2` will require twice the time to compute, which will be very noticable when running noisy simulations.

## Power spectral density

In some cases it can be useful to get the [power spectral density (PSD)](https://en.wikipedia.org/wiki/Spectral_density).
As calculation of this metric might differ between the different research domains, we included a function to get the PSD of a given spectrum using the following formula:

\[PSD = \frac{2 (\mathrm{Re}(F)^2+\mathrm{Im}(F)^2)}{n_\text{samples}^2}\]

where $F$ is the spectrum and $n_\text{samples}$ the length of the input vector.

```python
model = Model(
    n_qubits=4,
    n_layers=1,
    circuit_type="Circuit_19",
    random_seed=1000
)

coeffs, freqs = Coefficients.get_spectrum(model, mfs=1, mts=1, shift=True)

psd = Coefficients.get_psd(coeffs)
```

![Model PSD](figures/model_psd_light.png#center#only-light)
![Model PSD](figures/model_psd_dark.png#center#only-dark)

## Analytic Coefficients

All of the calculations above were performed by applying a Fast Fourier Transform to the output of our Model.
However, we can also calculate the coefficients analytically.

This can be achieved by the so called `FourierTree` class:
```python
from qml_essentials.coefficients import FourierTree

fourier_tree = FourierTree(model)
an_coeffs, an_freqs = fourier_tree.get_spectrum(force_mean=True)
``` 

Unlike the FFT, this gives us the precise coefficients, solely depending on the parameters.
We can verify this by comparing it to the previous results:

![Model Analytic Coefficients](figures/model_psd_an_light.png#center#only-light)
![Model Analytic Coefficients](figures/model_psd_an_dark.png#center#only-dark)

### Technical Details

We use an approach developed by [Nemkov et al.](https://arxiv.org/pdf/2304.03787), which was later extended by [Wiedmann et al.](https://arxiv.org/pdf/2411.03450).
The implementation is also inspired by the corresponding [code](https://github.com/idnm/FourierVQA) for Nemkov et al.'s paper.

In Nemkov et al.'s algorithm the first step is to separate Clifford and non-Clifford gates, such that all Clifford gates can be regarded as part of the observable, and the actual circuit only consists of Pauli rotations (cf. `qml_essentials.pauli.PauliCircuit`).
The main idea is then to split each Pauli rotation into sine and cosine product terms to obtain the coefficients, which are only dependent on the parameters of the circuit.

The Clifford commutation and Pauli bookkeeping are implemented symbolically on a stabilizer-tableau Pauli representation (`qml_essentials.operations.PauliWord`), and the parameter-dependent coefficients are evaluated with vectorized operations.
This makes the tree fast to build and evaluate, and it supports multiple input features: in that case `get_spectrum` returns, per observable, the multi-dimensional frequency vectors (shape `(n_freqs, n_features)`) and their coefficients.

The symbolic core is gate-set agnostic and lives in `qml_essentials.operations`:

- `PauliWord` — an n-qubit Pauli in the symplectic $i^{\text{phase}}\,X^x Z^z$ representation, with `compose`, `commutes_with`, `conjugate_by_clifford` (Clifford tableau evolution), and a `to_matrix`/`from_matrix` bridge to dense operators.
- `Operation.is_clifford` — a class flag marking the standard Clifford gates ($I, X, Y, Z, H, S, \text{CX}, \text{CY}, \text{CZ}, \text{SWAP}$); `conjugate_by_clifford` uses fast symbolic rules for the common ones and an exact matrix fallback for the rest.
- `Operation.decompose()` — expresses composite gates (`Rot`, `CRX`/`CRY`/`CRZ`, `CZ`) in terms of Clifford + Pauli-rotation primitives.

### Custom Circuits and Input Scaling

The `FourierTree` operates on whatever circuit the model's `script` records, so a custom variational circuit can be analysed by replacing `model.script`:

```python
import qml_essentials.jaqsi as js
from qml_essentials.gates import Gates as g

def variational(params, inputs, *args, **kwargs):
    params = params.squeeze()
    g.PauliRot(1 * inputs, "Y", wires=[0])
    g.PauliRot(params[0], "XZX", wires=[0, 1, 2])
    g.PauliRot(3 * inputs, "XZ", wires=[0, 1])
    g.PauliRot(params[1], "YY", wires=[0, 1])
    g.PauliRot(9 * inputs, "XY", wires=[0, 1])

model = Model(n_qubits=3, n_layers=1, output_qubit=0)
model._params_shape = (2, 1)
model.initialize_params()       # reallocate params for the custom shape
model.script = js.Script(f=variational, n_qubits=3)

coeffs, freqs = FourierTree(model).get_spectrum()
```

The tree distinguishes encoding from variational rotations **automatically**. 
Because every canonical rotation angle is an affine function of the inputs (encodings apply $\omega_k\,x_f$; Clifford commutation only flips a generator's sign), the tree perturbs each input feature in turn and reads off, from the change in the recorded angles, which rotations depend on which feature and with what signed integer scaling $\omega_k$.

The only requirement is that each encoding rotation be **linear in a single feature**, $\omega_k\,x_f$. Heterogeneous scalings such as $(1, 3, 9)$ are then resolved to the correctly scaled frequencies (here $\{-13, -11, -7, -5, 5, 7, 11, 13\}$) instead of unit counts. 
Non-integer scalings are rounded with a warning; a rotation depending on more than one feature raises `NotImplementedError`; and the `method="dp"` exact-support path does not support non-unit scaling.

For the numerical FFT (`Coefficients.get_spectrum`) a custom circuit additionally requires `model.degree` to be set high enough to resolve the largest frequency.
The sampling grid is built from `model.degree` (the number of frequencies, i.e. $2\,\omega_\text{max} + 1$), not `model.frequencies`.

## Estimating the Exact Spectrum

The number of frequencies a model can represent is, by default, estimated naively from the encoding (see `model.frequencies` / `model.degree`).
This estimate is an *upper bound*: as shown by Wiedmann et al., some coefficients are constrained to zero for all parameter values, so the true spectrum can be smaller.

The `FourierTree` lets us obtain the *exact* spectrum of a model via the opt-in `Model.exact_spectrum` method:
```python
model = Model(
    n_qubits=3,
    n_layers=1,
    circuit_type="Circuit_19",
)

exact = model.exact_spectrum()  # tuple of frequency arrays, one per input feature
```
The result is always a subset of `model.frequencies`.
For example, for the model above the naive estimate is `{-3, ..., 3}` while the exact spectrum is `{-2, -1, 0, 1, 2}`.

The support is derived *purely symbolically* — no parameter sampling is involved.
This exploits a structural property of the tree: every parameter contributes at most one sine **or** cosine factor per path, so the variational leaf factors are square-free monomials over $\{1, \cos\theta_i, i\sin\theta_i\}$, which are linearly independent.
A frequency is therefore absent if and only if all of its signature-grouped path weights sum to zero — an exact test, since all weights are dyadic rationals.
This correctly removes frequencies whose contributions cancel identically across paths, e.g. two consecutive encodings combining into a single rotation ($\langle Z \rangle = \cos(2x)$ has spectrum $\{-2, 2\}$, not $\{-2, \dots, 2\}$).

Two methods are available:

- `method="tree"` (default): fully exact, but enumerates the explicit tree, whose size can grow exponentially with circuit depth.
- `method="dp"`: merges tree nodes with identical (rotation index, observable) state — at most $n_\text{params} \cdot 4^{n_\text{qubits}}$ states — and tracks the achievable input sine/cosine counts per state.
  This scales to deep circuits where the explicit tree is infeasible (e.g. a 10-layer `Strongly_Entangling` ansatz on 5 qubits takes seconds instead of being intractable).
  It is exact per path, but cannot detect coefficients that cancel identically *across* paths with identical variational dependence (such as the repeated-encoding example above), where it returns a tight superset.
  Currently it supports a single input feature.

```python
exact = model.exact_spectrum(method="dp")  # for deep circuits
```

Note that `exact_spectrum` requires a Clifford + Pauli-rotation ansatz (the same restriction as the `FourierTree`).
Also keep in mind that a *structurally present* frequency can still have an exponentially small coefficient (the leaf weights scale with $0.5^{s+c}$), so numerically thresholding an FFT spectrum may show fewer frequencies than the exact symbolic support.


## Multi-Dimensional Coefficients

The `get_spectrum` method can also be used to calculate the coefficients of a model with multiple input dimensions.
This feature can be enabled, by explicitly providing an encoding that supports multi-dimensional input, e.g. a list of single encodings (see [*Usage*](usage.md) for details on how encodings are applied). 
Currently, only the FFT-based method supports this.

```python
model = Model(
    n_qubits=4,
    n_layers=1,
    circuit_type="Circuit_19",
    random_seed=1000,
    encoding=["RX", "RY"]
)

coeffs, freqs = Coefficients.get_spectrum(model, mfs=1, mts=1
, shift=True)

psd = Coefficients.get_psd(coeffs)
```

Using a logarithmic color bar, one obtains the following 2D-spectrum:

![2D Model Coefficients](figures/model_2d_psd_light.png#center#only-light)
![2D Model Coefficients](figures/model_2d_psd_dark.png#center#only-dark)

Note that "X1" refers to the "RX" encoding and "X2" to the "RY" encoding.

In the multidimensional case, the `freqs` variable now contains the frequency indices for each dimension.
This is an important detail, as due to the `data_reupload` argument, it is possible to have a different number of frequencies for each input dimension.

## Fourier Coefficient Correlation (FCC)

The FCC, as introduced in [Fourier Fingerprints of Ansatzes in Quantum Machine Learning](https://doi.org/10.48550/arXiv.2508.20868), is a metric that aims to predict the expected performance of an arbitrary Ansatz based on the the correlation between its Fourier modes.
In this framework, the FCC for a given `model` can be obtained as follows:

```python
from qml_essentials.coefficients import FCC

model = Model(
    n_qubits=6,
    n_layers=1,
    circuit_type="Hardware_Efficient",
    output_qubit=-1,
    encoding=["RY"],
)

fcc = FCC.get_fcc(
    model=model,
    n_samples=500,
    random_key=jax.random.key(1000),
)
```
Returns `0.1442` as already in Fig. 3a of aforementioned paper.

Optionally, you can choose a different correlation `method` (currently "pearson", "complex_pearson", and "spearman" are supported).
The default "pearson" method preserves the existing behavior for complex coefficients by correlating stacked real and imaginary parts.
The "complex_pearson" method computes a complex-valued Pearson coefficient via Hermitian normalized covariance, so the magnitude describes correlation strength and the angle describes the relative phase between coefficient vectors.
When calculating the scalar FCC, `calculate_fcc` uses the magnitude of the fingerprint entries.
Similar, other methods which require specifying `n_samples` (c.f. calculation of [expressibility](expressibility.md) and [entangling capability](entanglement.md)), methods in the `FCC` class take an optional parameter `scale` (defaults to `False`), which scales the number of samples depending on the number of qubits and the number of input features as $n_\text{samples} \cdot n_\text{params} \cdot 2^{n_\text{qubits}} \cdot n_\text{features}$.

As described in our paper, the FCC is calculated as the mean of the Fourier fingerprint, which in turn can be obtained separately as follows:

```python
fingerprint = FCC.get_fourier_fingerprint(
    model=model,
    n_samples=500,
    random_key=jax.random.key(1000),
)
```

![Fourier Fingerprint of Hardware Efficient Ansatz](figures/fourier_fingerprint_light.png#center#only-light)
![Fourier Fingerprint of Hardware Efficient Ansatz](figures/fourier_fingerprint_dark.png#center#only-dark)

Note that actually calculating the FCC as it is shown in the paper, requires removing all the redundant entries in the fingerprint.
This is implicitly done in `FCC.get_fourier_fingerprint` (and controlled using the `trim_redundant` argument), by
- removing all negative frequencies (because their coefficients are complex conjugates of the positive frequencies)
- removing symmetries inside the correlation matrix (the Fourier fingerprint), e.g. $c_{0,1} = c_{1,0}$
Note that `get_fcc` also (by default) trims down the fingerprint before calculating the actual FCC. 

Both `get_fcc` and `get_fourier_fingerprint` support a `weight` parameter, which can be used to weight the correlation matrix, such that high-frequency components receive a lower weight.
Intuitively this adresses the issue, that low frequency components have a higher impact on the mean-squared error (c.f. App. D in our paper). 