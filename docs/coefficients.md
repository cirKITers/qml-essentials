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

coeffs = Coefficients.get_spectrum(model)
```

But wait! There is much more to this. Let's keep on reading if you're curious :eyes:.

## Detailled Explanation

To visualize what happens, let's create a very simplified Fourier model
```python
class Model_Fct:
    def __init__(self, c, f):
        self.c = c
        self.f = f
        self.degree = max(f)

    def __call__(self, inputs, **kwargs):
        return np.sum([c * np.cos(inputs * f) for f, c in zip(self.f, self.c)], axis=0)
```

This model takes a vector of coefficients and frequencies on instantiation.
When called, these coefficients and frequencies are used to compute the output of the model, which is the sum of sine functions determined by the length of the vectors.
Let's try that for just two frequencies:

```python
freqs = [1,3]
coeffs = [1,1]

fs = max(freqs) * 2 + 1
model_fct = Model_Fct(coeffs,freqs)

x = np.arange(0,2 * np.pi, 2 * np.pi/fs)
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

Note that while this takes significantly longer to compute, it gives us the precise coefficients, solely depending on the parameters.
We can verify this by comparing it to the previous results:

![Model Analytic Coefficients](figures/model_psd_an_light.png#center#only-light)
![Model Analytic Coefficients](figures/model_psd_an_dark.png#center#only-dark)

### Technical Details

We use an approach developed by [Nemkov et al.](https://arxiv.org/pdf/2304.03787), which was later extended by [Wiedmann et al.](https://arxiv.org/pdf/2411.03450).
The implementation is also inspired by the corresponding [code](https://github.com/idnm/FourierVQA) for Nemkov et al.'s paper.

In Nemkov et al.'s algorithm the first step is to separate Clifford and non-Clifford gates, such that all Clifford gates can be regarded as part of the observable, and the actual circuit only consists of Pauli rotations (cf. qml_essentials.utils.PauliCircuit).
The main idea is then to split each Pauli rotation into sine and cosine product terms to obtain the coefficients, which are only dependent on the parameters of the circuit.

Currently, our implementation supports only one input feature, albeit more are theoretical possible.


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

# Fourier Coefficient Calculation (FCC)

The FCC, as introduced in [Fourier Fingerprints of Ansatzes in Quantum Machine Learning](https://doi.org/10.48550/arXiv.2508.20868), is a metric that aims to predict the expected performance of an arbitrary Ansatz based on the the correlation between its Fourier modes.
In this framework, the FCC for a given `model` can be obtained as follows:

```python
from qml_essentials.coefficients import FCC

fcc = FCC.get_fcc(
    model=model,
    n_samples=500,
    seed=1000,
)
```

Optionally, you can choose a different correlation `method` (currently "pearson" and "spearman" are supported) and enable scaling up the number of samples (depending on `n_qubit` and number of input features) by setting `scale=True`.

As described in our paper, the FCC is calculated as the mean of the Fourier fingerprint, which in turn can be obtained separately as follows:

```python
fingerprint = FCC.get_fourier_fingerprint(
    model=model,
    n_samples=500,
    seed=1000,
)
```

Note that actually calculating the FCC as it is shown in the paper, requires removing all the redundant entries in the fingerprint.
This is done in `FCC.calculate_fcc` by
- removing all negative frequencies (because their coefficients are complex conjugates of the positive frequencies)
- removing symmetries inside the correlation matrix (the Fourier fingerprint), e.g. c_0_1 = c_1_0

Similar other methods which require specifying `n_samples` (c.f. calculation of [expressibility](expressibility.md) and [entangling capability](entangling.md)), methods in the `FCC` class take an optional parameter `scale` (defaults to `False`), which scales the number of samples depending on the number of qubits and the number of input features as $\text{n_samples} \text{n_params} 2^{\text{n_qubits}} \text{n_features}$.