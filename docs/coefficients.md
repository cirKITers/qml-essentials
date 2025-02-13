# Coefficients

A characteristic property of any Fourier model are its coefficients.
Our package can, given a model, calculate the corresponding coefficients by utilizing the [Pennylane Fourier Coefficients](https://docs.pennylane.ai/en/stable/_modules/pennylane/fourier/coefficients.html) method.

In the simplest case, this could look as follows:
```python
from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients

model = Model(
            n_qubits=2
            n_layers=1
            circuit_type="HardwareEfficient",
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
```

We can now calculate the [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) of our model:
```python
X = np.fft.fft(out)
X_shift = np.fft.fftshift(X)
X_freq = np.fft.fftfreq(X.size, 1/fs)
X_freq_shift = np.fft.fftshift(X_freq)
```
Note that calling `np.fft.fftshift` is not required from a technical point of view, but makes our spectrum nicely zero-centered and projected correctly.

![Model Fct Spectr](model_fct_spectr_light.png#only-light)
![Model Fct Spectr](model_fct_spectr_dark.png#only-dark)

The same can be done with our framework, with a neat one-liner:
```python
X_shift, X_freq_shift = Coefficients.get_spectrum(model_fct, shift=True)
```

![Model Fct Spectr Ours](model_fct_spectr_ours_light.png#only-light)
![Model Fct Spectr Ours](model_fct_spectr_ours_dark.png#only-dark)

Note, that applying the shift can be controlled with the optional `shift` argument.

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

![Model Fct Spectr OS](model_fct_spectr_os_light.png#only-light)
![Model Fct Spectr OS](model_fct_spectr_os_dark.png#only-dark)

Note that, as the frequencies change with the `mts` argument, we have to take that into account when calculating the frequencies with the last call.

Feel free to checkout our [jupyter notebook](https://github.com/quantum-machine-learning/qml_essentials/blob/main/docs/notebooks/coefficients.ipynb) if you would like to play around with this.