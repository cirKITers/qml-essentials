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

coeffs = Coefficients.sample_coefficients(model)
```