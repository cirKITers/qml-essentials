# Entanglement

As one of the fundamental aspects of quantum computing, entanglement plays also an important role in quantum machine learning.
Our package offers methods for calculating the entangling capability of a particular model.
Currently, only the "Meyer-Wallach" measure is implemented, but other will be added soon!

In the simplest case, this could look as follows:
```python
from qml_essentials.model import Model
from qml_essentials.entanglement import Entanglement

model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="HardwareEfficient",
        )

ent_cap = Entanglement.meyer_wallach(
    model, n_samples=1000, seed=1000
)
```

Here, `n_samples` is the number of samples for the input domain in $[0, 2\pi]$, and `seed` is the random number generator seed.
Note, that every function in this class accepts keyword-arguments which are being passed to the model call, so you could e.g. enable caching by
```python
ent_cap = Entanglement.meyer_wallach(
    model, n_samples=1000, seed=1000, cache=True
)
```

If you set `n_samples=None`, we will use the currently stored parameters of the model to estimate the degree of entanglement.