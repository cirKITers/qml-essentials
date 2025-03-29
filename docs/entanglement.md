# Entanglement

As one of the fundamental aspects of quantum computing, entanglement plays also an important role in quantum machine learning.
Our package offers methods for calculating the entangling capability of a particular model.

In the simplest case, using the Meyer-Wallach measure, this could look as follows:
```python
from qml_essentials.model import Model
from qml_essentials.entanglement import Entanglement

model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Hardware_Efficient",
        )

ent_cap = Entanglement.meyer_wallach(
    model, n_samples=1000, seed=1000
)
```

Here, `n_samples` is the number of samples for the parameters, sampled according to the default initialization strategy of the model, and `seed` is the random number generator seed.

Note, that every function in this class accepts keyword-arguments which are being passed to the model call, so you could e.g. enable caching by

```python
ent_cap = Entanglement.meyer_wallach(
    model, n_samples=1000, seed=1000, cache=True
)
```

If you set `n_samples=None`, we will use the currently stored parameters of the model to estimate the degree of entanglement.

## Bell-Measurement

An alternate method for calculating the entangling capability is the Bell-measurement method.
We can utilize this by

```python
ent_cap = Entanglement.bell_measurements(
    model, n_samples=1000, seed=1000
)
```

## Relative Entropy

While calculating entanglement using the Meyer-Wallach or Bell-Measurements method works great for noiseless circuits, it won't result in the correct values when being used together with incoherent noise.
To account for this, you can use the relative entropy method as follows: 

```python
ent_cap = Entanglement.relative_entropy(
    model, n_samples=1000, n_sigmas=10, seed=1000, noise_params={"BitFlip": 0.1}
)
```

Note that this method takes an additional parameter `n_sigmas`, which is the number of density matrices of the next separable state that we use for comparison.
The runtime scales with `n_sigmas`$\times$`n_samples` and both increase exponentially if `scale=True` is set.

Internally, we compare the states, obtained from the PQC, against those from a [GHZ state](https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state) of the same size (which we consider the next separable state).
This approach is explained in detail in [this paper](https://doi.org/10.48550/arXiv.quant-ph/0504163) and illustrated in the following figure:

![Relative Entropy](figures/rel-entropy.svg#center)
