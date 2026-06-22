# Expressibility

Our package allows you estimate the expressiblity of a given model.
```python
model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type="Hardware_Efficient",
)

bins, dist_circuit = Expressibility.state_fidelities(
    n_samples=200,
    n_bins=10,
    model=model,
    random_key=jax.random.key(1000),
)
```

Here, `n_bins` is the number of bins used in the histogram and `n_samples` is the number of parameter pairs to generate (using the default initialization strategy of the model).
For each pair, the state fidelity is computed and histogrammed, so `state_fidelities` returns the bin edges and the corresponding fidelity distribution.
`random_key` is an optional JAX random key for parameter initialization; if omitted, the model's internal random key is used.

Note that `state_fidelities` accepts keyword arguments that are being passed to the model call.
This allows you to utilize e.g. caching.

Next, you can calculate the Haar integral (as reference), by
```python
input_domain, dist_haar = Expressibility.haar_integral(
    n_qubits=2,
    n_bins=10,
    cache=True,
)
```

Finally, the Kullback-Leibler divergence allows you to see how well the particular circuit performs compared to the Haar integral:
```python
kl_dist = Expressibility.kullback_leibler_divergence(dist_circuit, dist_haar).mean()
```

Alternatively, the `kl_divergence_to_haar` shortcut combines all three steps (sampling the fidelities, computing the Haar reference and the divergence) in a single call:
```python
kl_dist = Expressibility.kl_divergence_to_haar(
    model=model,
    n_samples=200,
    n_bins=10,
    random_key=jax.random.key(1000),
).mean()
```