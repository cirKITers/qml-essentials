# Expressibility

Our package includes tool to estimate the expressiblity of a particularly chosen Ansatz.

```python
model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type=HardwareEfficient,
)

input_domain, bins, dist_circuit = Expressibility.state_fidelities(
    n_bins=10,
    n_samples=200,
    n_input_samples=5,
    seed=1000,
    model=model,
)
```

Note that `state_fidelities` accepts keyword arguments that are being passed to the model call.
This allows you to utilize e.g. caching.

Next, you can calculate the Haar integral (as reference), by
```python
input_domain, dist_haar = Expressibility.haar_integral(
    n_qubits=2,
    n_bins=10,
    cache=False,
)
```

Finally, the Kullback-Leibler divergence allows you to see how well the particular circuit performs compared to the Haar integral:
```python
kl_dist = Expressibility.kullback_leibler_divergence(dist_circuit, dist_haar).mean()
```