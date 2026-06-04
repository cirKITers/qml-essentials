import jax

# Re-exported for backwards compatibility: the Pauli-Clifford circuit transform
# now lives in :mod:`qml_essentials.pauli` (integrated with the symbolic
# :class:`~qml_essentials.operations.PauliWord` core).
from qml_essentials.pauli import PauliCircuit, PauliTape  # noqa: F401


def safe_random_split(random_key: jax.random.PRNGKey, *args, **kwargs):
    if random_key is None:
        return None, None
    else:
        return jax.random.split(random_key, *args, **kwargs)
