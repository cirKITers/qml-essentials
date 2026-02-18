import pennylane as qml
import pennylane.numpy as np

from qml_essentials.entanglement.utils import EntanglementMeasure, ConvexRoofExtension, eigen_decomp_strategy


@EntanglementMeasure(name="meyer wallach")
def meyer_wallach(rho, n, **kwargs):
    return meyer_wallach_measure(rho, n, **kwargs)


def meyer_wallach_measure(rho, n, **kwargs):
    qb = list(range(n))
    entropy = 0
    for j in range(n):
        # Formula 6 in https://doi.org/10.48550/arXiv.quant-ph/0305094
        density = qml.math.partial_trace(rho, qb[:j] + qb[j + 1:])
        # only real values, because imaginary part will be separate
        # in all following calculations anyway
        # entropy should be 1/2 <= entropy <= 1
        entropy += np.trace((density @ density).real)

    # inverse averaged entropy and scale to [0, 1]
    return 2 * (1 - entropy / n)


@EntanglementMeasure(name="entanglement of formation")
@ConvexRoofExtension()
def entanglement_of_formation(rho, n, **kwargs):
    return meyer_wallach_measure(rho, n)


@EntanglementMeasure(name="concentratable entanglement", allowNoise=True, requiresDevice=True, requiresModel=True)
@ConvexRoofExtension()
def concentratable_entanglement(rho, n, getDevice, model):
    dev = getDevice(
        name="default.mixed",
        shots=model.shots,
        wires=n * 3,
    )
    _swap_test = qml.QNode(swap_test, dev)

    probs = _swap_test(rho, n)
    return 1 - probs[0]

def swap_test(rho, n):
    """
    Constructs a circuit to compute the concentratable entanglement using the
    swap test by creating two copies of a state given by a density matrix rho
    and mapping the output wires accordingly.

    Args:
        rho (np.ndarray): the density matrix of the state on which the swap
            test is performed.

    Returns:
        List[np.ndarray]: Probabilities obtained from the swap test circuit.
    """

    qml.QubitDensityMatrix(rho, wires=[i for i in range(n, 2 * n)])
    qml.QubitDensityMatrix(rho, wires=[i for i in range(2 * n, 3 * n)])

    # Perform swap test
    for i in range(n):
        qml.H(i)

    for i in range(n):
        qml.CSWAP([i, i + n, i + 2 * n])

    for i in range(n):
        qml.H(i)

    return qml.probs(wires=[i for i in range(n)])
