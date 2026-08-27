import os

import jax.numpy as jnp

from qml_essentials.ansaetze import Ansaetze
from qml_essentials.drawing import draw_mpl
from qml_essentials.operations import Barrier
from qml_essentials.tape import recording

edit_ansaetze_file = False
ansaetze = Ansaetze.get_available()
cwd = os.path.dirname(__file__)


def plot_circuit(q, ansatz):
    # record the ansatz directly with jaqsi to exclude any encoding gates
    pqc = ansatz()
    params = jnp.zeros(pqc.n_params_per_layer(q))

    with recording() as tape:
        pqc(params, q)

    # barriers and gate angles only add clutter to the ansatz overview
    ops = [op for op in tape if not isinstance(op, Barrier)]
    fig, _ = draw_mpl(ops, q, gate_values=False)

    fig.savefig(
        f"{cwd}/figures/circuits_{q}q/{ansatz.__name__}_light.png",
        dpi=100,
        transparent=True,
        bbox_inches="tight",
    )


for q in [4, 5, 6]:
    overview_txt = "\n"
    overview_txt += f"### {q} Qubit Circuits\n"

    for ansatz in ansaetze:
        plot_circuit(q, ansatz)

        overview_txt += f"#### {ansatz.__name__.replace('_', ' ')}\n"
        overview_txt += f"![{ansatz.__name__.replace('_', ' ')}](figures/circuits_{q}q/{ansatz.__name__}_light.png#circuit#only-light)\n"  # noqa
        overview_txt += f"![{ansatz.__name__.replace('_', ' ')}](figures/circuits_{q}q/{ansatz.__name__}_dark.png#circuit#only-dark)\n"  # noqa
        overview_txt += "\n"

    if edit_ansaetze_file:
        with open(f"{cwd}/ansaetze.md", "a") as f:
            f.write(overview_txt)
