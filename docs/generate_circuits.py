import os

from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze

edit_ansaetze_file = False
ansaetze = Ansaetze.get_available()

for q in [4, 5, 6]:
    overview_txt = "\n"
    overview_txt += f"### {q} Qubit Circuits\n"

    for ansatz in ansaetze:
        model = Model(
            n_qubits=q,
            n_layers=1,
            circuit_type=ansatz.__name__,
            output_qubit=-1,
            remove_zero_encoding=True,
            data_reupload=False,
        )

        fig, _ = model.draw(figure="mpl")

        cwd = os.path.dirname(__file__)
        fig.savefig(
            f"{cwd}/figures/circuits_{q}q/{ansatz.__name__}_light.png",
            dpi=100,
            transparent=True,
            bbox_inches="tight",
        )

        overview_txt += f"#### {ansatz.__name__.replace('_', ' ')}\n"
        overview_txt += f"![{ansatz.__name__.replace('_', ' ')}](figures/circuits_{q}q/{ansatz.__name__}_light.png#circuit#only-light)\n"  # noqa
        overview_txt += f"![{ansatz.__name__.replace('_', ' ')}](figures/circuits_{q}q/{ansatz.__name__}_dark.png#circuit#only-dark)\n"  # noqa
        overview_txt += "\n"

    if edit_ansaetze_file:
        with open(f"{cwd}/ansaetze.md", "a") as f:
            f.write(overview_txt)
