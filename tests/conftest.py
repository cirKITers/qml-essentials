import pytest

from qml_essentials.pulses import PulseInformation


@pytest.fixture(autouse=True)
def isolate_pulse_information_state():
    """Run every test from the canonical pulse configuration.

    The pulse backend stores the active envelope, RWA flag, frame, pulse
    parameters, and compiled-solver cache in process-global state.  xdist
    workers execute unrelated tests in the same Python process, so preserving
    whatever state a worker happens to have at test start is not enough: a
    polluted worker would keep restoring the polluted state.  Reset before and
    after each test to make ordering and worker assignment irrelevant.
    """
    PulseInformation.reset_defaults()
    try:
        yield
    finally:
        PulseInformation.reset_defaults()
