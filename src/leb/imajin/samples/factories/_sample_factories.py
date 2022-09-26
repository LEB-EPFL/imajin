from enum import Enum, auto
from typing import Optional

import numpy as np
import numpy.typing as npt

from leb.imajin import Sample
from leb.imajin.samples import Emitters, Fluorophore, NullSample
from leb.imajin.samples._state_machine import StateMachine


class SampleType(Enum):
    THREE_STATE_FLUOROPHORES = auto()


def sample_factory(sample_type: SampleType, *args, **kwargs) -> Sample:
    if sample_type == SampleType.THREE_STATE_FLUOROPHORES:
        return three_state_fluorophores(*args, **kwargs)

    return NullSample()


def three_state_fluorophores(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    z: npt.ArrayLike,
    *_,
    k_on=0.01,
    k_off=0.1,
    k_bleach=0.001,
    cross_section=1e-6,
    fluorescence_lifetime=1e-6,
    quantum_yield=0.8,
    wavelength=7,
    rng: Optional[np.random.Generator] = None,
    **__,
) -> Sample:
    """Creates fluorophores with three states: ON, OFF, and BLEACHED.

    Parameters
    ----------
    x: numpy.typing.ArrayLike
    y: numpy.typing.ArrayLike
    z: numpy.typing.ArrayLike
    k_on: float
        The transition rate constant from the OFF state to the ON state.
    k_off: float
        The first order rate coefficient from the ON state to the OFF state.
    k_bleach: float
        The first order rate coefficient from the ON state to the bleached state.
    cross_section: float
        The absorption cross section of the fluorophore.
    fluorescence_lifetime: float
        The average time spent in the excited state before a photon is emitted.
    quantum_yield: float
        The fraction of absorbed photons that produce fluorescence photons.
    wavelength: float
        The wavelength of the fluorescence photons.
    rng: Optional[np.random.Generator]
        The random number generator for the state machine.

    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    z = np.asanyarray(z)

    rate_constants = np.array([[0, 0, 0], [k_on, 0, 0], [0, 0, 0]])
    rate_coefficients = np.array([[[[0, k_off, k_bleach], [0, 0, 0], [0, 0, 0]]]])

    fluorophores = []
    for x_, y_, z_ in zip(x, y, z):
        state_machine = StateMachine(
            current_state=0,
            control_params=np.array([0]),
            rate_constants=rate_constants,
            rate_coefficients=rate_coefficients,
            rng=rng,
        )
        fluorophores.append(
            Fluorophore(
                x=x_,
                y=y_,
                z=z_,
                cross_section=cross_section,
                fluorescence_lifetime=fluorescence_lifetime,
                fluorescence_state=0,
                quantum_yield=quantum_yield,
                state_machine=state_machine,
                wavelength=wavelength,
            )
        )
    emitters = Emitters(fluorophores)

    return emitters
