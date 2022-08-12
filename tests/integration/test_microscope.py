from leb.imajin import Experiment
from leb.imajin.instruments import NullInstrument
from leb.imajin.samples import NullSample


def test_simple_experiment():
    instrument = NullInstrument()
    sample = NullSample()
    experiment = Experiment(instrument, sample)

    experiment.run()
