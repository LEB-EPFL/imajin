from leb.imajin.instruments.instrument import Instrument
from leb.imajin.samples.sample import Sample


class Experiment:
    def __init__(self, instrument: Instrument, sample: Sample) -> None:
        self.instrument = instrument
        self.sample = sample

    def run(self):
        pass
