from leb.imajin.instruments._instrument import Instrument
from leb.imajin.samples._sample import Sample


class Experiment:
    def __init__(self, instrument: Instrument, sample: Sample) -> None:
        self.instrument = instrument
        self.sample = sample

    def run(self):
        pass
