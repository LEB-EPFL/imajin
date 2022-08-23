from leb.imajin.instruments import Instrument
from leb.imajin.samples import Sample


class Experiment:
    def __init__(self, instrument: Instrument, sample: Sample) -> None:
        self.instrument = instrument
        self.sample = sample

    def run(self):
        pass
