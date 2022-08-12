from typing import Protocol, Tuple


class Detector(Protocol):
    def run(self):
        pass

class CMOSCamera:
    def __init__(self, num_pixels = Tuple[int, int]):
        self.num_pixels = num_pixels

    def run(self):
        pass
