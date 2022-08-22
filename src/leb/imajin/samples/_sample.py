from typing import Optional, Protocol

from leb.imajin import Response
from leb.imajin.instruments import Source


class Sample(Protocol):
    def response(self, source: Source, dt: float) -> Optional[Response]:
        pass
