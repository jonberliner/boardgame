from dataclasses import dataclass

from boardgame_v1.boardgame.coord import Coord
from boardgame_v1.boardgame.movement_type import MovementType


@dataclass
class Move:
    start: Coord
    end: Coord

    @property
    def type(self) -> MovementType:
        return MovementType.from_coords(
            start=self.start,
            end=self.end,
        )
