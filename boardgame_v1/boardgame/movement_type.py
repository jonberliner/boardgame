from dataclasses import dataclass
from typing import List

from boardgame_v1.boardgame.coord import Coord
from boardgame_v1.boardgame.direction import Direction
from boardgame_v1.boardgame.vector import Vector


@dataclass
class MovementType:
    vectors: List[Vector]

    @staticmethod
    def from_coords(
            start: Coord,
            end: Coord,
    ):
        return MovementType(
            vectors=[
                Vector(Direction.UP, end.rank - start.rank),
                Vector(Direction.RIGHT, end.file - start.file),
            ]
        )
