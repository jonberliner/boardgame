from dataclasses import dataclass

from boardgame_v1.boardgame.direction import Direction


@dataclass
class Vector:
    direction: Direction
    scalar: int
