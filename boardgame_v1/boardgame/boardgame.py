from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class Coord:
    rank: int
    file: int


@dataclass
class Player:
    name: str


class Direction(Enum):
    UP = 1
    RIGHT = 2


@dataclass
class Vector:
    direction: Direction
    scalar: int


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


@dataclass
class Piece:
    player: Player
    legal_moves: List[MovementType]

    def can_move(self, move: Move):
        return move.type in self.legal_moves


@dataclass
class Space:
    piece: Piece = None


@dataclass
class Board:
    grid: List[List[Space]]


@dataclass
class Game:
    board: Board
