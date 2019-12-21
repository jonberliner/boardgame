from dataclasses import dataclass
from typing import List


@dataclass
class Coord:
    rank: int
    file: int


@dataclass
class Move:
    target: Coord  # named target cause `from` is a reserved word
    to: Coord


@dataclass
class Player:
    name: str


@dataclass
class Piece:
    player: Player

    def can_move(self, move: Move):
        return True


@dataclass
class Space:
    piece: Piece = None


@dataclass
class Board:
    grid: List[List[Space]]


@dataclass
class Game:
    board: Board
