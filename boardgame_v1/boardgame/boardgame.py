from dataclasses import dataclass
from typing import List


@dataclass
class Player:
    name: str


@dataclass
class Piece:
    player: Player


@dataclass
class Space:
    piece: Piece = None


@dataclass
class Board:
    grid: List[List[Space]]


@dataclass
class Game:
    board: Board
