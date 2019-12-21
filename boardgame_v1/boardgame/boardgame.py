from dataclasses import dataclass
from typing import List


@dataclass
class Piece(object):
    pass


@dataclass
class Space(object):
    piece: Piece = None


@dataclass
class Board(object):
    grid: List[List[Space]]


@dataclass
class Game(object):
    board: Board
