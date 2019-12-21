from dataclasses import dataclass

from boardgame_v1.boardgame.piece import Piece


@dataclass
class Space:
    piece: Piece = None
