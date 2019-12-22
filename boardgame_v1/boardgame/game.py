from dataclasses import dataclass

from boardgame_v1.boardgame.board import Board
from boardgame_v1.boardgame.move import Move


@dataclass
class Game:
    board: Board

    def make_move(self, move: Move):
        return self
