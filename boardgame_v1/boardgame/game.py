from dataclasses import dataclass

from boardgame_v1.boardgame.board import Board


@dataclass
class Game:
    board: Board
