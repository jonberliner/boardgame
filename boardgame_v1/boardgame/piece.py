from dataclasses import dataclass
from typing import List

from boardgame_v1.boardgame.move import Move
from boardgame_v1.boardgame.movement_type import MovementType
from boardgame_v1.boardgame.player import Player


@dataclass
class Piece:
    player: Player
    legal_moves: List[MovementType]

    def can_move(self, move: Move):
        return move.type in self.legal_moves
