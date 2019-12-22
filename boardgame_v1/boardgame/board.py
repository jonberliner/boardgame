from dataclasses import dataclass
from typing import List

from boardgame_v1.boardgame.coord import Coord
from boardgame_v1.boardgame.move import Move
from boardgame_v1.boardgame.space import Space


@dataclass
class Board:
    # Make this immutable instead of Lists
    grid: List[List[Space]]

    def make_move(self, move: Move):
        return (
            self
            ._set(move.end, self._get(move.end))
            ._set(move.start, Space())
        )

    def _set(self, coord: Coord, space: Space):
        return self.copy(
            grid=
        )
