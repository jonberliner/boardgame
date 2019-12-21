from dataclasses import dataclass
from typing import List

from boardgame_v1.boardgame.space import Space


@dataclass
class Board:
    grid: List[List[Space]]
