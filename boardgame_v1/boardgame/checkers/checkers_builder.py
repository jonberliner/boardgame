from dataclasses import dataclass, replace

from boardgame_v1.boardgame.board import Board
from boardgame_v1.boardgame.game import Game
from boardgame_v1.boardgame.checkers.checker import checker
from boardgame_v1.boardgame.move import Move
from boardgame_v1.boardgame.player import Player
from boardgame_v1.boardgame.space import Space


@dataclass
class ValueObject:
    def copy(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class CheckersBuilder(ValueObject):
    player1: Player = Player(name="Player1")
    player2: Player = Player(name="Player2")
    board: Board = None

    def new_game(self):
        checker_p1 = lambda: checker(self.player1)
        checker_p2 = lambda: checker(self.player2)

        return self.copy(
            board=Board(
                [
                    [Space(), Space(checker_p2()), Space(), Space(checker_p2()), Space(), Space(checker_p2()), Space(), Space(checker_p2())],
                    [Space(checker_p2()), Space(), Space(checker_p2()), Space(), Space(checker_p2()), Space(), Space(checker_p2()), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(checker_p1()), Space(), Space(checker_p1()), Space(), Space(checker_p1()), Space(), Space(checker_p1()), Space()],
                    [Space(), Space(checker_p1()), Space(), Space(checker_p1()), Space(), Space(checker_p1()), Space(), Space(checker_p1())],
                ]
            )
        )

    def move(self, move: Move):
        return self.copy(
            board=self.board.make_move(move),
        )

    def build(self):
        return Game(
            self.board
        )
