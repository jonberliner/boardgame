from dataclasses import dataclass, replace

from boardgame_v1.boardgame.boardgame import Game, Board, Piece, Space


@dataclass
class ValueObject:
    def copy(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class CheckersBuilder(ValueObject):
    board: Board = None

    def new_game(self):
        checker = lambda: Piece()
        return self.copy(
            board=Board(
                [
                    [Space(), Space(checker()), Space(), Space(checker()), Space(), Space(checker()), Space(), Space(checker())],
                    [Space(checker()), Space(), Space(checker()), Space(), Space(checker()), Space(), Space(checker()), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(), Space(), Space(), Space(), Space(), Space(), Space(), Space()],
                    [Space(checker()), Space(), Space(checker()), Space(), Space(checker()), Space(), Space(checker()), Space()],
                    [Space(), Space(checker()), Space(), Space(checker()), Space(), Space(checker()), Space(), Space(checker())],
                ]
            )
        )

    def build(self):
        return Game(
            self.board
        )