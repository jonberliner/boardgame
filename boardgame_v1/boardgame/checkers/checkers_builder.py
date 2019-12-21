from dataclasses import dataclass

from boardgame_v1.boardgame.boardgame import Game, Board, Piece, Space


@dataclass
class CheckersBuilder(object):
    def build(self):
        checker = lambda: Piece()
        return Game(
            Board(
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