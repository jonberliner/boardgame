from unittest import TestCase

from boardgame_v1.boardgame.boardgame import Game, Board, Space, Piece
from boardgame_v1.boardgame.checkers.checkers_builder import CheckersBuilder


class TestCreateCheckersGame(TestCase):

    def test_create_8x8(self):
        checker = lambda: Piece()
        checkers = (
           CheckersBuilder()
           .new_game()
           .build()
        )

        self.assertEqual(
            checkers,
            Game(
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
        )
