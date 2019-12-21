from unittest import TestCase

from boardgame_v1.boardgame.boardgame import (
    Game,
    Board,
    Space,
    Piece,
    Player,
)
from boardgame_v1.boardgame.checkers.checkers_builder import CheckersBuilder


class TestCreateCheckersGame(TestCase):

    def test_create_initial_board_state(self):
        player1 = Player("1")
        player2 = Player("2")
        checker_p1 = lambda: Piece(player=player1)
        checker_p2 = lambda: Piece(player=player2)

        checkers = (
           CheckersBuilder(
               player1=player1,
               player2=player2,
           )
           .new_game()
           .build()
        )

        self.assertEqual(
            checkers,
            Game(
                Board(
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
        )
