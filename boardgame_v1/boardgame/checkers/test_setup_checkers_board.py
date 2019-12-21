from unittest import TestCase

from boardgame_v1.boardgame.board import Board
from boardgame_v1.boardgame.game import Game
from boardgame_v1.boardgame.checkers.checkers_builder import CheckersBuilder, checker
from boardgame_v1.boardgame.player import Player
from boardgame_v1.boardgame.space import Space


class TestCreateCheckersGame(TestCase):

    def test_create_initial_board_state(self):
        player1 = Player("1")
        player2 = Player("2")
        checker_p1 = lambda: checker(player=player1)
        checker_p2 = lambda: checker(player=player2)

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
