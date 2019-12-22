from unittest import TestCase

from boardgame_v1.boardgame.checkers.checkers_builder import CheckersBuilder
from boardgame_v1.boardgame.coord import Coord
from boardgame_v1.boardgame.move import Move
from boardgame_v1.boardgame.player import Player


class TestMakeFirstMove(TestCase):
    def setUp(self) -> None:
        self.player1 = Player("1")
        self.player2 = Player("2")
        self.checkers = (
            CheckersBuilder(
                player1=self.player1,
                player2=self.player2,
            )
            .new_game()
            .build()
        )

    def test_make_a_legal_move_updates_board_state(self):
        self.assertEqual(
            self.checkers.make_move(
                Move(
                    start=Coord(6, 0),
                    end=Coord(5, 1),
                )
            ),
            (
                CheckersBuilder(
                    player1=self.player1,
                    player2=self.player2,
                )
                .new_game()
                .move(
                    Move(
                        start=Coord(6, 0),
                        end=Coord(5, 1),
                    )
                )
                .build()
            )
        )

    def test_make_a_legal_move_updates_turn(self):
        pass
