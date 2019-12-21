from unittest import TestCase

from boardgame_v1.boardgame.boardgame import Move, Coord, Player
from boardgame_v1.boardgame.checkers.checkers_builder import checker


class TestCheckerMovement(TestCase):
    def setUp(self) -> None:
        self.checker = checker(Player("1"))

    def test_can_move_up_left_one(self):
        self.assertTrue(
            self.checker.can_move(
                Move(
                    target=Coord(1, 1),
                    to=Coord(0, 0)
                )
            )
        )

    def test_can_move_up_right_one(self):
        self.assertTrue(
            self.checker.can_move(
                Move(
                    target=Coord(1, 1),
                    to=Coord(0, 2)
                )
            )
        )

    def test_can_not_move_up_one(self):
        self.assertFalse(
            self.checker.can_move(
                Move(
                    target=Coord(1, 1),
                    to=Coord(0, 1)
                )
            )
        )

    def test_can_not_move_down_left_one(self):
        self.assertFalse(
            self.checker.can_move(
                Move(
                    target=Coord(1, 1),
                    to=Coord(2, 0)
                )
            )
        )

    def test_can_not_move_down_right_one(self):
        self.assertFalse(
            self.checker.can_move(
                Move(
                    target=Coord(1, 1),
                    to=Coord(2, 2)
                )
            )
        )

    def test_can_not_move_down_one(self):
        self.assertFalse(
            self.checker.can_move(
                Move(
                    target=Coord(1, 1),
                    to=Coord(2, 1)
                )
            )
        )
