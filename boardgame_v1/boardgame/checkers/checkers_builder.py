from dataclasses import dataclass, replace

from boardgame_v1.boardgame.boardgame import (
    Game,
    Board,
    Piece,
    Space,
    Player,
    MovementType, Vector, Direction)


@dataclass
class ValueObject:
    def copy(self, **kwargs):
        return replace(self, **kwargs)

forward_left_one = MovementType(
    vectors=[
        Vector(Direction.UP, -1),
        Vector(Direction.RIGHT, -1)
    ]
)
forward_right_one = MovementType(
    vectors=[
        Vector(Direction.UP, -1),
        Vector(Direction.RIGHT, 1)
    ]
)

checker = lambda player: Piece(
    player=player,
    legal_moves=[
        forward_left_one,
        forward_right_one,
    ]
)


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

    def build(self):
        return Game(
            self.board
        )
