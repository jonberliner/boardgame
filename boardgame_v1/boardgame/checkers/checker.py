from boardgame_v1.boardgame.direction import Direction
from boardgame_v1.boardgame.movement_type import MovementType
from boardgame_v1.boardgame.piece import Piece
from boardgame_v1.boardgame.vector import Vector

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
