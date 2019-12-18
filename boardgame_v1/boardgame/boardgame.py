import abc
import attr


class Board(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def make_move(self, move: Move):
        pass

@attr.s(frozen=True)
class Game(object):
    board = attr.ib(
        validator=attr.validators.instance_of(Board)
    )