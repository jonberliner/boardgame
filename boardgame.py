from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import dataclass, field
from numbers import Number
from typing import (
    Optional,
    Union,
    Any,
    List,
    Tuple,
    Dict,
    Callable,
    Set,
    Collection,
    Text,
    AnyStr,
    ClassVar,)
from random import randint
from typing import (
    NewType,
    TypeVar)
from uuid import (
    uuid4,
    UUID)
import itertools

from numpy.random import RandomState
from numpy import ndarray
from numpy import array as nparray

# I think typevar differs from type in that they're strictly for
# type checks, and cannot be called
T = TypeVar('T')  # for generics
# for easier casting
array_likes = (list, tuple, nparray)
# aliases
Array = TypeVar('Array', List, Tuple, ndarray)
SingleOrArray = TypeVar('SingleOrArray', Array, Text, Number)
Location = NewType('Location', SingleOrArray)
UName = NewType('UName', Text)


def to_array(x: SingleOrArray,
             atype: Optional[Array[T]]=list) -> Array[T]:
    if isinstance(x, (Number, Text)):
        assert atype is not None
        x = [x]
    assert isinstance(x, array_likes)
    return atype(x)


# the generic for stuff in the game universe
@dataclass
class GameObject:
    rng_seed: int = field(default_factory=lambda: randint(0, 1000000000))
    rng: RandomState = field(init=False)
    uuid: UUID = field(default_factory=uuid4)
    uname: Optional[UName] = None

    def __post_init__(self) -> None:
        self.rng = RandomState(self.rng_seed)

    def copy(self) -> GameObject:
        return copy(self)

    def deepcopy(self) -> GameObject:
        return deepcopy(self)

    @property
    def cls(self) -> T:
        "for easier type checking"
        return self.__class__

    @property
    def cls_name(self) -> Text:
        "for easier type checking"
        return self.__class__.__name__

    def to_array(self, *args, **kwargs):
        return to_array(*args, **kwargs)

G = TypeVar("G", bound=GameObject)

## PLAYERS
class Player(GameObject):
    name: str


class Piece(GameObject):
    """A piece is something that can be placed on the board, and which players
    may use to perform actions in actions"""
    actions: Collection[Action]


class Board(GameObject):
    "Board is a generic for the locations available on a board"
    locations: Collection[Location]


class PlayerPieceLocation(GameObject):
    player: Player
    piece: Piece
    location: Location


class GameState(GameObject):
    """base gamestate.  container for getting info about the current game
    not meant to contain logic outside getting / querying its attrs
    meant to get updated by game"""

    board: Board
    players: List[Player]
    _player_piece_locations: Optional[Dict[Player, Dict[Piece, Location]]]=None

    def __post_init__(self):
        if self._player_piece_locations is None:
            self._player_piece_locations = {
                player: {} for player in self.players}

    def player_piece_locations(self, player: Player) -> Dict[Piece, Location]:
        "Dict[Piece, Location] of pieces on board for player"
        return self._player_piece_locations[player]

    def player_pieces(self, player: Player) -> Tuple[Piece]:
        "set of pieces on board for player"
        return tuple(self.player_piece_locations(player).keys())

    def player_locations(self, player) -> Tuple[Location]:
        "return set of all locations occupied by a player"
        return tuple(self.player_piece_locations(player).values())

    def get_piece_locs(self,
                       player: Optional[Player]=None,
                       piece_type: Optional[Piece]=None,
                       action_type: Optional[Action]=None)\
            -> Dict[Piece, Location]:
        """return dict of pieces on board and their location on board
        optionally filtered on player, piece_type, and action_type"""
        players = [player] if player is not None else  self.players
        output = {}
        # filter player
        for player in players:
            piece_to_loc = self.player_piece_locations[player]
            # filter piece type
            if piece_type:
                piece_to_loc = {key: value for key, value in piece_to_loc.items()
                                if isinstance(key, piece_type)}
            # filter to action type
            if action_type:
                piece_to_loc = {key: value for key, value in piece_to_loc.items()
                                if action_type in value.actions}
                output = {**output, **piece_to_loc}
        return output

    def get_pieces(self, *args, **kwargs) -> Set[Piece]:
        """return filtered set of pieces currently on board"""
        piece_locs = self.get_piece_locs(*args, **kwargs)
        return set(piece_locs.keys())

    def get_occupied(self, *args, **kwargs) -> Set[Location]:
        """return filtered set of occupied locations on board"""
        piece_locs = self.get_piece_locs(*args, **kwargs)
        return set(piece_locs.values())

    def get_empty(self, *args, **kwargs):
        """return filtered set of empty locations on board"""
        return  self.board_locations - self.get_occupied(*args, **kwargs)

    @property
    def board_locations(self):
        "return all board locations"
        return set(self.board.locations)


class Action(ABC, GameObject):
    """Action mainly exists to to contain modular logic is_valid,
    which says if this action is true given player wants piece
    to take this action given the current gamestate"""
    @abstractmethod
    def isvalid(player: Player,
                piece: Piece,
                gamestate: GameState) -> bool:
        """determine if this action is valid by this player on this piece given
        a game state"""
        ...


class GridBoard(Board):
    shape: List[int] = 2

    def __post_init__(self):
        self.locations = list(
            itertools.product(*[range(dim) for dim in self.shape]))


class MoveNd(Action):
    """check a valid move in an nd space"""
    # EXAMPLES
    # in 2d, 1 for lrud, 2 to include diags
    # restrict a direction by setting its lo and hi to zero
    # restrict to diag with lo = hi > 0
    # lr only: lo = [-1, 0], hi = [1, 0],
    # knight: lo = [-2, -2], hi = [2, 2], min_delta_taxi = max_delta_taxi = 3
    # diag only: lo = [-1, -1], hi = [1, 1] min_delta_taxi = max_delta_taxi = 2

    # default config set up to allow moves to the 8 cells surrounding a 2d cell
    dim: int
    delta_lo: NumOrArray = -1  # use a list to specify per dimension
    delta_hi: NumOrArray = 1
    delta_step: NumOrArray = 1
    min_taxi: Optional[NumOrArray] = 0  # min taxi/manhattan distance of the move allowed on a dim
    max_taxi: Optional[NumOrArray] = 1  # max taxi/manhattan distance of the move allowed on a dim

    # for post-init
    dims: Tuple[int] = field(init=False)
    possible_deltas_per_dim: Dict[int, Array[Number]] = field(init=False)

    def __post_init__(self):
        self.dims = np.arange(self.dim)

        self.delta_lo = self.to_array(self.delta_lo, nparray)
        self.delta_hi = self.to_array(self.delta_hi, nparray)
        self.delta_step = self.to_array(self.delta_step, nparray)
        self.min_taxi = self.to_array(self.min_taxi, nparray)
        self.max_taxi = self.to_array(self.max_taxi, nparray)

        self.possible_deltas_per_dim = {}
        for dim in self.dims:
            self.possible_deltas_per_dim[dim] = list(range(
                start=self.delta_lo[dim],
                stop=self.delta_hi[dim] + self.delta_step[dim],
                step=self.delta_step[dim]))

    def isvalid(self, player, piece, gamestate, to_=None, delta_=None):
        """
        use to_ for allocentric moves (birdseye view)
        use delta_ for egocentric moves (1st person)
        """
        # must pass only 1
        assert (to_ is None) + (delta_ is None)  == 1  # xor

        # get current piece loc
        from_ = gamestate.occupied_locations(player=player, piece=piece)
        # assert only 1 location for piece
        assert len(from_) in (0, 1)
        # invalid if player doesn't own piece
        if not from_:
            return False
        from_ = from_.pop()

        # create to_ if passed a delta
        if to_ is None:
            to_ = from_ + delta
        # else create delta
        else:
            delta_ = to_ = from_

        # invalid if not on board
        if to_ not in gamestate.board_locations:
            return False

        # ensure valid move distances in each and all dims
        valid_delta = self.valid_taxi(delta_)
        return valid_delta

    def to_val_per_dim(self, x: NumOrListNum) -> List[Number]:
        "ensure we have a list of numbers of size self.dim"
        if isinstance(x, Number):
            x = [x] * self.dim
        x = list(x)
        assert isinstance(x, list)
        assert len(x)
        for _x in x:
            assert isinstance(_x, int)
        return x

    def meets_delta_constraints(self, delta: List[Number]) -> bool:
        assert len(delta) == self.dim,\
            f'expected shape {self.dim}; got shape {len(delta)}'
        # check valid distance moved
        delta_taxi = self.taxi(delta)
        valid_taxi = self.min_taxi <= delta_taxi <= self.max_taxi
        if not valid_taxi:
            return False

        # check valid per dim
        for dim in self.dims:
            if delta[dim] not in set(self.possible_deltas_per_dim[dim]):
                return False
        # all checks passed
        return True

    @ staticmethod
    def taxi(array, dim=None) -> Number:
        # np.linalg.norm(array, ord='nuc', axis=dim)  # think the same when dim=-1
        return np.sum(abs(array), axis=dim)


class Game(GameObject, ABC):
    """main game object.  handles logic of a game between players players
    with a board initialized with initialize_board and gamestate initialized
    with initialize_gamestate"""

    def __init__(self,
                 players: List[Player],
                 initialize_board: Callable[[...], Board],
                 initialize_gamestate: Callable[[Board, ...], GameState],
                 init_board_kwargs: Dict[str, Any],
                 init_gamestate_kwargs: Dict[str, Any]):
        super().__init__()
        self.players = players
        # set initializers
        self.initialize_board = initialize_board or self.initialize_board
        self.initialize_gamestate =\
            initialize_gamestate or self.initialize_gamestate

        # init board
        self.board = self.initialize_board(**init_board_kwargs)
        # init gamestate for this board
        self.gamestate = self.initialize_gamestate(
            board=self.board,
            players=self.players
            **init_gamestate_kwargs)

    @abstractmethod
    def next_turn(self,
                  gamestate: Optional[GameState]=None,
                  *args, **kwargs) -> Player:
        "logic for determining the next player given the gamestate"
        gamestate = gamestate or self.gamestate
        ...
        return player

    @abstractmethod
    def take_action(self,
                    player: Player,
                    piece: Piece,
                    action: Action,
                    gamestate: Optional[GameState]=None) -> GameState:
        """update gamestate with player using piece to take action"""
        ...
        return gamestate


    def is_valid_action(self, player, piece, action, gamestate=None) -> bool:
        """determine if a player using a piece to take an action
        given the game state is valid"""
        gamestate = gamestate or self.gamestate
        valid = False
        try:
            assert player in gamestate.players
            player_pieces = gamestate.player_pieces(player)
            assert piece in player_pieces
            assert action in piece.actions
            assert action.is_valid(gamestate=gamestate)
            valid = True
        except AssertionError as err:
            print('fails assertion {0}'.format(err))
        except Exception as err:
            print('unexpected error {0}'.format(err))
        return valid

    # @abstractmethod
    # def take_action(self,
    #                 player: Player,
    #                 piece: Piece,
    #                 action: Action,
    #                 gamestate: Optional[GameState]=None) -> GameState:
    #     gamestate = gamestate or self.gamestate
    #     ...
    #     return gamestate

    @abstractmethod
    def initialize_gamestate(self,
                             board: Board,
                             players: List[Player],
                             *args, **kwargs) -> GameState:
        """default method for initialing a gamestate given a
        board and a list of players"""
        ...
        return gamestate

    @abstractmethod
    def initialize_board(self, *args, **kwargs) -> Board:
        """default method for initialing a board for this game"""
        ...
        return board


# ## ACTIONS

#     @property
#     @lru_cache(max_size=1)
#     def possible_deltas(self,
#                         filter_: Optional[Callable[Location, bool]]=None):
#         dims = dims or self.dims
#         offset = offset or np.zeros(self.dim)
#         _coords = []
#         # collect possible values for each reachable place
#         for dim in dims:
#             inc = self.inc[dim]
#             # padding so grabs from [lo, hi] instead of [lo, hi)
#             lo = self.lo[dim]
#             hi = self.hi[dim]
#             mag = self.mag[dim]
#             lo = max(lo, offset - mag)
#             hi = min(hi, offset + mag)
#             _coords += list(np.range(self.lo[dim], self.hi[dim] + inc, inc))

#         # filter to locations close enough
#         coords = itertools.product(_coords)
#         possible = set()
#         for _coord in coords:
#             if np.linalg.norm(_coord) <= self.total_mag:
#                 coord = np.zeros(self.dim)
#                 coord[dims] = _coord
#                 possible.add(coord)
#         return possible

# def all_deltas(dims, lo, hi, min_taxi=None, max_taxi=None, offset=None):
#     dims = dims or self.dims
#     offset = offset or np.zeros(self.dim)

#     _coords = []
#     # collect possible values for each reachable place
#     for dim in dims:
#         inc = self.inc[dim]
#         # padding so grabs from [lo, hi] instead of [lo, hi)
#         lo = self.lo[dim]
#         hi = self.hi[dim]
#         mag = self.mag[dim]
#         lo = max(lo, offset - mag)
#         hi = min(hi, offset + mag)
#         _coords += list(np.range(self.lo[dim], self.hi[dim] + inc, inc))

#     # filter to locations close enough
#     coords = itertools.product(_coords)
#     possible = set()
#     for _coord in coords:
#         if np.linalg.norm(_coord) <= self.total_mag:
#             coord = np.zeros(self.dim)
#             coord[dims] = _coord
#             possible.add(coord)
#     return possible


# @dataclass
# class Move2d(Action):
#     min_step: Number = 1
#     max_step: Number = 1

#     deltas: List[List[Number]] = [[1, 0],[-1, 0],[0, 1], [0, -1]]  # r l u d

#     max_per_location: Optional[int]=None

#     def isvalid(self,
#                 from_,
#                 to_,
#                 player,
#                 piece,
#                 gamestate):

#         # ensure player has piece on board
#         if not self.player_owns(player, piece, gamestate):
#             return False

#         if not self.piece_can_occupy(piece, loc, gamestate):
#             return False

#         piece_loc = gamestate.occupied_locations(player=player, piece=piece)

#         delta = to_ - piece_loc
#         return delta in self.deltas

#     def player_owns(self, player, piece, gamestate):
#         piece_loc = gamestate.player_piece_locations(player).get(piece, None)
#         return piece_loc is not None

#     def piece_can_occupy(self, piece, loc, gamestate) -> bool:
#         if self.max_per_location > 0:
#             occupied_locations = gamestate.occupied_locations()
#             if self.max_per_location == 1:
#                 if piece_loc not in occupied_locations:
#                     return False
#             else:
#                 n_in_cell = len([loc for loc in occupied_locations
#                                  if to_ == loc])
#                 return n_in_cell < max_per_location




#         possible = self.all_possible(from_, to_, board_locations)
#         return

# [0, 0] # stay
# [0, 1] # right
# [0, -1] # left

# @dataclass(init=False, frozen=True)
# class Egocentric(GameObject):
#     def __init__(self, dim=2, lo=-1, hi=1, inc=1, mag=1, total_mag=1):
#         self.dim = dim
#         # dim1 is u/d dim2 is l/r dim3 is forward/backward, etc...
#         self.dims = np.array(list(range(dim)))
#         self.lo = np.array(self.ensure_list_num(lo, dim))
#         self.hi = np.array(self.ensure_list_num(hi, dim))
#         self.inc = np.array(self.ensure_list_num(inc, dim))
#         self.mag = np.array(self.ensure_list_num(mag, dim))
#         self.total_mag = total_mag

#     def ensure_list_num(self, x: NumOrListNum, dim: int):
#         # dim = self.dim
#         if isinstance(x, int):
#             x = [x] * dim
#         assert isinstance(x, list)
#         [assert isinstance(_x, int) for _x in x]
#         assert len(x) == dim
#         return x

#     def possible_moves(self, offset=None, dims=None):
#         dims = dims or self.dims
#         offset = offset or np.zeros(self.dim)
#         _coords = []
#         # collect possible values for each reachable place
#         for dim in dims:
#             inc = self.inc[dim]
#             # padding so grabs from [lo, hi] instead of [lo, hi)
#             lo = self.lo[dim]
#             hi = self.hi[dim]
#             mag = self.mag[dim]
#             lo = max(lo, offset - mag)
#             hi = min(hi, offset + mag)
#             _coords += list(np.range(self.lo[dim], self.hi[dim] + inc, inc))

#         # filter to locations close enough
#         coords = itertools.product(_coords)
#         possible = set()
#         for _coord in coords:
#             if np.linalg.norm(_coord) <= self.total_mag:
#                 coord = np.zeros(self.dim)
#                 coord[dims] = _coord
#                 possible.add(coord)
#         return possible


    # dim: int = 2
    # lo: NumOrListNum = self.ensure_list_num(lo, dim)
    # hi: NumOrListNum = self.ensure_list_num(hi, dim)
    # inc: NumOrListNum = self.ensure_list_num(inc, dim)
    # mag: NumOrListNum = self.ensure_list_num(mag, dim)

    # def possible_moves(self, lo=-1, hi=1, inc=1, dims=None):
    #     dims = dims if dims is not None else range(self.dim)



# class Egocentric(GameObject):
#     def __init__(self,
#                  dim: int=2,
#                  lo: IntOrListInt=-1,
#                  hi: IntOrListInt=1,
#                  inc: IntOrListInt=1) -> List[List[int]]:



#     @staticmethod
#     def ensure_list_int(self, x: IntOrListInt) -> List[int]:
#         if isinstance(x, int):
#             x = [x]
#         return x




#     def __init__(self,
#                  loc: Location,
#                  dirs: Optional[List[str]]=None) -> None:
#         dirs = dirs or self.directionss
#         remainder = set(reachable) - self.reachable
#         assert not remainder, 'unexpected dirs {0}'.format(remainder)
#         self.location = location
#         self.dirs = dirs

# class Move(Action):
#     def __init__(self, max_per_location=1):
#         super().__init__()
#         self.max_per_location = max_per_location

#     def isvalid(from_: Location, to_: Location,
#                 player, peice, gamestate,
#                 max_per_location=None) -> bool:
#         max_per_location = max_per_location or self.max_per_location




# ##### DEPRICATED
# class Game(GameObject):
#     def __init__(self,
#                  board: Board,
#                  players: List[Player],
#                  pieces: List[Piece],
#                  actions: List[action]):
#         self.board = board
#         self.players = players
#         self.pieces = pieces
#         self.actions = actions

#         self.num_players = len(players)

#         self.turn = 0

#     def next_turn(self):
#         return (self.turn + 1) % self.num_players

#     def turn(self, player, peice, action):
#         player = players[player]


#         peice = player.peices[peice]


# class CRUD(GameObject):
#     def create(...):
#         ...

#     def read(...):
#         ...

#     def update(...):
#         ...

#     def delete(...):
#         ...


