from abc import ABC, abstractmethod
from typing import (
    Optional,
    Union,
    Any,
    List,
    Dict,
    Callable,
    Set,
    Collection,
    NewType)
import itertools

from numpy.random import RandomState

Location = NewType('Location', Any)


# the generic for stuff in the game universe
class GameObject(ABC):
    def __init__(self, rng_seed: Optional[int]=None) -> None:
        # come out the box with internal random number generator
        self.rng_seed = rng_seed or RandomState().randint(int(1e9))
        self.rng = RandomState(self.rng_seed)

    @property
    def cls(self):
        "for easier type checking"
        return self.__class__

    @property
    def cls_name(self):
        "for easier type checking"
        return self.__class__.__name__

    @property
    def id(self):
        return id(self)


## PLAYERS
class Player(GameObject):
    pass


## BOARDS
class Board(GameObject):
    "Board is a generic for the locations available on a board"
    def __init__(self, locations: Collection[Any]) -> None:
        self.locations = locations


class GridBoard(Board):
    def __init__(self, shape: List[int]) -> None:
        "init a board as an N-d grid with dims specified by shape"
        self.shape = shape
        locations = list(itertools.product(*[range(dim) for dim in shape]))
        super().__init__(locations=locations)

## base gamestate.  container for getting info about the current game
##  not meant to contain logic outside getting / querying its attrs
##  meant to get updated by game
class GameState(GameObject):
    """a dict describing a current game state.  holds the data that a
    Game needs to perform its logic"""
    def __init__(self,
                 board: Board,
                 players: List[Player],
                 turn: Optional[Player]=None,
                 player_piece_locations: Optional[
                     Dict[Player, Dict[Piece, Location]]]=None) -> None:
        super().__init__()
        self.board = Board
        self.turn = turn

        self.players = players
        self.player_piece_locations = player_piece_locations or {
            player: {} for player in players}

    def player_pieces(self, player):
        return set(self.player_piece_locations[player].keys())

    def player_locations(self, player):
        "return set of all locations occupied by a player"
        return set(self.player_piece_locations[player].values())

    def get_piece_locs(self,
                       player: Optional[Player]=None,
                       piece_type: Optional[Piece]=None,
                       action_type: Optional[Action]=None)\
            -> Dict[Piece, Location]:
        """return dict of pieces on board and their location on board
        optionally filtered on player, piece_type, and action_type"""
        players = [player] or self.players
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

    def get_pieces(self, *args, **kwargs):
        piece_locs = self.get_piece_locs(*args, **kwargs)
        return set(piece_locs.keys())

    def get_locations(self, *args, **kwargs):
        piece_locs = self.get_piece_locs(*args, **kwargs)
        return set(piece_locs.values())

    def empty_locations(self,
                        player=None,
                        piece_type=None,
                        action_type=None):
        """return locations not occupied with optional by player, piece, action
        filters"""
        return  self.board_locations - self.get_locations(
            player=player,
            piece_type=piece_type,
            action_type=action_type)

    @property
    def board_locations(self):
        "return all board locations"
        return set(self.board.locations)


class Game(GameObject):
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



class Piece(GameObject):
    def __init__(self,
                 actions: Collection[Action]):
        super().__init__()
        self.actions = actions


class Action(GameObject):
    """Action mainly exists to to contain modular logic is_valid,
    which says if this action is true given player wants piece
    to take this action given the current gamestate"""
    @abstractmethod
    def isvalid(player: Player,
                piece: Piece,
                gamestate: GameState) -> bool:
        "determine if this action is valid by this player on this piece given a game state"
        ...


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


