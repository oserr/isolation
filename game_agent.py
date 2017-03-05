"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math
import sys


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')
    return moves_minus_distance(game, player)


def moves_minus_distance(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    :param game
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    :param player
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    :return
        The heuristic value of the current game state to the specified player.
    """
    my_moves = len(game.get_legal_moves(player))
    enemy_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(my_moves - enemy_moves) - get_player_distance(game)


def get_player_distance(game):
    """Computes the euclidean distance between the two players on the board.

    :param game
        The current game board.
    """
    p1 = game.get_player_location(game.active_player)
    p2 = game.get_player_location(game.inactive_player)
    x_part = (p1[0] - p2[0])**2
    y_part = (p1[1] - p2[1])**2
    return math.sqrt(x_part + y_part)


def compute_location_value(game, player):
    """Computes the value of the player's location, with higher values for
    locations that are closer to the center.

    :param game
        The current game board.
    :param player
        The player whose location is being evaluated.
    :return
        The value of the location as a float.
    """
    # Don't evaluate for boards that are particularly small.
    if game.height < 4 or game.width < 4:
        return 0
    cutoff_point = CutoffPoint(game)
    location = game.get_player_location(player)
    if cutoff_point.is_center(location):
        return .75
    elif cutoff_point.is_inner_edge(location):
        return .50
    else:
        return .25


class CutoffPoint:
    """A utility class that computes two cutoff points where the board is
    roughly split into outer edge, inner edge, and center.
    """

    def __init__(self, game):
        """Initializes the CutoffPoint instance with a game, which are used to
        obtain dimensions of the board and the current location of the player.

        :param game
            The current board game.
        """
        if game.height < 4 or game.width < 4:
            raise ValueError('board is too small for this class to be useful')
        self.num_rows = game.height
        self.num_cols = game.width
        self.center_row = self.num_rows // 2
        self.center_col = self.num_cols // 2
        self.init_cutoff_points()

    def init_cutoff_points(self):
        """Initializes the center and inner edge points."""
        y_unit = self.num_rows / 3
        x_unit = self.num_cols / 3
        self.center_points = (self.center_row+y_unit, self.center_col+x_unit)
        self.edge_points = (
            self.center_points[0] + y_unit,
            self.center_points[1] + x_unit
        )

    def is_center(self, location):
        """Returns true if player is located toward center of board, false
        otherwise.

        :param location
            A (row, column) tuple.
        """
        location = self.normalize_location(location)
        return location[0] <= self.center_points[0] \
            and location[1] <= self.center_points[1]

    def is_inner_edge(self, location):
        """Returns true if player is located somewhere in the inner edge of the
        board, false otherwise.

        :param location
            A (row, column) tuple.
        """
        location = self.normalize_location(location)
        return location[0] <= self.edge_points[0] \
            and location[1] <= self.edge_points[1]

    def normalize_location(self, location):
        """Normalizes a board location with respect to the center row and
        column.

        :param location
            A (row, column) tuple.
        :return
            A (row, column) tuple, where the row and column represent the
            absolute distance from the center row and columns.
        """
        normal_row = abs(location[0] - self.center_row)
        normal_col = abs(location[1] - self.center_col)
        return (normal_row, normal_col)


def choose_best(moves, maximize=True):
    """Choses the best move among a list of moves.

    :param moves
        A list of tuples (value, move)
        - value is the value of the move
        - move is the move
    :param maximize
        Boolean value that indicates if best represents a maximum or minimum
        value.
    :return
        The move with the highest value. If more than one move have the
        highest value, then select one randomly.
    """
    assert len(moves) > 0, 'moves needs to contain at least one move'
    length = len(moves)
    if length == 1:
        return moves[0]
    moves = sorted(moves, reverse=maximize)
    i = 1
    while i < length:
        valueLast = moves[i-1][0]
        valueNext = moves[i][0]
        if valueLast != valueNext:
            break
        i += 1
    else:
        return moves[0]
    return random.choice(moves[0:i])


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth if search_depth > 0 else sys.maxsize
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        if game.move_count <= 1:
            if game.move_count == 0:
                # Move to center of board.
                return (game.height // 2, game.width // 2)
            elif game.move_count == 1:
                # Move to center of board if not taken.
                square = (game.height // 2, game.width // 2)
                if square in game.get_blank_spaces():
                    return square
                # Opponent has center square, so move to non-symmetric square.
            #else:
        search_fn = self.minimax if self.method == 'minimax' else self.alphabeta
        value_move = (float('-inf'), (-1, -1))

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if not self.iterative:
                value_move = search_fn(game, self.search_depth)
            else:
                for i in range(1, self.search_depth+1):
                    other_move = search_fn(game, i)
                    value_move = max(value_move, other_move)
        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return value_move[1]

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        assert depth > 0, 'depth must be a positive integer'
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (float('-inf'), (-1, -1))
        depth -= 1
        board_moves = ((game.forecast_move(m), m) for m in legal_moves)
        moves = [(self.minvalue(b, depth), m) for b, m in board_moves]
        return choose_best(moves)

    def maxvalue(self, game, depth):
        """Computes the maximizing value of the current state of the game.

        :param game
            A game board.
        :param depth
            The remaining number of steps to explore before game is evaluated
            with heuristic function.
        :return
            The maximum value of a backed-up state.
        """
        assert depth >= 0, 'depth cannot be a negative value'
        value = float('-inf')
        legal_moves = game.get_legal_moves()
        if not legal_moves or not depth:
            return self.score(game, game.active_player)
        depth -= 1
        for m in legal_moves:
            game_next_move = game.forecast_move(m)
            value = max(value, self.minvalue(game_next_move, depth))
        return value


    def minvalue(self, game, depth):
        """Computes the minimum value of the current state of the game.

        :param game
            A game board.
        :param depth
            The remaining number of steps to explore before game is evaluated
            with heuristic function.
        :return
            The minimum value of a backed-up state.
        """
        assert depth >= 0, 'depth cannot be a negative value'
        value = float('inf')
        legal_moves = game.get_legal_moves()
        if not legal_moves or not depth:
            return self.score(game, game.inactive_player)
        depth -= 1
        for m in legal_moves:
            game_next_move = game.forecast_move(m)
            value = min(value, self.maxvalue(game_next_move, depth))
        return value


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        assert depth > 0, 'depth must be positive'
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        value_move = (float('-inf'), (-1, -1))
        depth -= 1
        moves = []
        for m in game.get_legal_moves():
            game_next_move = game.forecast_move(m)
            next_value = self.minbeta(game_next_move, depth, alpha, beta)
            value_move = max(value_move, (next_value, m))
            moves.append(value_move)
            value = value_move[0]
            if value >= beta:
                break
            alpha = max(alpha, value)
        if moves:
            value_move = choose_best(moves)
        return value_move

    def maxalpha(self, game, depth, alpha, beta):
        """Computes the maximizing value of the current state of the game.

        :param game
            A game board.
        :param depth
            The remaining number of steps to explore before game is evaluated
            with heuristic function.
        :param alpha
            The current alpha value.
        :param beta
            The current beta value.
        :return
            The maximum value of a backed-up state.
        """
        assert depth >= 0, 'depth cannot be a negative value'
        value = float('-inf')
        legal_moves = game.get_legal_moves()
        if not legal_moves or not depth:
            return self.score(game, game.active_player)
        depth -= 1
        for m in legal_moves:
            game_next_move = game.forecast_move(m)
            next_value = self.minbeta(game_next_move, depth, alpha, beta)
            value = max(value, next_value)
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def minbeta(self, game, depth, alpha, beta):
        """Computes the minimum value of the current state of the game.

        :param game
            A game board.
        :param depth
            The remaining number of steps to explore before game is evaluated
            with heuristic function.
        :param alpha
            The current alpha value.
        :param beta
            The current beta value.
        :return
            The minimum value of a backed-up state.
        """
        assert depth >= 0, 'depth cannot be a negative value'
        value = float('inf')
        legal_moves = game.get_legal_moves()
        if not legal_moves or not depth:
            return self.score(game, game.active_player)
        depth -= 1
        for m in legal_moves:
            game_next_move = game.forecast_move(m)
            next_value = self.maxalpha(game_next_move, depth, alpha, beta)
            value = min(value, next_value)
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value
