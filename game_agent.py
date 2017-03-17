"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import functools
import math
import random
import sys
import copy


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


class Point:
    def __init__(self, p):
        self.x = p[0]
        self.y = p[1]

    def distance(self, p):
        """Returns the euclidean distance between this point and point p."""
        this_point = (self.x, self.y)
        return compute_distance(this_point, p)

    def closest(self, point_list):
        """Returns the point closest to this point."""
        _, point = min((self.distance(p), p) for p in point_list)
        return point


def get_center(game):
    """Compute the center of the board."""
    row = game.height // 2
    col = game.width // 2
    return (row, col)


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
    return custom_score_3(game, player)


def custom_score_1(game, player):
    value = float(0)
    value += moves_diff(game, player)
    value -= distance_from_center(game, player)
    return value


def custom_score_2(game, player):
    return moves_minus_distance(game, player)


def custom_score_3(game, player):
    value = custom_score_1(game, player)
    weight = more_important_at_end(game)
    value += weight * get_blank_square_density_difference(game, player, 2)
    return value


def custom_score_4(game, player):
    value = float(0)
    value += moves_diff(game, player)
    weight = more_important_at_beginning(game)
    value -= weight * distance_from_center(game, player)
    return value


def custom_score_5(game, player):
    wb = more_important_at_beginning(game)
    we = more_important_at_end(game)
    value = float(0)
    value += moves_diff(game, player)
    value -= wb * distance_from_center(game, player)
    value += we * get_blank_square_density_difference(game, player, 2)
    return value


def custom_score_6(game, player):
    wb = more_important_at_beginning(game)
    we = more_important_at_end(game)
    value = float(0)
    value += moves_diff(game, player)
    value -= wb * distance_from_center(game, player)
    value += we * get_blank_square_density_difference(game, player, 2)
    value += attack_move_point(game, player)
    return value


def custom_score_7(game, player):
    value = float(0)
    value += moves_diff(game, player)
    weight = more_important_at_beginning(game)
    value -= weight * distance_from_center(game, player)
    value += attack_move_point(game, player)
    return value


def custom_score_8(game, player):
    value = float(0)
    value += moves_diff(game, player)
    weight = more_important_at_beginning(game)
    value -= weight * distance_from_center(game, player)
    value += weight * attack_move_point(game, player)
    return value


MOVE_DIRECTIONS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                   (1, -2),  (1, 2), (2, -1),  (2, 1)]
def attack_move_point(game, player):
    row, col = game.get_player_location(player)
    location_opponent = game.get_player_location(game.get_opponent(player))
    is_attack = False
    for r, c in MOVE_DIRECTIONS:
        if (row+r, col+c) == location_opponent:
            is_attack = True
            break
    if not is_attack:
        return 0
    if player == game.active_player:
        return -1
    else:
        return 1


def more_important_at_beginning(game):
    board_size = get_board_size(game)
    return (board_size-game.move_count) / board_size


def more_important_at_end(game):
    board_size = get_board_size(game)
    return game.move_count / board_size


def get_adjacent_squares(game, player, level):
    row, col = game.get_player_location(player)
    return [(row+r, col+c) for r in range(-level, level+1)
                           for c in range(-level, level+1)]


def get_number_of_blank_adjacent_squares(game, player, blank_squares, level):
    adjacent_squares = get_adjacent_squares(game, player, level)
    total = 0
    for s in adjacent_squares:
        if s in blank_squares:
            total += 1
    return total


def get_blank_square_density_difference(game, player, level):
    blank_squares = game.get_blank_spaces()
    num_p1 = get_number_of_blank_adjacent_squares(game, player,
                                                  blank_squares, level)
    num_p2 = get_number_of_blank_adjacent_squares(game,
                                                  game.get_opponent(player),
                                                  blank_squares, level)
    return float(num_p1 - num_p2)


def get_board_size(game):
    return game.height * game.width


def moves_diff(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player by subtracting the number of moves available to the
    opponent to the number of moves for this player.

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
    return float(my_moves - enemy_moves)


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
    return moves_diff(game, player) - get_player_distance(game)


def get_player_distance(game):
    """Computes the euclidean distance between the two players on the board.

    :param game
        The current game board.
    """
    p1 = game.get_player_location(game.active_player)
    p2 = game.get_player_location(game.inactive_player)
    return compute_distance(p1, p2)


def distance_from_center(game, player):
    player_node = game.get_player_location(game.active_player)
    center_node = get_center(game)
    return compute_distance(player_node, center_node)


def compute_distance(point_a, point_b):
    """Computes the euclidean distance between two points."""
    x_part = (point_a[0] - point_b[0])**2
    y_part = (point_a[1] - point_b[1])**2
    return math.sqrt(x_part + y_part)


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

    def __init__(self, search_depth=35, score_fn=custom_score,
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

        if not legal_moves:
            return (-1, -1)
        elif len(legal_moves) == 1:
            return legal_moves[0]

        other_moves = game.get_legal_moves(game.inactive_player)
        if game.move_count <= 1:
            if game.move_count == 0:
                # Move to center of board.
                return get_center(game)
            elif game.move_count == 1:
                # Move to center of board if not taken.
                center_point = Point(get_center(game))
                return center_point.closest(other_moves)
                #return random.choice(other_moves)
        move = (-1, -1)
        value = float('-inf')

        try:
            if not self.iterative:
                if self.method == 'minimax':
                    value, move = self.minimax(game, self.search_depth)
                else:
                    value, move = self.alphabeta(game, self.search_depth)
            else:
                moves = [[float('-inf'), m] for m in legal_moves]
                if self.method == 'minimax':
                    for depth in range(1, self.search_depth+1):
                        d = depth-1
                        moves = sorted(moves, key=lambda x: x[0], reverse=True)
                        for i, (_, m) in enumerate(moves):
                            next_value, _ = self.minimax(game.forecast_move(m), d, False)
                            if next_value > value:
                                value = next_value
                                move = m
                            # Save the new value for this move
                            moves[i][0] = value
                else:
                    b = float('inf')
                    for depth in range(1, self.search_depth+1):
                        a = float('-inf')
                        d = depth-1
                        moves = sorted(moves, key=lambda x: x[0], reverse=True)
                        for i, (_, m) in enumerate(moves):
                            next_value, _ = self.alphabeta(game.forecast_move(m), d, a, b, False)
                            if next_value > value:
                                value = next_value
                                move = m
                            # Save the new value for this move
                            moves[i][0] = value
                            if value > a:
                                a = value
                            if b <= a:
                                break
        except Timeout:
            # Handle any actions required at timeout, if necessary
            # print('GOT A TIMEOUT')
            pass

        # Return the best move from the last completed search iteration
        return move

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
        assert depth >= 0, 'depth must be a positive integer'
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        legal_moves = game.get_legal_moves()
        if maximizing_player:
            value_move = (float('-inf'), (-1, -1))
            if not legal_moves:
                return value_move
            if not depth:
                return (self.score(game, game.active_player), (1, 1))
            depth -= 1
            for m in legal_moves:
                child_board = game.forecast_move(m)
                value, _ = self.minimax(child_board, depth, False)
                value_move = max(value_move, (value, m))
            return value_move
        else:
            value_move = (float('inf'), (-1, -1))
            if not legal_moves:
                return value_move
            if not depth:
                return (self.score(game, game.inactive_player), (1, 1))
            depth -= 1
            for m in legal_moves:
                child_board = game.forecast_move(m)
                value, _ = self.minimax(child_board, depth, True)
                value_move = min(value_move, (value, m))
            return value_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        legal_moves = game.get_legal_moves()
        move = (-1, -1)
        if maximizing_player:
            value = float('-inf')
            if not legal_moves:
                return (value, move)
            if not depth:
                return (self.score(game, game.active_player), move)
            depth -= 1
            for m in legal_moves:
                next_value, _ = self.alphabeta(game.forecast_move(m), depth, alpha, beta, False)
                if next_value > value:
                    value = next_value
                    move = m
                if value > alpha:
                    alpha = value
                if beta <= alpha:
                    break
            return value, move
        else:
            value = float('inf')
            if not legal_moves:
                return value, move
            if not depth:
                return self.score(game, game.inactive_player), move
            depth -= 1
            for m in legal_moves:
                next_value, _ = self.alphabeta(game.forecast_move(m), depth, alpha, beta, True)
                if next_value < value:
                    value = next_value
                    move = m
                if value < beta:
                    beta = value
                if beta <= alpha:
                    break
            return value, move
