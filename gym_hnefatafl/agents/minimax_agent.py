import copy
import math
import operator

from gym_hnefatafl.envs import HnefataflEnv
from gym_hnefatafl.envs.board import Player

MINIMAX_SEARCH_DEPTH = 3


# returns the other player
def other_player(this_player):
    return Player.black if this_player == Player.white else Player.white


# returns "<" for black and ">" for white
def minimax_comp(this_player):
    return operator.__lt__() if this_player == Player.black else operator.__gt__()


# an agent that uses a minimax search for estimating which move is best
class MinimaxAgent(object):
    def __init__(self, player):
        self.player = player

    # chooses a move based on a minimax search with the __evaluate__ heuristic further below
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        minimax_action, minimax_value = self.minimax_search(env, self.player, 0)
        return minimax_action

    # does nothing yet
    def give_reward(self, reward):
        pass

    # returns the minimax action and minimax value for the given environment and the turn player.
    # The calculation is cut off at the depth that is specified at the top of this file
    # white is maximizer, black is minimizer
    def minimax_search(self, env, turn_player, depth):
        # evaluate this node using the heuristic if the max depth is reached
        if depth == MINIMAX_SEARCH_DEPTH:
            return self.__evaluate__(env.hnefatafl)

        # initialize minimax value with either positive or negative infinity
        best_minimax_value_found = math.inf if turn_player == Player.black else -math.inf
        best_action_found = None

        # loop over all actions and calculate the action with best minimax value
        for action in env.action_space:
            # deep copy so that nothing on the real board is changed
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
            subtree_minimax_action, subtree_minimax_value = self.minimax_search(env_copy, depth + 1,
                                                                                other_player(turn_player))
            # if better play is found, update minimax action and minimax value
            if minimax_comp(turn_player)(subtree_minimax_value, best_minimax_value_found):
                best_minimax_value_found = subtree_minimax_value
                best_action_found = action
        return best_action_found, best_minimax_value_found

    # estimates the value of the given board
    def __evaluate__(self, board):
        raise NotImplementedError
