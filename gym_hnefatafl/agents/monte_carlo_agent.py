import copy
import random

import numpy as np
from gym_hnefatafl.envs import HnefataflEnv
from gym_hnefatafl.envs.board import Outcome, Player

MONTE_CARLO_ITERATIONS = 10
MIN_NUM_VISITS_INTERNAL = 5  # may have to be much higher go uses 9*9

#################################################################################################################
#################################################################################################################
#################################################################################################################
OUTCOME_BLACK_VALUE = -1
OUTCOME_WHITE_VALUE = 1
OUTCOME_DRAW_VALUE = 0

# An den Werten hier muss bestimmt noch was geändert werden, weil auch die Quadrate davon gespeichert werden!!!
#################################################################################################################
#################################################################################################################
#################################################################################################################


# represents a monte carlo search tree
class Tree(object):
    # board: the current board
    # player: the player that this agent represents
    def __init__(self, board, player):
        self.root = Node(player)
        self.board = board
        self.num_simulations = 0
        self.player = player

    # simulates an entire game
    def simulate_game(self):
        # init
        self.num_simulations += 1
        # game history saves all actions done during this simulation
        game_history = []
        board_copy = copy.deepcopy(self.board)
        current_node = self.root

        # simulate actions within the tree until we are no longer at a stored node
        while board_copy.outcome == Outcome.ongoing:
            self.player = current_node.player
            current_node, action = current_node.choose_and_simulate_action(board_copy)
            game_history.append(action)
            if current_node is None:
                self.player = other_player(self.player)
                break

        # finish game
        while board_copy.outcome == Outcome.ongoing:
            game_history.append(self.__choose_and_simulate_action__(board_copy))
            self.player = other_player(self.player)

        # calculate game value
        game_value = OUTCOME_BLACK_VALUE if board_copy.outcome == Outcome.black \
            else OUTCOME_WHITE_VALUE if board_copy.outcome == Outcome.white \
            else OUTCOME_DRAW_VALUE

        current_node = self.root
        for action in game_history:
            current_node.update_value(game_value)
            current_node.update_mean_variance(self.num_simulations, board_copy)
            current_node.update_action_probabilities()
            if action not in current_node.children:
                break
            current_node = current_node.children[action]

        # # back value up
        # # TODO: this part is now implemented according to the paper however not yet invoked in the rest of the code
        # width = 11
        # height = 11
        # value = 0
        # current_node = self.root
        # meanValue = 2 * width * height
        # if self.simulations>16 * width * height:
        #     meanValue *= self.simulations / (16 * width * height)
        # value = meanValue
        # if self.tGames[1] and  self.TGames[0]:
        #     tAveragedValue=[2]
        #     for i in tAveragedValue:
        #         tAveragedValue[i]=(self.tGames[i] * self.tValues[i] + meanValue * value) / (self.tGames[i] + meanValue)
        #     if self.tGames[1]> self.tGames[0]:
        #         if self.tValues[1]> value:
        #             value=tAveragedValue[1]
        #         elif self.tValues[0]<value:
        #             value=tAveragedValue[0]
        #     else:
        #         value=tAveragedValue[0]
        # else:
        #     value=self.tValues[0]
        # return value


    # chooses an action and simulates it. Then returns the action
    def __choose_and_simulate_action__(self, board):
        actions = board.get_valid_actions(self.player)
        action = random.choice(actions)
        board.do_action(action, self.player)

    # returns the best action found
    def get_best_action(self):
        max_mu = -np.math.inf
        best_action = None
        for action in self.root.children:
            if max_mu < self.root.children[action].mean:
                max_mu = self.root.children[action].mean
                best_action = action
        return best_action
        # raise NotImplementedError


# represents a node within the monte carlo search tree (that is actually stored in memory -> see the paper)
class Node(object):
    def __init__(self, player):
        self.children = {}
        self.number_of_visits = 0
        self.sum_of_values = 0
        self.sum_of_squared_values = 0
        self.mean = 0.0
        self.variance = 0.0
        self.player = player
        self.is_internal = False
        self.sorted_child_indices = []
        self.action_probabilities = []

        # self.tValue = []
        # self.tGames = []
        # self.simulations=0

    # chooses and simulates an action
    def choose_and_simulate_action(self, board):
        self.number_of_visits += 1
        self.is_internal = self.number_of_visits > MIN_NUM_VISITS_INTERNAL
        action = self.__choose_action__(board)
        board.do_action(action, self.player)
        # create child node if this node has already been visited
        if self.number_of_visits > 1 and action not in self.children:
            child_node = Node(other_player(self.player))
            self.children[action] = child_node
            return child_node, action
        else:
            return None, action

    # chooses an action
    def __choose_action__(self, board):
        actions = board.get_valid_actions(self.player)
        # mus = np.empty(len(actions))
        # sigmas_squared = np.empty(len(actions))
        if self.is_internal:  # do random move based on probability distribution
            action_index = np.random.choice(len(self.children),  p=self.action_probabilities)
            index = self.sorted_child_indices[action_index]
            sorted_actions = [c for c in self.children.keys()]
            return sorted_actions[index]
        else:  # do random move for external nodes
            return random.choice(actions)
        # if self.is_internal:
        #
        # else:
        #     for i, action in enumerate(actions):
        #         ################################################################################################
        #         # parameter that somehow needs to reflect "points on the board", i. e. empty intersections in go
        #         # could possibly be chosen as "number of pieces on the board"
        #         p = 0
        #         ################################################################################################
        #         if action in self.children:
        #             child = self.children[action]
        #             mu = child.sum_of_values / child.number_of_visits
        #             sigma_squared = (child.sum_of_squared_values - child.number_of_visits*mu*mu + 4*p*p) \
        #                         / (child.number_of_visits + 1)
        #         else:
        #             mu = 0
        #             sigma_squared = - mu*mu + 4*p*p
        #         mus[i] = mu
        #         sigmas_squared[i] = sigma_squared
        # max_index = np.argmax(mus)
        # mu_max = mus[max_index]
        # sigma_of_mu_max = sigmas_squared[max_index]
        # ########################################################################################################
        # # parameter that somehow needs to reflect "urgency of a move". mustn't be zero
        # # could possibly be chosen as "pieces captured" along with the scaling factor described in the paper
        # e = np.ones(len(actions))
        # ########################################################################################################
        # probabilities = np.exp(-2.4 * (mu_max - mus) / np.math.sqrt(2*(sigma_of_mu_max+sigmas_squared))) + e
        # prob_sum = np.sum(probabilities)
        # probabilities /= prob_sum
        #
        # random_action = np.random.choice(actions, p=probabilities)
        # return random_action


    # updates the internal values
    def update_value(self, value):
        self.sum_of_values += value
        self.sum_of_squared_values += value * value

    def update_mean_variance(self, num_simulations, board):
        ################################################################################################
        # parameter that somehow needs to reflect "points on the board", i. e. empty intersections in go
        # could possibly be chosen as "number of pieces on the board"
        p = 11*11 - (board.white_pieces + board.black_pieces)
        ################################################################################################
        self.mean = self.sum_of_values/num_simulations
        self.variance = (self.sum_of_squared_values - self.number_of_visits*self.mean*self.mean + 4*p*p) \
                        / (num_simulations + 1)

    def update_action_probabilities(self):
        if len(self.children) > 0:
            mean = [c.mean for c in self.children.values()]
            sigma = [c.variance for c in self.children.values()]
            # sort actions/children by mean value of children
            self.sorted_child_indices = np.argsort(mean)
            mean0 = mean[self.sorted_child_indices[0]]
            sigma0 = sigma[self.sorted_child_indices[0]]
            # compute probabilities for each move
            self.action_probabilities = np.asarray([self.probability(mean0, sigma0, self.sorted_child_indices[i], mean, sigma) \
                                         for i in range(0,len(mean))])
            self.action_probabilities /= np.sum(self.action_probabilities)


    def probability(self, m0, s0, i, mean, sigma):
        ########################################################################################################
        # parameter that somehow needs to reflect "urgency of a move". mustn't be zero
        # could possibly be chosen as "pieces captured" along with the scaling factor described in the paper
        a = 1
        eps = (0.1 + 1/pow(2, i) + a)/len(self.children)
        ########################################################################################################
        #print(i)
        #print(self.children)
        #print(self.children.values()[i])
        mi = mean[i]
        si = sigma[i]
        return np.exp(-2.4*(m0-mi)/np.sqrt(2*(s0*s0+si*si))) + eps

# returns the opponent of the given player
def other_player(player):
    return Player.white if player == Player.black else Player.white


class MonteCarloAgent(object):
    def __init__(self, player):
        self.player = player

    # chooses a random move and returns it
    # in order for games to finish in a reasonable amount of time,
    # the agent always sends the king to one of the corners if able
    # (this causes white to win basically all the time)
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        tree = Tree(env.get_board(), self.player)
        for i in range(MONTE_CARLO_ITERATIONS):
            tree.simulate_game()
        return tree.get_best_action()

    # does nothing in this agent, but is here because other agents need it
    def give_reward(self, reward):
        pass