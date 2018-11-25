import copy
import numpy as np
from gym_hnefatafl.envs import HnefataflEnv
from gym_hnefatafl.envs.board import Outcome, Player

MONTE_CARLO_ITERATIONS = 1000

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
        self.state = board

    # simulates an entire game
    def simulate_game(self):
        # init
        # game history saves all actions done during this simulation
        game_history = []
        board_copy = copy.deepcopy(self.board)
        current_node = self.root

        # simulate actions within the tree until we are no longer at a stored node
        while board_copy.outcome == Outcome.ongoing:
            current_node, action = current_node.simulate_action(board_copy)
            game_history.append(action)
            if current_node is None:
                break

        # continue on with external nodes
        while board_copy.outcome == Outcome.ongoing:
            game_history.append(self.__choose_and_simulate_action__())

        # calculate game value
        game_value = OUTCOME_BLACK_VALUE if board_copy.outcome == Outcome.black \
            else OUTCOME_WHITE_VALUE if board_copy.outcome == Outcome.white \
            else OUTCOME_DRAW_VALUE

        # back value up
        # TODO: this part is probably not correct and needs to be changed to the algorithm on page 7 of the paper
        current_node = self.root
        for action in game_history:
            current_node.update_value(game_value)
            current_node = current_node.children[action]
            if current_node is None:
                break

    # chooses an action and simulates it. Then returns the action
    def __choose_and_simulate_action__(self):
        raise NotImplementedError

    # returns the best action found
    def get_best_action(self):
        raise NotImplementedError


# represents a node within the monte carlo search tree (that is actually stored in memory -> see the paper)
class Node(object):
    def __init__(self, player):
        self.children = {}
        self.number_of_visits = 0
        self.sum_of_values = 0
        self.sum_of_squared_values = 0
        self.player = player
        self.is_internal = False

    # chooses and simulates an action
    def choose_and_simulate_action(self, board):
        # TODO: self.is_internal needs to be set after a certain amount of visits or according to another heuristic
        self.number_of_visits += 1
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
        mus = np.empty(len(actions))
        sigmas_squared = np.empty(len(actions))
        if self.is_internal:
            # TODO: implement heuristic or just do the else part if no time left or so
            raise NotImplementedError
        else:
            for i, action in enumerate(actions):
                ################################################################################################
                # parameter that somehow needs to reflect "points on the board", i. e. empty intersections in go
                # could possibly be chosen as "number of pieces on the board"
                p = 0
                ################################################################################################
                if action in self.children:
                    child = self.children[action]
                    mu = child.sum_of_values / child.number_of_visits
                    sigma_squared = (child.sum_of_squared_values - child.number_of_visits*mu*mu + 4*p*p) \
                                / (child.number_of_visits + 1)
                else:
                    mu = 0
                    sigma_squared = - mu*mu + 4*p*p
                mus[i] = mu
                sigmas_squared[i] = sigma_squared
        max_index = np.argmax(mus)
        mu_max = mus[max_index]
        sigma_of_mu_max = sigmas_squared[max_index]
        ########################################################################################################
        # parameter that somehow needs to reflect "urgency of a move". mustn't be zero
        # could possibly be chosen as "pieces captured" along with the scaling factor described in the paper
        e = np.ones(len(actions))
        ########################################################################################################
        probabilities = np.exp(-2.4 * (mu_max - mus) / np.math.sqrt(2*(sigma_of_mu_max+sigmas_squared))) + e
        prob_sum = np.sum(probabilities)
        probabilities /= prob_sum

        random_action = np.random.choice(actions, p=probabilities)
        return random_action

    # updates the internal values
    def update_value(self, value):
        self.sum_of_values += value
        self.sum_of_squared_values += value * value


# returns the opponent of the given player
def other_player(player):
    return Player.white if player == Player.black else Player.white


class MonteCarloAgent(object):
    # chooses a random move and returns it
    # in order for games to finish in a reasonable amount of time,
    # the agent always sends the king to one of the corners if able
    # (this causes white to win basically all the time)
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        tree = Tree(env.get_board())
        for i in range(MONTE_CARLO_ITERATIONS):
            tree.simulate_game()
        return tree.get_best_action()

    # does nothing in this agent, but is here because other agents need it
    def give_reward(self, reward):
        pass
