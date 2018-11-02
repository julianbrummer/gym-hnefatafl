import random

from gym_hnefatafl.envs import HnefataflEnv


class RandomAgent(object):
    def __init__(self, player):
        self.player = player
        self.corners = [(1, 1), (1, 11), (11, 1), (11, 11)]

    # chooses a random move and returns it
    # in order for games to finish in a reasonable amount of time,
    # the agent always sends the king to one of the corners if able
    # (this causes white to win basically all the time)
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        # for pos_from, pos_to in env.action_space:
        #    if pos_to in self.corners:
        #        return pos_from, pos_to
        return random.choice(env.action_space)

    # does nothing in this agent, but is here because other agents need it
    def give_reward(self, reward):
        pass
