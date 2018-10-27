import random

from gym_hnefatafl.envs import HnefataflEnv


class RandomAgent(object):
    def __init__(self, player):
        self.player = player

    # chooses a random move and returns it
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        return random.choice(env.action_space)

    # does nothing in this agent, but is here because other agents need it
    def give_reward(self, reward):
        pass
