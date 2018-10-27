import time

from gym_hnefatafl.agents.random_agent import RandomAgent
from gym_hnefatafl.envs.hnefatafl_env import HnefataflEnv
from gym_hnefatafl.envs.board import Player


def __playgame__():

    # returns the agent whose turn it is
    def turn_agent():
        return agent1 if agent1turn else agent2

    env = HnefataflEnv()
    agent1 = RandomAgent(Player.black)
    agent2 = RandomAgent(Player.white)
    agent1turn = True

    # for i in range(1000):
    while True:
        # ask the agents for a move
        action = turn_agent().make_move(env)
        observation, reward, done, info = env.step(action)

        env.render()
        # print(env)
        # print()  # blank line
        time.sleep(.2)

        turn_agent().give_reward(reward)

        if done:
            return info

        agent1turn = not agent1turn


if __name__ == "__main__":
    print(__playgame__())
    # just for testing
    # board = HnefataflBoard()
    # print(board)
    # env = HnefataflEnv()
    # env.step(0)
