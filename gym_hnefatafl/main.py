import time

from gym_hnefatafl.agents.random_agent import RandomAgent
from gym_hnefatafl.envs.hnefatafl_env import HnefataflEnv
from gym_hnefatafl.envs.board import Player


# plays a game with the two given agents and returns the outcome
def __play_game__(black_agent, white_agent):

    # returns the agent whose turn it is
    def turn_agent():
        return black_agent if black_turn else white_agent

    env = HnefataflEnv()
    black_turn = True

    # for i in range(1000):
    while True:
        # ask the agents for a move
        action = turn_agent().make_move(env)
        observation, reward, done, info = env.step(action)

        # render the scene
        env.render()
        time.sleep(.1)

        # give reward
        turn_agent().give_reward(reward)

        # return if finished
        if done:
            return info

        # switch turn player
        black_turn = not black_turn


if __name__ == "__main__":
    print(__play_game__(RandomAgent(Player.black), RandomAgent(Player.white)))

