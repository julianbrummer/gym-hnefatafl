from gym_hnefatafl.envs.hnefatafl_env import HnefataflEnv
from gym_hnefatafl.envs.board import HnefataflBoard

if __name__ == "__main__":
    # just for testing
    board = HnefataflBoard()
    print(board)
    env = HnefataflEnv()
    env.step(0)
