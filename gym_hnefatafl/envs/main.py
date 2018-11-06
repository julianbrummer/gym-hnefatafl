import gym_hnefatafl.envs.hnefatafl_env
if __name__=="__main__":
    env = gym_hnefatafl.envs.hnefatafl_env.HnefataflEnv();
    print(env.hnefatafl)
    env.render()
    env.close()

    try:
        del env
    except ImportError:
        pass