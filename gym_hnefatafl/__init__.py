from gym.envs.registration import register

register(
    id='hnefatafl-v0',
    entry_point='gym_hnefatafl.envs:HnefataflEnv',
)