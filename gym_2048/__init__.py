from gym.envs.registration import register

register(
    id='2048-v0',
    entry_point='gym_2048.envs:Env_2048',
)