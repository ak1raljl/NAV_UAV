from gym.envs.registration import register

register(
    id='airsim_env',
    entry_point='env.airsim_env:AirsimEnv',
    max_episode_steps=300
)