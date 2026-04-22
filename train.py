import os
import datetime
import argparse
import numpy as np
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
import torch
import gym
import yaml
import env.airsim_env


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training thread without plot"
    )
    parser.add_argument(
        '-c',
        '--config',
        help='config file name in configs folder, such as config_default',
        default='config_Forest_SimpleMultirotor'
    )
    parser.add_argument(
        '-n',
        '--note',
        help='training objective',
        default='depth_upper_split_5'
    )

    return parser


def run(cfg):
    now = datetime.datetime.now()
    now_string = now.strftime('%Y_%m_%d_%H_%M')
    file_path = 'logs/' + now_string
    log_path = file_path + '/tb_logs'
    model_path = file_path + '/models'
    config_path = file_path + '/config'
    data_path = file_path + '/data'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(config_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    env_name = cfg['env_name']
    env = gym.make(env_name)

    activation_function = torch.nn.Tanh
    # policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[400, 300]), activation_fn=activation_function)
    policy_kwargs = dict(activation_fn=activation_function)
    policy_base = 'MultiInputPolicy'
    n_actions = env.action_space.shape[-1]
    noise_sigma = 0.1 * np.ones(n_actions)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)
    if cfg['algo'] == 'TD3':
        model = TD3(
            policy_base,
            env,
            action_noise=action_noise,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            learning_starts=12000,
            batch_size=4096,
            train_freq=100,
            gradient_steps=10,
            tensorboard_log='log_path',
            seed=123,
            verbose=2
        )
    elif cfg['algo'] == 'PPO':
        model = PPO(
            policy_base,
            env,
            batch_size=512,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log='log_path',
            seed=123,
            verbose=2
        )
    elif cfg['algo'] == 'SAC':
        model = SAC(
            policy_base,
            env,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            learning_starts=2000,
            batch_size=1280,
            train_freq=100,
            gradient_steps=10,
            tensorboard_log='log_path',
            seed=123,
            verbose=2
        )

    print('start training model')
    total_timesteps = int(cfg['total_timesteps'])
    env.model = model
    env.data_path = data_path

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=cfg['tb_log_name']
    )
    model_name = 'model'
    model.save(model_path + '/' + model_name)

    print('training finished')
    print('model saved to: {}'.format(model_path))


def main():
    config_file = 'cfg/nav_cfg.yaml'
    with open(config_file, 'r', encoding='utf-8') as config_stream:
        cfg = yaml.safe_load(config_stream) or {}
    run(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
