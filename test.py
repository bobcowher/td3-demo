import time

import gym
import pybullet_envs
import numpy as np
from td3_torch import Agent
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
# from robosuite_environment import RoboSuiteWrapper
from robosuite.wrappers import GymWrapper


if __name__ == '__main__':

    env_name = "Lift"

    env = suite.make(
        env_name,  # Environment
        robots=["Panda"],  # Use two Panda robots
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
        # controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=True,  # Enable rendering
        use_camera_obs=False,
        horizon=200,
        render_camera="frontview",           # Camera view
        has_offscreen_renderer=True,        # No offscreen rendering
        reward_shaping=True,
        control_freq=20,  # Control frequency
    )
    env = GymWrapper(env)

    alpha=0.001
    beta=0.001
    batch_size=128
    layer1_size=256
    layer2_size=128

    agent = Agent(alpha=alpha, beta=beta, tau=0.005, input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
                  layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, noise=0.05)    # agent = Agent(input_dims=env.input_dims, env=env, n_actions=env.action_dim)
    n_games = 5
    best_score = 0
    score_history = []
    load_checkpoint = False
    episode_identifier = 4



    for i in range(n_games):
        agent.load_models()
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, validation=True)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = observation_
            time.sleep(0.03)
        # time.sleep(15)

        print(f"Episode score was {score}")

