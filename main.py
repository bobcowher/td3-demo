import time
import os
import gym
import pybullet_envs
import numpy as np
from td3_torch import Agent
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
# from robosuite_environment import RoboSuiteWrapper
from robosuite.wrappers import GymWrapper


if __name__ == '__main__':

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "Lift"

    env = suite.make(
        env_name,  # Environment
        robots=["Panda"],  # Use two Panda robots
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
        # controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,  # Enable rendering
        use_camera_obs=False,
        horizon=300,
        # render_camera="sideview",           # Camera view
        # has_offscreen_renderer=True,        # No offscreen rendering
        reward_shaping=True,
        control_freq=20,  # Control frequency
    )
    env = GymWrapper(env)

    agent = Agent(alpha=0.001, beta=0.001, tau=0.005, input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], layer1_size=400, layer2_size=300)
    # agent = Agent(input_dims=env.input_dims, env=env, n_actions=env.action_dim)
    writer = SummaryWriter('logs')
    n_games = 3000
    best_score = 0
    score_history = []
    load_checkpoint = False
    episode_identifier = 0

    models_loaded = agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_
        score_history.append(score)
        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if(len(score_history)>100):
            avg_score = np.mean(score_history[-100])
        else:
            avg_score = np.mean(score_history)



        if(i % 10 == 0):
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)