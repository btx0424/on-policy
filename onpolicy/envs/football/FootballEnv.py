import gym
from gym import spaces
import numpy as np
from gfootball.env import create_environment
import os.path as osp

class FootballEnv(gym.Env):
    def __init__(self, args, rank, isEval=False, render=False):
        self.isEval = isEval
        self.num_agents = args.num_agents
        self.representation = args.representation

        self.env: gym.Env = create_environment(
            env_name=args.scenario, 
            rewards="scoring" if isEval else args.rewards,
            representation=args.representation,
            write_full_episode_dumps=render,
            write_video=render,
            render=render,
            logdir=osp.join('log', args.scenario+'/'),
            number_of_left_players_agent_controls=args.num_agents,
            channel_dimensions=(48, 36)
        )
        self.action_space = list(self.env.action_space)

        obs_space = self.env.observation_space
        if self.representation == "simple115v2":
            self.observation_space = [
                gym.spaces.Box(low, high) 
                for low, high in zip(obs_space.low,obs_space.high)] 
        elif self.representation == "extracted":
            self.observation_space = [
                gym.spaces.Box(low, high) 
                for low, high in zip(obs_space.low.transpose(0,3,1,2),obs_space.high.transpose(0,3,1,2))] 
        self.share_observation_space = self.observation_space.copy()

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        if self.representation == "extracted": obs = obs.transpose(0, 3, 1, 2)
        if self.representation == "simple115v2": self.players = obs[:, 97:108].argmax(1)
        return obs

    def step(self, actions):
        obs, rewards, done, info = self.env.step(actions)
        if self.representation == "extracted": obs = obs.transpose(0, 3, 1, 2)
        # if self.representation == "simple115v2": 
        #     players = obs[:, 97:108].argmax(1)
        #     assert (players == self.players).all(), f"player switched! {players}, {self.players}"
        #     self.players = players

        if done and (rewards < 1).all(): rewards[:] = -0.02
        dones = np.array([done] * self.num_agents)
        return obs, rewards, dones, info

    def close(self):
        self.env.close()
