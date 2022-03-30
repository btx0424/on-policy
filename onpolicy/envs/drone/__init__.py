from gym_pybullet_drones.envs.multi_agent_rl import \
    FlockAviary, LeaderFollowerAviary, MeetupAviary, BaseMultiagentAviary
from gym.spaces import Box
import numpy as np
from collections import OrderedDict

name2env = {
    "flock": FlockAviary,
    "leader-follower": LeaderFollowerAviary,
    "meetup": MeetupAviary
}

def make_drone_env(name:str, config={}):
    env_class: BaseMultiagentAviary = name2env.get(name)
    class CoopAviaryEnv(env_class):
        def __init__(self) -> None:
            super().__init__(num_drones=config.num_agents)
            obs_space = self.observation_space
            num_drones = self.NUM_DRONES
            
            # obs_shape: (n_drones, n_drones*(state_dim+1))
            low = np.concatenate(
                [obs_space[i].low for i in range(num_drones)]+[np.zeros(num_drones)])
            high = np.concatenate(
                [obs_space[i].high for i in range(num_drones)]+[np.ones(num_drones)])
            
            self.observation_space = \
                [Box(low, high, dtype=float) for _ in range(num_drones)]
            self.share_observation_space = self.observation_space
            self.action_space = \
                [self.action_space[i] for i in range(self.NUM_DRONES)]

        def _computeObs(self):
            obs = super()._computeObs()
            n_drones = self.NUM_DRONES
            obs = np.concatenate([obs[i] for i in range(n_drones)])
            obs = np.repeat(obs[None], n_drones, 0) # (n_drones, n_drones*state_dim)
            obs = np.concatenate([obs, np.eye(n_drones, n_drones)], axis=1) # (n_drones, n_drones*(state_dim+1))
            return obs
        
        def step(self, actions):
            actions = {i:actions[i] for i in range(self.NUM_DRONES)}
            obs, rewards, done, info = super().step(actions)
            rewards = [rewards[i] for i in range(self.NUM_DRONES)]
            done    = [done[i] for i in range(self.NUM_DRONES)]
            return obs, rewards, done, info

    return CoopAviaryEnv()
