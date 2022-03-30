from gym_pybullet_drones.envs.multi_agent_rl import \
    FlockAviary, LeaderFollowerAviary, MeetupAviary, BaseMultiagentAviary
from matplotlib.pyplot import cla
import numpy as np

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
            self.observation_space = \
                [self.observation_space[i] for i in range(self.NUM_DRONES)]
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
        
    return CoopAviaryEnv()
