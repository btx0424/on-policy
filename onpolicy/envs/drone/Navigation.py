from turtle import distance
import numpy as np
import pybullet as p
from gym.spaces import Box
from gym_pybullet_drones.envs.multi_agent_rl import BaseMultiagentAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

class NavigationAviary(BaseMultiagentAviary):
    def __init__(self, config):
        act = ActionType(config.get("act", "rpm"))
        self.debug = config.get("debug", False)
        super().__init__(
            num_drones=config.get("num_agents", 2), 
            act=act, 
            gui=config.get("gui", False),
            record=self.debug)

        obs_space = self.observation_space
        num_drones = self.NUM_DRONES
        
        # state obs_shape: (n_drones, state_dim+3))
        low = np.concatenate([obs_space[0].low, -np.inf*np.ones(3)])
        high = np.concatenate([obs_space[0].high, np.inf*np.ones(3)])
        
        self.observation_space = \
            [Box(low, high, dtype=float) for _ in range(num_drones)]
        self.share_observation_space = self.observation_space
        self.action_space = \
            [self.action_space[i] for i in range(self.NUM_DRONES)]

    def reset(self):
        self.goals = np.random.random((self.NUM_DRONES, 3))*2 - np.array([1., 1., 0.])
        obs = super().reset()
        self.distance = np.linalg.norm(self.pos - self.goals, axis=1) 
        self.distance_max = self.distance
        self.success = np.zeros(self.NUM_DRONES, bool)

        if self.debug: 
            # goal indicator
            for i in range(self.NUM_DRONES):
                vshape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1., 0., 0., 1.])
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=vshape_id, basePosition=self.goals[i])

            p.removeAllUserDebugItems()
            self.line_ids = [p.addUserDebugLine(pos, goal, [1., 0., 0.], lineWidth=2.) for pos, goal in zip(self.pos, self.goals)]

        return obs
    
    def step(self, actions):
        if self.debug:
            for pos, goal, line_id in zip(self.pos, self.goals, self.line_ids):
                p.addUserDebugLine(pos, goal, [1., 0., 0.], lineWidth=2., replaceItemUniqueId=line_id)
        
        actions = {i:actions[i] for i in range(self.NUM_DRONES)}
        obs, rewards, done, info = super().step(actions)
        rewards = [rewards[i] for i in range(self.NUM_DRONES)]
        done    = [done[i] for i in range(self.NUM_DRONES)]
        return obs, rewards, done, info
        
    def _computeObs(self):
        obs = super()._computeObs()
        obs = np.array([obs[i] for i in range(self.NUM_DRONES)])
        obs = np.concatenate([obs, self.goals], axis=1) # (num_drones, state_dim+3)
        return obs

    def _computeReward(self):
        distance = np.linalg.norm(self.pos - self.goals, axis=1)
        distance_reduction = self.distance - distance
        reward = distance_reduction / self.distance_max
        self.distance = distance
        self.success |= distance < 0.1
        return reward # - distance / self.distance_max
    
    def _computeDone(self):
        bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        return [bool_val for _ in range(self.NUM_DRONES)]
    
    def _computeInfo(self):
        return [{"success":self.success[i]} for i in range(self.NUM_DRONES)]
        
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        return state
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC # 15
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC # 5

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
        
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in MeetupAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in MeetupAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in MeetupAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in MeetupAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in MeetupAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
