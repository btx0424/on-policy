import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.multi_agent_rl import BaseMultiagentAviary

class PredatorPreyAciary(BaseMultiagentAviary):
    def _computeReward(self):
        return 
    
    def _computeDone(self):
        bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False
        return done 

    def _computeInfo(self):
        return 
    
    def _line_of_sight(self):
        """ 
        Test whether predators (0 to (n-2)) have the prey (n-1) in sight
        """
        ray_from = self.pos[:-1]
        ray_to = np.ones_like(ray_from) * self.pos[-1]
        ray_hit = p.rayTestBatch(
            rayFromPositions=ray_from,
            rayToPositions=ray_to,
            physicsClientId=self.CLIENT)
        ray_hit = np.array([hit[0] for hit in ray_hit]) == self.DRONE_IDS[-1]
        return ray_hit