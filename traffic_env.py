import os
import numpy as np
import traci
import gym
from gym import spaces

class SumoTrafficEnv(gym.Env):
    def __init__(self, sumo_cfg, max_steps=1000):
        super(SumoTrafficEnv, self).__init__()

        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0
        self.phases = [
            "GGGggrrrrrGGGggrrrrr",  # Phase 0: NS Green
            "yyyyyrrrrryyyyyrrrrr",  # Phase 1: NS Yellow
            "rrrrrGGGggrrrrrGGGgg",  # Phase 2: EW Green
            "rrrrryyyyyrrrrryyyyy",  # Phase 3: EW Yellow
            "rrrrrrrrrrrrrrrrrrrr"   # Phase 4: All Red
        ]
        self.action_space = spaces.Discrete(5)  # Update to 5 actions
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)

        # Fix: Add state_size and action_size attributes
        self.state_size = self.observation_space.shape[0]
        self.action_size = self.action_space.n

        


    def _start_sumo(self):
        if traci.isLoaded():
            traci.close()
        sumo_cmd = ["sumo", "-c", self.sumo_cfg, "--start"]
        traci.start(sumo_cmd, port=8813)
        traci.simulationStep()  # Take one step to initialize

    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except traci.exceptions.FatalTraCIError:
            pass


    def reset(self):
        self._start_sumo()
        self.step_count = 0
        # Activate program "1" defined in traffic_lights.tll.xml
        traci.trafficlight.setProgram("B", "1")
        return self._get_state()


    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward()
        done = self.step_count >= self.max_steps
        return next_state, reward, done, {}

    def _get_state(self):
        lanes = ["AB_0", "BC_0", "DB_0", "BE_0"]  
        queue_lengths = np.array([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
        return queue_lengths / 100.0  

    def _apply_action(self, action):
        phase = self.phases[action]
        
        traci.trafficlight.setRedYellowGreenState("B", phase)



    def _get_reward(self):
        total_wait = sum(traci.edge.getWaitingTime(edge) for edge in ["AB", "BC", "DB", "BE"])
        return -total_wait  

   