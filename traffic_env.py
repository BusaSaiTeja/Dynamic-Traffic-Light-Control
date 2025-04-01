import os
import numpy as np
import gym
from gym import spaces
import traci
import json

class SumoTrafficEnv(gym.Env):
    def __init__(self, sumo_cfg):
        super(SumoTrafficEnv, self).__init__()
        
        self.sumo_cfg = sumo_cfg
        self.step_count = 0
        self.max_steps = 1000
        self.max_lanes = 20
        
        # Load scenario configuration
        self._load_scenario_config()
        
        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(self.max_lanes,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_actions)
        
        # Traffic light settings
        self.tls_id = "B"
        self.phases = self._load_phases()

    def _load_phases(self):
        return [
            "GGGggrrrrrGGGggrrrrr",  # Phase 0
            "yyyyyrrrrryyyyyrrrrr",   # Phase 1
            "rrrrrGGGggrrrrrGGGgg",  # Phase 2
            "rrrrryyyyyrrrrryyyyy",   # Phase 3
            "rrrrrrrrrrrrrrrrrrrr"    # Phase 4
        ]

    def _load_scenario_config(self):
        config_path = os.path.join(
            os.path.dirname(self.sumo_cfg), 
            "scenario_config.json"
        )
        with open(config_path) as f:
            config = json.load(f)
            
        self.valid_actions = list(map(int, config["valid_phases"]))
        self.raw_state_size = config["state_size"]
        self.phase_mapping = {int(k): v for k, v in config["phase_mapping"].items()}
        self.max_actions = config["max_actions"]

    def _start_sumo(self):
        if traci.isLoaded():
            traci.close()
        sumo_cmd = ["sumo", "-c", self.sumo_cfg, "--start"]
        traci.start(sumo_cmd, port=8813)
        traci.simulationStep()

    def reset(self):
        self._start_sumo()
        self.step_count = 0
        traci.trafficlight.setProgram(self.tls_id, "1")
        return self._get_state()

    def _get_state(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)[:self.raw_state_size]
        
        queue_lengths = np.array([
            traci.lane.getLastStepHaltingNumber(lane) 
            for lane in lanes if traci.lane.getLength(lane) > 0
        ])
        
        padded_state = np.pad(
            queue_lengths, 
            (0, self.max_lanes - len(queue_lengths)),
            mode='constant'
        )
        return padded_state / 100.0

    def step(self, action):
        if action not in self.phase_mapping:
            raise ValueError(f"Invalid action {action} for current junction")
            
        phase_idx = self.phase_mapping[action]
        self._apply_action(phase_idx)
        traci.simulationStep()
        self.step_count += 1
        
        next_state = self._get_state()
        reward = self._get_reward()
        done = self.step_count >= self.max_steps
        
        return next_state, reward, done, {}

    def _apply_action(self, phase_idx):
        phase = self.phases[phase_idx]
        if len(phase) != len(self.phases[0]):
            raise ValueError("Phase length mismatch")
        traci.trafficlight.setRedYellowGreenState(self.tls_id, phase)

    def _get_reward(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        return -sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)

    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

    @property
    def valid_phases(self):
        return list(self.phase_mapping.keys())