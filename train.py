import os
import numpy as np
import torch
from tqdm import tqdm
import time
from traffic_env import SumoTrafficEnv
from dqn_agent import DQNAgent
import traci

# Constants
TRAINING_DIR = "traffic_dataset"
NUM_SCENARIOS = 10
NUM_EPISODES = 1000
BATCH_SIZE = 32
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
SAVE_INTERVAL = 50
SAVE_DIR = "saved_models"

# Initialize
os.makedirs(SAVE_DIR, exist_ok=True)
scenarios = [os.path.join(TRAINING_DIR, f"scenario_{i}") for i in range(NUM_SCENARIOS)]

def generate_sumo_config(scenario_path):
    net_file = os.path.abspath(os.path.join(scenario_path, "network.net.xml"))
    route_file = os.path.abspath(os.path.join(scenario_path, "routes.rou.xml"))
    tll_file = os.path.abspath(os.path.join(scenario_path, "traffic_lights.tll.xml"))
    sumo_cfg = os.path.join(scenario_path, "sumo_config.sumocfg")

    config_content = f'''<configuration>
        <input>
            <net-file value="{net_file.replace("\\", "/")}"/>
            <route-files value="{route_file.replace("\\", "/")}"/>
            <additional-files value="{tll_file.replace("\\", "/")}"/>
        </input>
        <time>
            <begin value="0"/>
            <end value="3600"/>
        </time>
    </configuration>'''

    with open(sumo_cfg, 'w') as f:
        f.write(config_content)
    return sumo_cfg

# Training loop
env = None
agent = None
epsilon = 1.0

for episode in tqdm(range(NUM_EPISODES), desc="Training Progress"):
    # Scenario switching
    if episode % 100 == 0 or env is None:
        scenario_idx = (episode // 100) % NUM_SCENARIOS
        scenario_path = scenarios[scenario_idx]
        
        # Cleanup previous
        if env is not None:
            try:
                env.close()
                time.sleep(1)
            except:
                pass
        
        # Initialize new
        sumo_cfg = generate_sumo_config(scenario_path)
        env = SumoTrafficEnv(sumo_cfg)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        epsilon = 1.0  # Reset exploration

    # Episode setup
    state = env.reset()
    total_reward = 0
    done = False
    valid_actions = list(env.valid_phases_mapping.keys())

    while not done:
        try:
            action = agent.act(state, epsilon, valid_actions)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay(BATCH_SIZE, valid_actions)
        except traci.exceptions.FatalTraCIError as e:
            print(f"Connection error: {str(e)}")
            done = True
            
            env.close()

    # Post-episode
    agent.update_target_model()
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    
    if (episode + 1) % SAVE_INTERVAL == 0:
        torch.save(agent.model.state_dict(), 
                 os.path.join(SAVE_DIR, f"dqn_ep{episode+1}.pth"))

# Final save
if env is not None:
    env.close()
torch.save(agent.model.state_dict(), os.path.join(SAVE_DIR, "dqn_final.pth"))
print("Training completed successfully!")