import os
import numpy as np
import torch
from tqdm import tqdm
import traci
from traffic_env import SumoTrafficEnv
from dqn_agent import DQNAgent
import time
# Constants
TRAINING_DIR = "traffic_dataset"
NUM_SCENARIOS = 10
NUM_EPISODES = 1000
BATCH_SIZE = 32
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
SAVE_INTERVAL = 50  # Save model every 50 episodes

# Get all scenario directories
scenarios = [os.path.join(TRAINING_DIR, f"scenario_{i}") for i in range(NUM_SCENARIOS)]

def generate_sumo_config(scenario_path):
    # Get absolute paths relative to the scenario folder
    net_file = os.path.abspath(os.path.join(scenario_path, "network.net.xml"))
    route_file = os.path.abspath(os.path.join(scenario_path, "routes.rou.xml"))
    tll_file = os.path.abspath(os.path.join(scenario_path, "traffic_lights.tll.xml"))
    sumo_cfg = os.path.join(scenario_path, "sumo_config.sumocfg")

    # Ensure paths use forward slashes for SUMO compatibility
    net_file = net_file.replace("\\", "/")
    route_file = route_file.replace("\\", "/")
    tll_file = tll_file.replace("\\", "/")

    config_content = f"""<configuration>
        <input>
            <net-file value="{net_file}"/>
            <route-files value="{route_file}"/>
            <additional-files value="{tll_file}"/>
        </input>
        <time>
            <begin value="0"/>
            <end value="3600"/>
        </time>
    </configuration>"""

    with open(sumo_cfg, "w") as f:
        f.write(config_content)
    return sumo_cfg

# Initialize first scenario
# Initialize first scenario
scenario_idx = 0
scenario_path = scenarios[scenario_idx]
sumo_cfg = generate_sumo_config(scenario_path)  # Corrected call

# Initialize environment and agent
env = SumoTrafficEnv(sumo_cfg)
agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

epsilon = 1.0  # Initial exploration rate

for episode in tqdm(range(NUM_EPISODES), desc="Training Progress"):
    # Switch scenario every 100 episodes
    # In the training loop:
    # In your training loop:
    if episode % 100 == 0:
        scenario_idx = (episode // 100) % NUM_SCENARIOS
        scenario_path = scenarios[scenario_idx]
        
        # Close existing connection properly
        if 'env' in locals():
            try:
                env.close()
            except traci.exceptions.FatalTraCIError:
                pass  # Ignore if already closed
            time.sleep(1)  # Add small delay
    
    # Create new environment
    sumo_cfg = generate_sumo_config(scenario_path)
    env = SumoTrafficEnv(sumo_cfg)

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        try:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        except traci.exceptions.FatalTraCIError as e:
            print(f"Connection error: {str(e)}")
            done = True
            env.close()
            break

        agent.replay(BATCH_SIZE)

    agent.update_target_model()
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    if (episode + 1) % SAVE_INTERVAL == 0:
        torch.save(agent.model.state_dict(), os.path.join("saved_models", f"dqn_traffic_model_ep{episode + 1}.pth"))

env.close()
torch.save(agent.model.state_dict(), os.path.join("saved_models", "dqn_traffic_model_final.pth")) 

print("Training Complete! Model Saved.")
