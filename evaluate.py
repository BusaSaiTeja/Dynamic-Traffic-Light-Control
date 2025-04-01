import json
import os
import numpy as np
import torch
from tqdm import tqdm
import time
import traci
from traffic_env import SumoTrafficEnv
from dqn_agent import DQNAgent

# Evaluation Settings
NUM_EPISODES = 10  # Evaluation episodes per scenario
SCENARIOS_DIR = "traffic_dataset"
SAVED_MODEL_PATH = "saved_models/dqn_final.pth"  # Path to trained model
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_scenario(scenario_path, model_path):
    """Evaluate a single traffic scenario"""
    # Generate SUMO configuration
    sumo_cfg = os.path.join(scenario_path, "sumo_config.sumocfg")
    if not os.path.exists(sumo_cfg):
        raise FileNotFoundError(f"SUMOCFG file missing for {scenario_path}")
    
    # Initialize environment
    env = SumoTrafficEnv(sumo_cfg)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Load trained model
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    
    metrics = {
        'rewards': [],
        'queue_lengths': [],
        'waiting_times': [],
        'avg_speeds': [],
        'vehicles_arrived': []
    }

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_metrics = {
            'reward': 0,
            'queues': [],
            'waiting': [],
            'speeds': [],
            'arrived': 0
        }

        while not done:
            # Get valid actions from current environment
            valid_actions = list(env.valid_phases_mapping.keys())
            
            # Get action from agent (no exploration)
            action = agent.act(state, epsilon=0, valid_actions=valid_actions)
            
            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            # Record metrics
            episode_metrics['reward'] += reward
            episode_metrics['queues'].append(sum(traci.lane.getLastStepHaltingNumber(lane) 
                                       for lane in traci.trafficlight.getControlledLanes(env.tls_id)))
            episode_metrics['waiting'].extend([
                traci.vehicle.getWaitingTime(veh)
                for veh in traci.vehicle.getIDList()
            ])
            episode_metrics['speeds'].extend([
                traci.vehicle.getSpeed(veh)
                for veh in traci.vehicle.getIDList()
            ])
            episode_metrics['arrived'] += traci.simulation.getArrivedNumber()
            
            state = next_state

        # Store episode results
        metrics['rewards'].append(episode_metrics['reward'])
        metrics['queue_lengths'].append(np.mean(episode_metrics['queues']))
        metrics['waiting_times'].append(np.mean(episode_metrics['waiting']) if episode_metrics['waiting'] else 0)
        metrics['avg_speeds'].append(np.mean(episode_metrics['speeds']) if episode_metrics['speeds'] else 0)
        metrics['vehicles_arrived'].append(episode_metrics['arrived'])
        
        env.close()
        time.sleep(1)  # Ensure proper cleanup

    return {
        'avg_reward': np.mean(metrics['rewards']),
        'avg_queue': np.mean(metrics['queue_lengths']),
        'avg_waiting_time': np.mean(metrics['waiting_times']),
        'avg_speed': np.mean(metrics['avg_speeds']),
        'total_arrived': np.sum(metrics['vehicles_arrived'])
    }

def convert_to_native_types(data):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.int32):  # Handle numpy integer types
        return int(data)
    elif isinstance(data, np.float32):  # Handle numpy float types (if any)
        return float(data)
    else:
        return data
    
if __name__ == "__main__":
    # Get all scenarios
    scenarios = [os.path.join(SCENARIOS_DIR, f"scenario_{i}") 
                for i in range(len(os.listdir(SCENARIOS_DIR))-1)]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\nEvaluating {os.path.basename(scenario)}...")
        try:
            scenario_results = evaluate_scenario(scenario, SAVED_MODEL_PATH)
            results[os.path.basename(scenario)] = scenario_results
            print(f"Results for {os.path.basename(scenario)}:")
            print(f"• Avg Reward: {scenario_results['avg_reward']:.2f}")
            print(f"• Avg Queue Length: {scenario_results['avg_queue']:.2f}")
            print(f"• Avg Waiting Time: {scenario_results['avg_waiting_time']:.2f}s")
            print(f"• Avg Speed: {scenario_results['avg_speed']:.2f}m/s")
            print(f"• Total Vehicles Arrived: {scenario_results['total_arrived']}")
        except Exception as e:
            print(f"Error evaluating {scenario}: {str(e)}")
            continue
    results = convert_to_native_types(results)
    # Save full results
    results_file = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Full results saved to {results_file}")