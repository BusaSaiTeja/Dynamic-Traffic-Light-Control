import json
import os
import numpy as np
import torch
from tqdm import tqdm
import time
import traci
import matplotlib.pyplot as plt
from traffic_env import SumoTrafficEnv
from dqn_agent import DQNAgent

NUM_SCENARIOS = 10
NUM_EPISODES = 10
SCENARIOS_DIR = "traffic_dataset"
SAVED_MODEL_PATH = "saved_models/dqn_final.pth"
RESULTS_DIR = "evaluation_results"
MAX_ACTIONS = 5
os.makedirs(RESULTS_DIR, exist_ok=True)

# New configuration
STATIC_PHASE_DURATION = 30  # Seconds for each phase in static control

def load_scenario_config(scenario_path):
    config_path = os.path.join(scenario_path, "scenario_config.json")
    with open(config_path) as f:
        return json.load(f)

def evaluate_controller(scenario_path, controller_fn, **kwargs):
    sumo_cfg = os.path.join(scenario_path, "sumo_config.sumocfg")
    env = SumoTrafficEnv(sumo_cfg)
    
    metrics = {
        'queue_lengths': [],
        'waiting_times': [],
        'avg_speeds': [],
        'vehicles_arrived': []
    }

    for _ in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_metrics = {
            'queues': [],
            'waiting': [],
            'speeds': [],
            'arrived': 0
        }

        while not done:
            action = controller_fn(env, **kwargs)
            next_state, reward, done, _ = env.step(action)
            
            # Record metrics
            episode_metrics['queues'].append(sum(
                traci.lane.getLastStepHaltingNumber(lane) 
                for lane in traci.trafficlight.getControlledLanes(env.tls_id)
            ))
            episode_metrics['waiting'].extend([
                traci.vehicle.getWaitingTime(veh)
                for veh in traci.vehicle.getIDList()
            ])
            episode_metrics['speeds'].extend([
                traci.vehicle.getSpeed(veh)
                for veh in traci.vehicle.getIDList()
            ])
            episode_metrics['arrived'] += traci.simulation.getArrivedNumber()

        # Store episode results
        metrics['queue_lengths'].append(np.mean(episode_metrics['queues']))
        metrics['waiting_times'].append(np.nanmean(episode_metrics['waiting']) if episode_metrics['waiting'] else 0)
        metrics['avg_speeds'].append(np.nanmean(episode_metrics['speeds']) if episode_metrics['speeds'] else 0)
        metrics['vehicles_arrived'].append(episode_metrics['arrived'])
        
        env.close()
        time.sleep(1)

    return {
        'avg_queue': float(np.nanmean(metrics['queue_lengths'])),
        'avg_waiting_time': float(np.nanmean(metrics['waiting_times'])),
        'avg_speed': float(np.nanmean(metrics['avg_speeds'])),
        'total_arrived': int(np.sum(metrics['vehicles_arrived']))
    }

# Controller implementations
def dqn_controller(env, agent):
    return agent.act(env._get_state(), 0, env.valid_phases)

def static_controller(env, phase_timer, valid_phases):
    current_time = traci.simulation.getTime()
    if current_time - phase_timer['last_change'] >= STATIC_PHASE_DURATION:
        phase_timer['index'] = (phase_timer['index'] + 1) % len(valid_phases)
        phase_timer['last_change'] = current_time
    return valid_phases[phase_timer['index']]

def plot_comparison(results, metric):
    scenarios = list(results.keys())
    static_vals = [results[scen]['static'][metric] for scen in scenarios]
    dqn_vals = [results[scen]['dqn'][metric] for scen in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, static_vals, width, label='Static')
    rects2 = ax.bar(x + width/2, dqn_vals, width, label='DQN')

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison of {metric.replace("_", " ")} by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_comparison.png"))
    plt.close()

if __name__ == "__main__":
    scenarios = [os.path.join(SCENARIOS_DIR, f"scenario_{i}") 
                for i in range(NUM_SCENARIOS)]
    comparison_results = {}

    # Initialize DQN agent once
    agent = DQNAgent(state_dim=20, max_action_dim=MAX_ACTIONS)
    agent.model.load_state_dict(
        torch.load(SAVED_MODEL_PATH, map_location=agent.device, weights_only=True),
        strict=False
    )
    agent.model.eval()

    for scenario in scenarios:
        scen_name = os.path.basename(scenario)
        print(f"\nEvaluating {scen_name}...")
        
        try:
            # DQN Evaluation
            dqn_results = evaluate_controller(
                scenario,
                dqn_controller,
                agent=agent
            )
            
            # Static Control Evaluation
            config = load_scenario_config(scenario)
            static_results = evaluate_controller(
                scenario,
                static_controller,
                valid_phases=config['valid_phases'],
                phase_timer={'index': 0, 'last_change': 0}
            )
            
            comparison_results[scen_name] = {
                'static': static_results,
                'dqn': dqn_results
            }
            
            # Print results
            print(f"\n{scen_name} Results:")
            print("Metric\t\tStatic\t\tDQN")
            print(f"Queue\t\t{static_results['avg_queue']:.1f}\t\t{dqn_results['avg_queue']:.1f}")
            print(f"Wait Time\t{static_results['avg_waiting_time']:.1f}s\t\t{dqn_results['avg_waiting_time']:.1f}s")
            print(f"Speed\t\t{static_results['avg_speed']:.2f}m/s\t{dqn_results['avg_speed']:.2f}m/s")
            print(f"Arrived\t\t{static_results['total_arrived']}\t\t{dqn_results['total_arrived']}")
            
        except Exception as e:
            print(f"Error evaluating {scen_name}: {str(e)}")
            continue

    # Save and plot results
    results_file = os.path.join(RESULTS_DIR, "comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    # Generate comparison plots
    metrics = ['avg_queue', 'avg_waiting_time', 'avg_speed', 'total_arrived']
    for metric in metrics:
        plot_comparison(comparison_results, metric)

    print(f"\nEvaluation complete. Results saved to {RESULTS_DIR}")