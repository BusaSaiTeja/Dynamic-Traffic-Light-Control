import torch
from traffic_env import SumoTrafficEnv
from dqn_agent import DQN
import time

# Use raw string for Windows paths
SUMO_CFG = r"traffic_dataset\\scenario_0\\sumo_config.sumocfg"

def run_simulation():
    env = SumoTrafficEnv(SUMO_CFG)
    try:
        model = DQN(state_dim=4, action_dim=4)
        model.load_state_dict(torch.load(r"saved_models\\dqn_traffic_model.pth"))
        model.eval()

        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():  # Disable gradient computation
                action = torch.argmax(model(state_tensor)).item()
            
            state, _, done, _ = env.step(action)
            time.sleep(0.1)  # Control simulation speed
            
    finally:
        env.close()

if __name__ == "__main__":
    run_simulation()