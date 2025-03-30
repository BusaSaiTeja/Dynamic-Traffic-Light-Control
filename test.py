import torch
from traffic_env import SumoTrafficEnv
from dqn_agent import DQN

SUMO_CFG = "your_sumo_config.sumocfg"

env = SumoTrafficEnv(SUMO_CFG)
model = DQN(state_dim=4, action_dim=4)
model.load_state_dict(torch.load("dqn_traffic_model.pth"))
model.eval()

state = env.reset()
done = False
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()
    state, _, done, _ = env.step(action)

env.close()
