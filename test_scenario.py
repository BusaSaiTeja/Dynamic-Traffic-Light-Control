import traci
import os

def test_scenario(scenario_dir):
    sumo_cmd = [
        "sumo-gui",
        "-n", os.path.join(scenario_dir, "network.net.xml"),
        "-r", os.path.join(scenario_dir, "routes.rou.xml")
    ]
    traci.start(sumo_cmd)
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    
    traci.close()

# Test first 3 scenarios
test_scenario(f"traffic_dataset/scenario_{0}")