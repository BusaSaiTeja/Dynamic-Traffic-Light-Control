import os
import random
import subprocess
import json
import shutil

SUMO_HOME = os.environ.get("SUMO_HOME", "C:\\Program Files (x86)\\Eclipse\\Sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")

if not os.path.exists(SUMO_TOOLS):
    raise EnvironmentError("SUMO_HOME is not set correctly. Please check your SUMO installation.")

def create_x_road(output_dir):
    """Creates an X-junction (4-way intersection)"""
    nodes_content = """<nodes>
    <node id="A" x="-100" y="0" type="priority"/>
    <node id="B" x="0" y="0" type="traffic_light"/>
    <node id="C" x="100" y="0" type="priority"/>
    <node id="D" x="0" y="100" type="priority"/>
    <node id="E" x="0" y="-100" type="priority"/>
</nodes>"""

    edges_content = """<edges>
    <edge id="AB" from="A" to="B" numLanes="2" />
    <edge id="BA" from="B" to="A" numLanes="2" />
    
    <edge id="BC" from="B" to="C" numLanes="2" />
    <edge id="CB" from="C" to="B" numLanes="2" />
    
    <edge id="DB" from="D" to="B" numLanes="2" />
    <edge id="BD" from="B" to="D" numLanes="2" />
    
    <edge id="BE" from="B" to="E" numLanes="2" />
    <edge id="EB" from="E" to="B" numLanes="2" />
    </edges>"""

    return generate_network(output_dir, nodes_content, edges_content)

def create_t_road(output_dir):
    """Creates a T-junction (3-way intersection)"""
    nodes_content = """<nodes>
    <node id="A" x="-100" y="0" type="priority"/>
    <node id="B" x="0" y="0" type="traffic_light"/>
    <node id="C" x="100" y="0" type="priority"/>
    <node id="D" x="0" y="100" type="priority"/>
</nodes>"""

    edges_content = """<edges>
    <edge id="AB" from="A" to="B" numLanes="2" />
    <edge id="BA" from="B" to="A" numLanes="2" />
    
    <edge id="BC" from="B" to="C" numLanes="2" />
    <edge id="CB" from="C" to="B" numLanes="2" />
    
    <edge id="DB" from="D" to="B" numLanes="2" />
    <edge id="BD" from="B" to="D" numLanes="2" />
</edges>"""

    return generate_network(output_dir, nodes_content, edges_content)

def generate_network(output_dir, nodes_content, edges_content):
    """Generate SUMO network files and convert them to .net.xml"""
    os.makedirs(output_dir, exist_ok=True)
    
    nodes_file = os.path.join(output_dir, "network.nod.xml")
    edges_file = os.path.join(output_dir, "network.edg.xml")
    net_file = os.path.join(output_dir, "network.net.xml")

    with open(nodes_file, "w") as f:
        f.write(nodes_content)
    with open(edges_file, "w") as f:
        f.write(edges_content)

    try:
        subprocess.run([
            "netconvert",
            "--node-files", nodes_file,
            "--edge-files", edges_file,
            "--output-file", net_file
        ], check=True)
        return net_file
    except subprocess.CalledProcessError as e:
        print(f"Network generation failed: {e}")
        return None

def generate_traffic(net_file, output_dir):
    """Generate traffic routes with randomized traffic density"""
    routes_file = os.path.join(output_dir, "routes.rou.xml")
    
    try:
        cmd = [
            "python", os.path.join(SUMO_TOOLS, "randomTrips.py"),
            "-n", net_file,
            "-o", routes_file,
            "-e", "3600",
            "-p", str(random.uniform(1, 5)),
            "--validate",
            "--allow-fringe",
            "--fringe-factor", "10",
            "--vehicle-class", "passenger"
        ]
        
        subprocess.run(cmd, check=True)
        return routes_file
    except subprocess.CalledProcessError as e:
        print(f"Traffic generation failed: {e}")
        return None

def create_traffic_lights(output_dir):
    """Creates a traffic light logic where only one direction moves at a time."""
    tls_content = """<additional>
    <tlLogic id="B" type="static" programID="1" offset="0">
        <!-- Phase 1: North-South Green (Others Red) -->
        <phase duration="30" state="GGgrrrGGgrrr" />
        <!-- Phase 2: East-West Green (Others Red) -->
        <phase duration="30" state="rrrGGgrrrGGg" />
        <!-- Phase 3: Pedestrian/All Red -->
        <phase duration="5" state="rrrrrrrrrrrr" />
    </tlLogic>
</additional>"""

    tls_file = os.path.join(output_dir, "traffic_lights.tll.xml")
    with open(tls_file, "w") as f:
        f.write(tls_content)
    
    return tls_file

def generate_dataset(root_dir="traffic_dataset", num_scenarios=10):
    """Generate multiple SUMO scenarios with traffic lights"""
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir, ignore_errors=True)
    
    scenarios = []
    
    for i in range(num_scenarios):
        scenario_dir = os.path.join(root_dir, f"scenario_{i}")
        os.makedirs(scenario_dir, exist_ok=True)
        
        road_type = "x" if i % 2 == 0 else "t"
        net_file = create_x_road(scenario_dir) if road_type == "x" else create_t_road(scenario_dir)
        
        if not net_file:
            print(f"Skipping scenario {i} due to network generation failure.")
            continue
        
        route_file = generate_traffic(net_file, scenario_dir)
        if not route_file:
            print(f"Skipping scenario {i} due to traffic generation failure.")
            continue
        
        tls_file = create_traffic_lights(scenario_dir)

        scenarios.append({
            "id": f"scenario_{i}",
            "road_type": road_type,
            "network_file": net_file,
            "route_file": route_file,
            "tls_file": tls_file
        })
    
    with open(os.path.join(root_dir, "manifest.json"), "w") as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"Generated {len(scenarios)} scenarios successfully.")

if __name__ == "__main__":
    generate_dataset(num_scenarios=10)
