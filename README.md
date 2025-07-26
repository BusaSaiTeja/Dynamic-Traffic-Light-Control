# 🚦 Dynamic Traffic Light Management System

A smart traffic light control system that uses **Q-learning** to dynamically adjust signal timings based on real-time traffic conditions. This project uses **SUMO** (Simulation of Urban MObility) and **TraCI** (Traffic Control Interface) to simulate and control traffic flow.

## 🔧 Tech Stack

- Python  
- SUMO (Simulation of Urban MObility)  
- TraCI (Traffic Control Interface)  
- Q-learning (Reinforcement Learning)  
- Matplotlib (optional for visualization)  
- NumPy, Pandas (for data processing)

## 🎯 Objective

The aim of this project is to reduce traffic congestion and waiting times at intersections by:
- Monitoring the number of vehicles in real time  
- Using reinforcement learning (Q-learning) to choose the best traffic light phase  
- Adapting signal timings based on traffic density  

## 🚀 Features

- Real-time traffic simulation with SUMO  
- Adaptive signal timing optimized by Q-learning algorithm  
- Supports multi-lane and multi-direction intersections  
- Detailed logging and performance metrics  
- Visualization of learning progress and traffic flow (optional)  

## 🧠 How It Works

1. **SUMO Simulation**: Simulates vehicle flow in a custom road network  
2. **TraCI API**: Provides real-time interaction between the Python agent and SUMO  
3. **Q-Learning Agent**:  
   - Observes traffic states such as vehicle counts and waiting times  
   - Selects actions by adjusting traffic light phases to optimize flow  
   - Receives feedback via rewards to learn optimal strategies  
4. **Training and Evaluation**: Agent is trained over multiple episodes with performance tracked and analyzed  

## 🛠️ Installation & Setup

1. Install SUMO: https://sumo.dlr.de/docs/Installing.html  
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dynamic-traffic-light.git
   cd dynamic-traffic-light
   
Install dependencies:

pip install -r requirements.txt
Run the simulation:
python run_simulation.py

📊 Results & Evaluation

Significant reduction in average vehicle waiting times compared to fixed-time traffic signals
Improved throughput and smoother traffic flow demonstrated in simulation
Learning curve showing Q-learning agent performance improvement over episodes

📌 Future Improvements

Integrate Deep Reinforcement Learning techniques such as DQN for complex intersections
Incorporate real-world traffic data from sensors or cameras for live adaptation
Develop a web-based dashboard for real-time monitoring and control
Extend support for multi-intersection coordinated control

🤝 Contribution

Contributions, issues, and feature requests are welcome!
Feel free to fork the project and submit pull requests. Please adhere to the existing code style and include tests where applicable.

👨‍💻 Author

Sai Teja
GitHub: @BusaSaiTeja
LinkedIn: linkedin.com/in/busa-saiteja

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
