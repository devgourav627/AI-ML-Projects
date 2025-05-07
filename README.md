# AI-ML-Projects
# My Defense Tech Projects

Hey! I’m a B.Tech student trying to get into the Indian Army Internship Program (IAIP) 2025. I made this repo to show some projects I did for defense stuff, like catching hackers, fixing gear, and guiding drones. I’m new to this, but I worked hard to learn AI and coding for these. Hope you like them! They’re meant for IAIP’s tech goals, like cyber and drones.

## What I Built

### 1. Catching Network Hackers
This project tries to spot bad network stuff, like if someone’s hacking the Army’s systems.
- **What I Did**: I made fake network data (1000 bits of info, like how big packets are or how long they take). I used a thing called KNN to figure out what’s normal and what’s fishy. Something called PCA made it faster.
- **What Happened**: It got **[YourAccuracy, e.g., 85%]** right, so it caught most hackers! The picture below shows a “confusion matrix” (fancy name for what it got right or wrong). PCA kept about **[YourVarianceSum, e.g., 75%]** of the info (numbers: **[YourVariance, e.g., [0.45, 0.30]]]**).
- **Where It’s At**: `cyber-threat-detection/code/cyber_knn.ipynb`, `cyber-threat-detection/results/confusion_matrix.png`
- **Why It’s Cool for IAIP**: It could help keep Army networks safe from cyber attacks.

### 2. Predicting When Gear Breaks
This one guesses when military stuff (like trucks) might need fixing so they don’t break in a mission.
- **What I Did**: I made up 500 sensor readings (like temperature or how much it shakes) to predict how many hours are left before gear fails. Used Random Forest, which is like a smart guesser.
- **What Happened**: My guesses were off by about **[YourMSE, e.g., 1500]]** (that’s the “Mean Squared Error”). The picture shows my guesses vs. the real numbers—most are close to the red line!
- **Where It’s At**: `predictive-maintenance/code/maintenance_rf.ipynb`, `predictive-maintenance/results/rul_prediction.png`
- **Why It’s Cool for IAIP**: It could save the Army from gear breaking at the worst time.

### 3. Planning Drone Paths
This helps a drone fly around stuff, like avoiding danger zones.
- **What I Did**: I made a 10x10 grid with some blocked spots (like 20% of it). Used A* (a path trick) to find the shortest way from one corner (0,0) to the other (9,9).
- **What Happened**: Found a path in **[YourPathSteps, e.g., 12 steps]]**! The picture shows the path (red line), start (green dot), goal (blue star), and blocked spots (black).
- **Where It’s At**: `drone-path-optimization/code/drone_pathfinding.ipynb`, `drone-path-optimization/results/drone_path.png`
- **Why It’s Cool for IAIP**: Could help Army drones fly smart and safe.

## How to Check It Out
1. Grab my repo: `git clone https://github.com/devgourav627/AI-ML-Projects.git`
2. Get the tools: `pip install -r requirements.txt` (it’s got all the stuff you need).
3. Open the `.ipynb` files in each folder (like `cyber-threat-detection/code/cyber_knn.ipynb`) using Jupyter or Google Colab.
4. Hit run, and you’ll see my numbers and pictures!

## Stuff I Used
- **Coding**: Just Python (I’m still learning!).
- **Tools**: Things like `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn` (for the cyber project).
- **Tricks**: KNN, PCA, Random Forest, A* (sound fancy, but I’m figuring them out).

## What You Need
Check `requirements.txt` for the list:
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## What I Got Out of It
- Learned how to make data smaller with PCA without messing it up.
- Random Forest is cool for guessing numbers like gear life.
- A* is like a GPS for drones—super neat!
- Defense tech is hard but exciting, and I wanna learn more.

## Why I Want IAIP
I’m new to this, but these projects show I can try stuff like stopping hackers, fixing gear, or flying drones. That’s what IAIP’s tech team seems to care about, and I’d love to help the Army with it!

## Talk to Me
If you’ve got tips or wanna chat, email me at gouravzx7@gmail.com or find me on [GitHub](https://github.com/devgourav627). Thanks for checking my work!


