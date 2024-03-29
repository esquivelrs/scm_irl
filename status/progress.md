---
marp: true
author: Rolando Esquivel
paginate: true
math: mathjax

---
<style>
    h1 {
        font-size: 100px;
    }
    img[alt~="center"] {
      display: block;
      margin: 0 auto;
    }
</style>

## Reinforcement Learning and Inverse Optimization for Autonomous Navigation

## Special course - Fall 2023
Rolando Esquivel-Sancho

---
## Week 9 - 27-02-2024:

* Reviewing literature mainly in inverse RL, RL with human feedback and COLREG-compliant collision avoidance.

* Exploring the Data:
    * Question? Land.pickle returning empty list.
    * Question? Depth.pickle vrs depth data in seachart.json.

* Structuring the Python Package:
    * Defining modules and classes for data processing, analysis, and environment implementation.

---
## Week 10 - 04-03-2024:

Imitation library: https://imitation.readthedocs.io/en/latest/#
![center width:200px](image.png)
Algorithms:
* Behavioral Cloning (Policy *)
* Generative Adversarial Imitation Learning (Policy *)
* Adversarial Inverse Reinforcement Learning (Policy, recovers reward func *)
* DAgger (Policy similar to BC but online *)
* Density-Based Reward Modeling (Reward function no interpretable *)
* Maximum Causal Entropy Inverse Reinforcement Learning (Reward *)
* Preference Comparison (Reward)
* Soft Q Imitation Learning (Policy DQN)

---
Topics & next steps
* Create the env with Gymnasium's API
* Actions:
    * Used sog and cog as action vector and keep lon and lat as observations.
    * create the dynamic model.
* Observations:
    * Model as a multiagent systems.
    * Fix number of dynamic objects.
    * observation horizon.
* Limitations on Horizon Length
* Trajectories to Transitions.
* Metrics


---
## Week 11 - 12-03-2024:

#### Status:
* action space: `Box(-1, 1, (2,), float32)` -- normalization (max-speed)

* observation space: `Box(0, 255, (W, H), np.uint8)` and `Box(-1, 1, (n_vessels,4), np.uint8)`

---
* Using the agent accions COG (radians) and SOG (m/s) to generate location.

![alt text](image-1.png) ![alt text](image-2.png)


---
#### Next steps
* Create Land/Depth observation relative to the agent
* Create trajectory metric.

