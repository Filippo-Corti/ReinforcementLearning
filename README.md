# RL Car Racing
















## Project Proposal (Project ID: PG-4):

### Main Focus
Policy gradient, deep neural policies

### Scientific Objective 
Understand the impact of the complexity of the policy space on the performance of the agent and the amount of interactions needed to converge.

### Problem Description 
You want to control an autonomous Formula 1 car so that it completes a circuit in the shortest time possible. 
The car has access to its position (relative to the circuit) and velocity. 
It controls the acceleration and the steering wheel. It should be heavily penalized for going off track.

### Tasks
1. Choose/design a circuit (e.g. The Circuit de Monaco) and model the problem as an MDP with continuous actions. Think carefully about how to model the circuit and the relative position of the car. You are allowed to include in the state additional information about the specific circuit you are trying to solve, such as landmarks or information about the curvature, if you think this is helpful.
2. Define a reward function that encourages the agent to complete the circuit in the shortest time possible without going off track.
3. Define a parametric policy (Gaussian or deterministic) that maps states to actions (or mean actions) using a deep neural network (e.g. a fully connected neural network a.k.a. multi-layer perceptron). Implement it so that you can easily try neural networks of different sizes (number of layers and width of the layers).
4. Train your agent using a deep RL algorithm of your choice and compare the results obtained with policy networks of different sizes, in terms of:
    - Final performance
    - Number of training episodes needed to converge
    - Time needed to converge (this is machine dependent so make sure to run all experiments on the same computer)

### Challenging Variants 
Try to learn a policy that can solve multiple circuits, in particular circuits not seen during training. You may want to train it on multiple, diverse circuits.
