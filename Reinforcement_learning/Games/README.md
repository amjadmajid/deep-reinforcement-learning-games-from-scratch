# Grid World

This code defines a simple 2D grid world for a reinforcement learning environment with specified actions and states.

The code can be explained as follows:

1. The `Action` class represents the action space of an agent in the grid world. The agent can move up, down, left or right. The `get_action_by_idx` method returns an action given its index, while `get_action_by_key` returns an action given its name.

2. The `GridWorld` class represents the grid world environment. The grid world has a certain shape and can contain obstacles and a terminal state (goal). The agent's position is represented by `agent_pos`.

3. In `GridWorld`, the `step` method simulates the agent's movement in the grid world when it takes an action. The agent receives a reward of 0 when it reaches the terminal state and -1 otherwise. The method also checks if the new state after the agent's action is valid (not an obstacle or an edge).

4. The `simulated_step` method is similar to the `step` method, but does not change the actual position of the agent. It is useful for algorithms like value iteration where we simulate the outcome of an action without actually performing it.

5. The `reset` method reinitializes the environment to its original state.

6. The `render` method prints the current state of the grid world.

7. The utils.py provides a function for poltting the state value function.

