# Aysnchronous Advantage Actor-Critic (A2C) Algorithm

**Overview of A2C:**

The A2C algorithm combines the actor-critic framework with the advantage function approach to improve reinforcement learning. It consists of two main components: the actor, which learns a policy, and the critic, which estimates the value function.

The actor determines the action to take in a given state, while the critic evaluates the value of a state or state-action pair. By incorporating an advantage function, A2C aims to reduce variance and improve learning efficiency.

**Mathematical Equations:**

1. **Policy:** The policy $\pi(a|s;\theta)$ is a parameterized probability distribution that defines the agent's action selection behavior given a state $s$. It is parameterized by $\theta$, which represents the weights of the policy network.

2. **Value Function:** The value function $V(s;\phi)$ estimates the expected return (or cumulative reward) starting from state $s$ and following the policy $\pi$ parameterized by $\theta$. It is parameterized by $\phi$, which represents the weights of the value network.

3. **Advantage Function:** The advantage function $A(s,a;\theta,\phi)$ quantifies the advantage of taking action $a$ in state $s$ over the average expected return when following the policy $\pi$ parameterized by $\theta$. It is calculated as the difference between the estimated value and the state-action pair's value:

$$A(s,a;\theta,\phi) = Q(s,a;\theta,\phi) - V(s;\phi)$$

where $Q(s,a;\theta,\phi)$ is the estimated state-action value.

**A2C Algorithm Steps:**

1. Initialize the actor network with random weights $\theta$ and the critic network with random weights $\phi$.
2. Repeat until convergence or a maximum number of steps:
   - Observe the current state $s$ from the environment.
   - Compute the action probabilities given the current state: $\pi(a|s;\theta)$.
   - Sample an action $a$ from the action probabilities.
   - Execute the action in the environment and receive the next state $s'$, reward $r$, and done flag indicating the end of the episode.
   - Calculate the advantage $A(s,a;\theta,\phi)$ using the current state-action pair.
   - Update the value network by minimizing the mean squared error loss:

   $$L_{\text{critic}} = \frac{1}{2}(V(s;\phi) - (r + \gamma V(s';\phi)))^2$$

   where $\gamma$ is the discount factor.

   - Update the actor network by maximizing the expected reward using the advantage estimate:

   $$L_{\text{actor}} = -\log(\pi(a|s;\theta)) \cdot A(s,a;\theta,\phi)$$

   - Compute the gradients and perform the weight updates using stochastic gradient descent or another optimization algorithm.
   - Repeat the above steps for a predefined number of episodes or until convergence.

**Relation to the Code:**

The provided code implements the A2C training module as follows:

1. The code imports the necessary libraries, modules, and custom classes.
2. Several constants are defined, including learning rate (`LR`), discount factor (`GAMMA`), number of episodes (`EPISODES`), rendering options (`RENDER` and `RENDER_EVERY`), and hidden layer size (`HIDDEN_SIZE`).
3. A named tuple `Transitions` is defined to store state-action-reward transitions.
4. The device is set to CUDA if available, or CPU as a fallback.
5. The GridWorld environment is initialized with the desired parameters, such as shape, initial agent position, terminal position, and obstacles.
6. The `compute_returns` function calculates the discounted returns for each time step in an episode.
7. The `play_episode` function interacts with the environment, collects log probabilities, values, rewards, and entropy for each step in an episode.
8. The `optimize_model` function updates the actor and critic networks using the collected data and performs gradient descent.
9. The `trainIters` function performs the training loop, executing multiple episodes and calling `play_episode` and `optimize_model`.
10. Inside the `if __name__ == '__main__':` block, the actor and critic networks are created, along with the optimizers.
11. The `trainIters` function is called to train the networks over a specified number of iterations.

The code aligns with the A2C algorithm's main steps, including interacting with the environment, estimating values and action probabilities, calculating advantages, and updating the actor and critic networks through optimization.