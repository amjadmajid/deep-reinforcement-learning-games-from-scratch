# Policy Gradient Methods
Policy gradient methods are a class of reinforcement learning algorithms that aim to optimize the policy of an agent directly, rather than estimating the value function. These methods are especially useful in scenarios where the action space is continuous or the policy space is complex.

To understand policy gradient methods, let's break down the key components and steps involved:

1. Markov Decision Process (MDP): Policy gradient methods operate within the framework of an MDP, which consists of a tuple $(S, A, P, R, \gamma)$. Here, $S$ represents the state space, $A$ represents the action space, $P(s'|s, a)$ is the transition probability function defining the probability of transitioning to state $s'$ given state $s$ and action $a$, $R(s, a)$ is the immediate reward obtained from taking action $a$ in state $s$, and $\gamma$ is the discount factor determining the importance of future rewards.

2. Policy Function: A policy is a mapping from states to probability distributions over actions. It defines the agent's behavior in the environment. In policy gradient methods, the policy is often represented by a parametric function, such as a neural network, with parameters $\theta$.

3. Objective Function: The objective of policy gradient methods is to find the optimal policy that maximizes the expected cumulative reward over time. This objective is typically formulated as the expected return or total discounted reward:

   $$J(\theta) = \mathbb{E}[R(\tau)] = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]$$

   Here, $\tau$ represents a trajectory or an episode, which is a sequence of states and actions encountered by the agent.

4. Policy Gradient Theorem: The policy gradient theorem provides a way to compute the gradient of the expected return with respect to the policy parameters. It states that the gradient of the objective function with respect to the policy parameters is given by:

   $$\nabla J(\theta) \approx \mathbb{E}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi(a_t|s_t) Q(s_t, a_t)\right]$$

   Here, $\pi(a_t|s_t)$ is the probability of taking action $a_t$ in state $s_t$ according to the policy, and $Q(s_t, a_t)$ is the action-value function representing the expected return starting from state $s_t$, taking action $a_t$, and following the policy thereafter.

5. Estimating the Gradient: The policy gradient can be estimated using either the Monte Carlo method or the actor-critic method.

   - Monte Carlo: In the Monte Carlo method, multiple trajectories are sampled by executing the policy in the environment. The gradient is estimated by averaging the product of the log probability of the actions and the corresponding cumulative rewards for each trajectory.

   - Actor-Critic: The actor-critic method combines policy gradient with value function estimation. It maintains two components: an actor (policy) and a critic (value function). The critic estimates the state-value function $ V(s)$, which is then used to compute the advantage function $A(s, a) = Q(s, a) - V(s)$. The advantage function measures how much better or worse an action is compared to the average action in a given state. The policy gradient is then computed using the advantage function instead of the action-value function.

6. Optimization: Once the gradient is estimated, it can be used to update the policy parameters using optimization techniques like stochastic gradient ascent. The policy parameters are iteratively updated to increase the objective function, thus improving the policy.

7. Exploration vs. Exploitation: Policy gradient methods can incorporate exploration strategies to encourage the agent to explore the environment and discover new, potentially better policies. This can be achieved through techniques like adding entropy regularization to the objective function, which encourages the policy to be more stochastic.

8. Training Process: The training process of policy gradient methods typically involves iteratively interacting with the environment, collecting trajectories, estimating gradients, and updating the policy parameters. The process continues until the policy converges to an optimal or near-optimal policy.

Overall, policy gradient methods provide a powerful framework for training agents in reinforcement learning tasks. By directly optimizing the policy, these methods can handle continuous action spaces and effectively learn complex behaviors.

--- 
# Policy Gradient: The Monte Carlo method

In the Monte Carlo method, the goal is to estimate the gradient of the objective function $J(\theta)$ with respect to the policy parameters $\theta$. This gradient is approximated by sampling multiple trajectories and computing their contributions to the gradient.

1. Generating Trajectories: To start, we execute the current policy in the environment to generate a set of trajectories. Each trajectory consists of a sequence of states and actions encountered by the agent. Let's consider a trajectory $\tau$ that consists of $t$ time steps, represented as $\tau = \{(s_0, a_0), (s_1, a_1), \ldots, (s_T, a_T)\}$, where $s_t$ is the state at time step $t$ and $a_t$ is the action taken at time step $t$.

2. Computing Cumulative Rewards: For each trajectory, we calculate the cumulative reward obtained from following the policy. The cumulative reward at time step $t$ is defined as the sum of the immediate rewards from time step $t$ to the end of the trajectory, and is denoted as $$R(\tau) = \sum_{t'=t}^{T} \gamma^{t'-t} R(s_{t'}, a_{t'})$$, where $\gamma$ is the discount factor and $R(s_{t'}, a_{t'})$ is the immediate reward obtained at time step $t'$.

3. Estimating the Policy Gradient: To estimate the policy gradient, we need to compute the gradient of the logarithm of the policy's action probabilities and multiply it by the cumulative rewards. The gradient of the objective function $ J(\theta) $ can be approximated as:

   $$\nabla J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_{\theta} \log \pi(a_{t} | s_{t}) R(\tau_i)$$

   Here, $N$ represents the number of trajectories sampled, $\pi(a_{t} | s_{t})$ denotes the probability of taking action $a_t$ in state $s_t$ according to the policy, and $R(\tau_i)$ is the cumulative reward obtained from trajectory $\tau_i$.

4. Update Policy Parameters: With the estimated policy gradient, we can update the policy parameters using an optimization algorithm, such as stochastic gradient ascent. The update rule typically follows:

   $$\theta \leftarrow \theta + \alpha \nabla J(\theta)$$

   Here, $\alpha$ is the learning rate that determines the step size during parameter updates.

By iteratively sampling trajectories, estimating the policy gradient, and updating the policy parameters, the Monte Carlo method allows the agent to improve its policy towards maximizing the expected return.

It's worth noting that in practice, variance reduction techniques, such as baseline subtraction or advantage estimation, are often employed to reduce the variance of the gradient estimates and improve learning stability. These techniques involve subtracting a learned baseline or estimating the advantage function $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$, where $Q(s_t, a_t)$ is the action-value function and $V(s_t)$ is the state-value function.