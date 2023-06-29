Q-learning is a model-free reinforcement learning algorithm. The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances. It does not require a model (hence the connotation "model-free") of the environment, and it can handle problems with stochastic transitions and rewards, without requiring adaptations.

For any finite Markov decision process (FDP), Q-learning finds a policy that is optimal in the sense of maximizing the expected value of the total reward over any and all successive steps, starting from the current state. 'Q' names the function that the algorithm computes with the maximum expected rewards for an action taken in a given state.

![Q-learning diagram](https://miro.medium.com/max/1400/1*QeoQEqWYYPs1P8yUwyaJVQ.png)

(Source: Medium)

Here's how Q-Learning works:

1. **Initialize Q-values:** Initialize the Q-values table, Q(s, a). The Q-table guides the agent to the best action at each state. Q-table is a simple look-up table where we have a row for every state (s) and a column for every action (a). Initially, Q-table values are set to zero.

2. **Choose an action:** In each episode, choose an action (a) for the current state (s) based on the Q-value table. The action can be selected either randomly (exploration) or by choosing the action with the highest Q-value (exploitation). The choice between exploration and exploitation is managed by the ε-greedy policy.

3. **Perform the action:** Perform the action (a) and move to the new state (s'). Get the reward (r).

4. **Update the Q-value:** Update the Q-value of the performed action using the Q-learning update rule. The rule uses a learning rate 'α' to update the Q-value, and a discount factor 'γ' to consider future rewards.

   The update rule is as follows:

   Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]

5. **Repeat the process:** If the new state (s') is the terminal state, end the episode and start a new one. Otherwise, move to the new state (s') and repeat the process from step 2.

The algorithm continues until it reaches a convergence. When the algorithm converges, we have an optimal policy, π*, and the corresponding optimal Q-value function, Q*.

In a nutshell, Q-learning learns by continuously updating Q-values for each state-action pair and using these Q-values to make optimal action choices. It uses the idea of expected future rewards and exploration-exploitation trade-off to balance the need to try new actions and continue with actions that have known rewards. 

For more advanced and complex environments with large or continuous state-action spaces, we may not be able to use the simple table-based Q-Learning approach and may need to use function approximators, like neural networks, to approximate the Q-values. These types of approaches are called Deep Q-Learning and are commonly used in practice.