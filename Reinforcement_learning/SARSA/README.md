SARSA (State-Action-Reward-State-Action) is a fundamental method for learning a Markov Decision Process policy, used in reinforcement learning. The name SARSA actually stems from the fact that the main function for updating the Q-value depends on the current "State", "Action" taken, the "Reward" received, the new "State" arrived at, and the new "Action" taken at that new state. 

The update function for SARSA is as follows:

Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]

Where:
- Q(s, a) is the current estimate of the Q-value.
- α is the learning rate.
- r is the reward for the current step.
- γ is the discount factor.
- Q(s', a') is the estimated Q-value for the next step.

Unlike Q-learning, which is an off-policy learner, SARSA is an on-policy learner. This means that it learns the value of the policy being carried out by the agent, including the exploration steps.

Now, let's illustrate the SARSA algorithm with a simple gridworld example. In a gridworld, an agent can move in four directions (Up, Down, Left, Right), with the aim to reach a goal state from a starting state.

![Grid World](https://www.researchgate.net/profile/Alexey-Melnikov-3/publication/262526038/figure/fig2/AS:296823253159943@1447779586992/The-grid-world-task-The-goal-of-the-game-is-to-find-the-star-At-the-beginning-of-each.png)

(Source: Medium)

Here is a step-by-step explanation of the SARSA algorithm with the gridworld:

1. Initialize Q-table with zeros. The Q-table will have dimensions of (Number of States x Number of Actions), so in this case, it would be (25 states x 4 actions) for a 5x5 gridworld.

2. For each episode:
   - Choose an action (a) in the current world state (s) based on a epsilon-greedy policy.
   - Take this action and observe the reward (r) and the new state (s').
   - Choose the new action (a') in the new state (s') based on the same epsilon-greedy policy.
   - Update the Q(s, a) as follows:
     - Q(s, a) = Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
   - Set the new state (s') as the current state (s) and new action (a') as the current action (a) and repeat the process.
   - If goal state is reached, end the episode and start a new one.

The agent will repeat this process for a large number of episodes to learn the optimal policy. The learned policy will enable the agent to choose actions that maximize the cumulative reward over time, i.e., reach the goal in the shortest possible path.

In the SARSA algorithm, we continuously update Q-values until the Q-values converge (stop changing by a substantial amount). This means the agent has "learned" the optimal policy, knowing what action to take in each state.

The SARSA algorithm also introduces the concept of exploration and exploitation using an epsilon-greedy policy. Initially, epsilon is high so that the agent explores the environment (randomly selecting actions), but with time epsilon decays, and the agent starts exploiting its learned knowledge about the environment.