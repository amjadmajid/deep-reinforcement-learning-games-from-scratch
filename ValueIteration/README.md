<h1> Value Iteration</h1>
<p style="text-align:center; font-size:.9em">
<img width="500" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Markov_Decision_Process.svg/1920px-Markov_Decision_Process.svg.png"> 
<br />
Figure 1: Simple Markov Decision Process (MDP) model, with three states (green circles), two actions (orange circles), and two reward signals (orange arrows). Source <a href="https://en.wikipedia.org/wiki/Markov_decision_process">Wikipedia</a>.
</p>

Value Iteration is a dynamic programming algorithm used in reinforcement learning to find the optimal policy for a Markov Decision Process (MDP) (Figure 1). It is a method for computing the optimal state-value function, and subsequently deriving the optimal policy.

Here's a high-level overview of the algorithm:

1. **Initialization**: Initialize a table of value function estimates for each state (V(s)) arbitrarily (typically all zeros).

2. **Update**: For each state, perform a "full backup". That is, update the value of the state based on the sum of the expected rewards and expected future values of all possible next states. The value is computed as the maximum over all actions of the sum of immediate reward and discounted future value for each state-action pair.

   ```
   V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
   ```

   Where:
   - R(s,a) is the immediate reward for taking action a in state s,
   - γ (gamma) is the discount factor,
   - P(s'|s,a) is the transition probability of reaching state s' by taking action a in state s,
   - V(s') is the value of the next state s'.

3. **Convergence**: Repeat the update process for all states until the value function converges, that is, the maximum change in the value estimate is less than a small threshold (θ).

4. **Policy Extraction**: After convergence, extract the optimal policy by selecting the action that maximizes the sum of immediate reward and discounted future value for each state.

The following image visually illustrates this process:

![value_iteration](https://cdn-images-1.medium.com/max/800/1*MrRyZlA3ZzcmDQ-eSYNi8A.png)

In this image, each state (represented by a square in the grid) is updated based on the maximum value over all actions of the sum of immediate reward and discounted future value.

Please note that I'm an AI language model and I can't generate images. You can search online for "Value Iteration" if you need more visual aids.
