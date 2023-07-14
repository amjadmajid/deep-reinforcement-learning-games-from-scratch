
# DEEP Q-NETWORK (DQN) FOR SNAKE GAME
1. You can select between DQN and DDQN by setting the variable `MODEL` to either 
`'DQN'` or `'DDQN'` in the file `train.py`.
2. You can find the original DQN paper [here](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

##  Double DQN (DDQN)

The key differences between DQN (Deep Q-Learning) and DDQN (Double Deep Q-Learning) can be summarized as follows:

In DQN, the estimated Q-value of the next state is found by (i) predicting the Q-values associated with all possible actions for the next state using the target network and then (ii) selecting the maximum Q-value from those predicted values.

The DDQN improves on the DQN by reducing overestimation of Q-values. In DDQN, the estimation of the Q-value of the next state is obtained by (i) using both the primary (online) network and the target network to predict Q-values for all possible actions for the next state. Then, (ii) the action that maximizes the Q-value is determined by the primary network, and (iii) the Q-value corresponding to this action is chosen from the Q-values predicted by the target network.

The purpose of this process in DDQN is to decouple the selection of the action from the evaluation of that action, which helps in reducing overoptimistic value estimates that may occur in DQN.

# TODO
---
 - [ ] add GPU support.
 - [ ] limit the maximum length of an episode.
