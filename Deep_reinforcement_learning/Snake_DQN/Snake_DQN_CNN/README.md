
# DEEP Q-NETWORK (DQN) FOR SNAKE GAME
1. You can select between DQN and DDQN by setting the variable `MODEL` to either 
`'DQN'` or `'DDQN'` in the file `train.py`.
2. You can find the original DQN paper [here](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

##  Double DQN (DDQN)

In the original DQN to produce the estimated Q-value of the next state we use a 
target network to (i) predict the Q-values associated with all  actions and the 
next state, and then, (ii) select the maximum Q-value from the predicted values. 
<br>

In DDQN, to estimate the Q-value of the next state, we (i) use the primary and
the target networks to produce the the estimated Q-values assoiciated with all 
the actions and the next state twice. Then, (ii) we identify the index of the 
maximum Q-value of predicted by the primary network. Finally, (iii) we select 
the estimated Q-value from the output of the target network using the 
index identified in step (ii). 

# TODO
---
 - [ ] add GPU support.
 - [ ] limit the maximum length of an episode.
