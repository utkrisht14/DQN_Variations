## DQN_Variations
### Double DQN (Double Deep Q-Network)

Double DQN is an extension of the standard DQN algorithm designed to reduce overestimation bias when estimating Q-values in Reinforcement Learning. In standard DQN, the same network is used for both selecting and evaluating actions, which can lead to overly optimistic value estimates. Double DQN mitigates this by using two separate networks: one (the "online" network) for selecting the best action and another (the "target" network) for evaluating the value of that action.

The update rule for Double DQN is modified to decouple action selection and evaluation. Instead of using the max action's value from the target network, Double DQN selects the action with the online network and evaluates it using the target network. This separation reduces bias and results in more stable learning. The mathematical update for Double DQN is:

```math
y_t^{\text{Double DQN}} = r_t + \gamma \cdot Q_{\theta'}(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1}, a'))
```

where:

- \( y_t^{\text{Double DQN}} \): Target Q-value for time step \( t \) in Double DQN.
- \( r_t \): Reward received after taking action at time step \( t \).
- \( \gamma \): Discount factor that determines the importance of future rewards (usually between 0 and 1).
- \( s_{t+1} \): The next state at time step \( t + 1 \).
- \( Q_{\theta'} \): Q-value function estimated by the target network with parameters \( \theta' \).
- \( Q_{\theta} \): Q-value function estimated by the online network with parameters \( \theta \).
- \( \arg\max_{a'} Q_{\theta}(s_{t+1}, a') \): Action that maximizes the Q-value in the online network for the next state \( s_{t+1} \).


