## DQN_Variations
### Double DQN (Double Deep Q-Network)

Double DQN is an extension of the standard DQN algorithm designed to reduce overestimation bias when estimating Q-values in Reinforcement Learning. In standard DQN, the same network is used for both selecting and evaluating actions, which can lead to overly optimistic value estimates. Double DQN mitigates this by using two separate networks: one (the "online" network) for selecting the best action and another (the "target" network) for evaluating the value of that action.

The update rule for Double DQN is modified to decouple action selection and evaluation. Instead of using the max action's value from the target network, Double DQN selects the action with the online network and evaluates it using the target network. This separation reduces bias and results in more stable learning. The mathematical update for Double DQN is:

```math
y_t^{\text{Double DQN}} = r_t + \gamma \cdot Q_{\theta'}(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1}, a'))
```

where:

- **y<sub>t</sub><sup>Double DQN</sup>**: Target Q-value for time step **t** in Double DQN.
- **r<sub>t</sub>**: Reward received after taking action at time step **t**.
- **γ**: Discount factor that determines the importance of future rewards (usually between 0 and 1).
- **s<sub>t+1</sub>**: The next state at time step **t + 1**.
- **Q<sub>θ'</sub>**: Q-value function estimated by the target network with parameters **θ'**.
- **Q<sub>θ</sub>**: Q-value function estimated by the online network with parameters **θ**.
- **arg max<sub>a'</sub> Q<sub>θ</sub>(s<sub>t+1</sub>, a')**: Action that maximizes the Q-value in the online network for the next state **s<sub>t+1</sub>**.


## Duleing DQN

Dueling DQN is an extension of the DQN architecture designed to enhance learning efficiency by separating the estimation of the state-value and the action-value. In standard DQN, the Q-value is estimated directly for each action, which may not be efficient for states where choosing an action does not significantly affect the outcome. Dueling DQN introduces two streams within the network: one for estimating the **state-value** (how good it is to be in a given state) and another for estimating the **advantage** of each action (how much better it is to take a specific action over others in the same state).

The Q-value is then computed by combining the two as follows:

```math
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)
```


where:

1. **Q(s, a)**: The estimated Q-value for state **s** and action **a**.
2. **V(s)**: The value function representing the value of being in state **s**, independent of any action.
3. **A(s, a)**: The advantage function which represents how much better or worse action **a** is compared to the average action in state **s**.
4. **|𝒜|**: The number of possible actions in the action space.

The equation and symbols are written in LaTeX using the `$$` format for equations and should display correctly on GitHub as long as the platform supports LaTeX-style syntax. If GitHub doesn't render it, the equation might not appear correctly, but it is formatted as per the standard LaTeX syntax.

Let me know if you need further modifications!

