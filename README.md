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

<hr/>

### Dueling DQN

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

<hr/>

### Noisy DQN 

Noisy DQN is an enhancement of the DQN architecture that introduces stochastic exploration by adding noise directly into the neural network’s weights. This helps the agent explore more effectively without relying on traditional exploration strategies like ϵ-greedy. The amount of noise is learnable and can be adjusted during training, allowing the agent to balance exploration and exploitation more effectively.

The Q-value in Noisy DQN is computed as:

```math
Q(s, a; \theta, \epsilon) = Q(s, a; \theta + \sigma \odot \epsilon)
```

<div> <ul> <li><strong>Q(s, a; θ, ϵ)</strong>: The Q-value estimated by the network for state <strong>s</strong> and action <strong>a</strong>, with parameters <strong>θ</strong> and noise <strong>ϵ</strong>.</li> <li><strong>θ</strong>: The learnable parameters (weights) of the neural network.</li> <li><strong>σ</strong>: The standard deviation of the noise applied to the network's weights.</li> <li><strong>ϵ</strong>: The noise term, sampled from a Gaussian distribution (typically zero-mean with some variance).</li> <li><strong>⊙</strong>: Element-wise multiplication, applied between <strong>σ</strong> and <strong>ϵ</strong> to perturb the weights.</li> </ul> </div>

<b> Note: </b>

**Noisy Layers**: These replace the standard layers and introduce learnable noise into the weights, encouraging the agent to explore different actions based on the noise added during training.

<hr/>

### Prioritized Experience Replay (PER)

Prioritized Experience Replay is an extension of the standard Experience Replay technique used in DQN, which improves learning efficiency by replaying more important experiences more frequently. In standard experience replay, transitions (state, action, reward, next state) are uniformly sampled from the replay buffer. However, in PER, transitions are sampled based on their **priority**, which is usually determined by the **temporal difference (TD) error**. Experiences with higher TD errors are considered more important because they represent unexpected or poorly understood transitions.



 **Priority Sampling**: Instead of sampling experiences uniformly, transitions with larger TD errors (i.e., those where the agent was surprised by the outcome) are sampled more often. The probability of sampling a transition <i>i</i> is proportional to its priority p<sub>i</i> :
   
   ```math
   P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
   ```

   <div>
  <ul>
    <li><strong>P(i)</strong>: The probability of sampling transition <strong>i</strong>, proportional to its priority.</li>
    <li><strong>p<sub>i</sub></strong>: The priority of transition <strong>i</strong>, typically based on the TD error |δ<sub>i</sub>|.</li>
    <li><strong>α</strong>: A hyperparameter controlling the level of prioritization (α = 0 corresponds to uniform sampling).</li>
    <li><strong>w<sub>i</sub></strong>: The importance sampling weight for transition <strong>i</strong> to correct for bias.</li>
    <li><strong>N</strong>: The total number of transitions stored in the replay buffer.</li>
    <li><strong>β</strong>: A hyperparameter controlling the strength of the importance sampling correction. Typically annealed towards 1.</li>
    <li><strong>TD Error |δ<sub>i</sub>|</strong>: The temporal difference error for transition <strong>i</strong>, indicating how surprising or incorrect the Q-value estimate was for that transition.</li>
  </ul>
</div>


   #### Notes:

1. **Prioritized Sampling**: PER samples transitions with higher TD errors more frequently, allowing the agent to focus on learning from more important experiences.
2. **Importance Sampling (IS)**: To correct the bias introduced by prioritized sampling, IS weights are applied when updating the network, ensuring that the updates remain unbiased.
3. **Efficiency Considerations**: Using data structures like a **Sum Tree** ensures efficient computation and updating of priorities, which is important when scaling up the buffer size.

Below given table provides some information about these DQN variations: <br/>
| Algorithm                    | Problem Addressed                                     | Key Idea                                                                                                                                                                                                 | Benefits                                                                                          |
|------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Double DQN**                | Overestimation of action values in standard DQN       | Uses two networks: the online network to select the best action and the target network to evaluate the value of that action. This decouples action selection from evaluation, reducing the overestimation bias caused by the max operator in the Q-learning update. | - Reduces overestimation bias<br>- Improves learning stability<br>- Leads to better policy evaluation |
| **Dueling DQN**               | Inefficient learning of state values and action advantages | Splits the network into two streams after some shared layers: one estimates the state value function **V(s)** and the other estimates the advantage function **A(s, a)**. The two streams are then combined to produce the Q-values. This architecture allows the network to separately learn the value of a state and the advantage of each action in that state. | - Learns state values and advantages separately<br>- Improves learning efficiency<br>- Performs better in states where actions have similar effects |
| **Prioritized Experience Replay** | Uniform sampling from replay buffer is suboptimal  | Assigns a priority to each experience in the replay buffer based on the magnitude of its temporal-difference (TD) error. Experiences with higher TD errors are sampled more frequently. Importance Sampling weights are used to correct the bias introduced by prioritization. | - Focuses learning on more significant experiences<br>- Improves sample efficiency<br>- Accelerates convergence |
| **Noisy DQN**                 | Inefficient exploration with ε-greedy policies       | Introduces parameterized noise into the network weights, replacing deterministic weights with distributions. During training, the network learns both the mean and variance of the weights, allowing it to adjust the level of noise (exploration) in a state-dependent manner. This encourages more effective exploration without the need for an explicit exploration schedule like ε-greedy. | - Enables state-dependent exploration<br>- Improves exploration efficiency<br>- Often outperforms ε-greedy exploration strategies |



