import gym
import random
import torch
import torch.nn as nn
import os
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# Set the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define some hyperparameters
GAMMA = 0.995
BATCH_SIZE = 64
EPSILON_MIN = 0.01

# Define the Q_Network.
# This Q network would be same as standard DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience Replay Buffer with the Prioritized Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer =deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6 # Prioritization exponent
        self.epsilon = 1e-2 # Small amount to avoid zero priority

    def add(self, experience, td_error=None):
        # Adds experiences to the buffer, with priority based on TD error.
        self.buffer.append(experience)
        if td_error is None:
            priority = max(self.priorities, default=1.0)
        else:
            priority =(abs(td_error) + self.epsilon) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        """" Sample a mini-batch from the buffer based on the priorities """
        priorities = np.array(self.priorities, dtype=np.float32)
        # Normalize priorities
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        # Calculate importance-sampling weights to correct bias from prioritized sampling
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /=  weights.max()
        return experiences, indices, torch.tensor(weights, dtype=torch.float32).to(device)

    def update_priorities(self, indices, td_errors):
        """ Update the priorities of sampled experiences based on the new TD errors """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# Define the Agent class
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.memory = ReplayMemory(10000)

        # Initialize the networks
        self.online_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.update_target_network()

        # Define the optimizer and the loss function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss(reduction="none")  # Loss function with no reduction (required for PER)

    def update_target_network(self):
        """ Update the target network periodically """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state):
        """ Select action to perform in the environment """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.online_net(state)
            return q_values.argmax().item() # Choose action with the highest q-value

    def store_experience(self, state, action, reward, next_state, done):
        # Estimate TD error for prioritization
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_value = self.online_net(state_tensor)[0, action].item()
            next_q_value = self.target_net(next_state_tensor).max(1)[0].item()
            td_error = reward + GAMMA * next_q_value * (1 - done) - q_value
        self.memory.add((state, action, reward, next_state, done), td_error)

    def train_step(self):
        """ Trains the online network by sampling a batch from replay memory """
        if len(self.memory) < BATCH_SIZE:
            return

        experiences, indices, weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert them into the tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Current q values
        actions = actions.unsqueeze(1)
        q_values = self.online_net(states).gather(1, actions).squeeze(1)

        # Double DQN updates
        next_actions = self.online_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute TD errors and losses, then apply importance sampling weights
        td_errors = target_q_values.detach() - q_values
        losses = self.loss_fn(q_values, target_q_values.detach()) * weights
        loss = losses.mean()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Backpropagation and memory step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """ Decays the epsilon to reduce exploration over time """
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= 0.995


# Training function
def train_agent():
    env = gym.make("LunarLander-v2")
    env.seed(42)

    # Define the state size and the action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the agent
    agent = Agent(state_size, action_size)
    scores = []
    best_avg_rewards = -float("inf")

    # Set up the video recording
    video_path = "./videos/prioritized_dqn/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_path, episode_trigger= lambda e: e%100==0)

    for episode in range(1000): # Run for 1000 episodes
        state = env.reset()
        total_reward = 0
        done = False

        for t in range(1000):
            action = agent.select_action(state) # Choose action
            next_state, reward, done, _ = env.step(action) # Take action and observe the result

            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            state =  next_state # Move to the next state
            total_reward += reward

            if done:
                break

        agent.update_epsilon() # Update epsilon for exploration-exploitation

        # Update the target network periodically
        if episode % 100 == 0:
            agent.update_target_network()

        scores.append(total_reward)

        # Print progress every 10 episodes
        if episode % 10 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Reward: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

            # Save the best model
            if avg_score > best_avg_rewards:
                best_avg_rewards = avg_score
                torch.save(agent.online_net.state_dict(), "./models/prioritized_dqn_best.pth")

        # Early stopping if environment is solved
        if len(scores) >= 100 and np.mean(scores[-100:]) > 200:
            print(f"Solved in episode: {episode}")
            break

    env.close()

    # Plot the learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Episode - Priortized DQN")
    plt.savefig("./results/prioritized_dqn_learning_curve.png")
    plt.show()

# Testing function
def test_agent():
    env = gym.make("LunarLander-v2")
    env.seed(42)

    # Define the state size and the action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define the agent
    agent = Agent(state_size, action_size)
    agent.online_net.load_state_dict(torch.load("./models/prioritized_dqn_best.pth")) # Load the best model
    agent.online_net.to(device)
    agent.online_net.eval() # Set the network in the evaluation mode

    total_rewards = []

    # Set up the video recording for testing
    video_path = "./videos/prioritized_dqn_test/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_path)

    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.online_net(state_tensor)
            action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Test Reward over 10 episodes: {avg_reward}")
    env.close()

if __name__ == "__main__":
    # Create directories for models, results, and videos
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./videos', exist_ok=True)

    # Train the agent
    train_agent()

    # Test the agent
    test_agent()












