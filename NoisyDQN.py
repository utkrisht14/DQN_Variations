import gym
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque
import os
import matplotlib.pyplot as plt

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set the random seed "42" for the reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.995
EPSILON_MIN = 0.01

# Define the noisy layer instead of standard linear layer
class NoisyLayer(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for the weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        # Learnable parameters for the biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma_init = sigma_init # Initial value for the sigma
        self.reset_parameters() # Initialize weight and bias
        self.reset_noise() # Initialize noise for exploration

    def reset_parameters(self):
        """ Initialize weight and bias parameters using uniform distribution  """
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """  Reset the noise for both weights and biases using a factorized noise approach """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        # Apply noise to the weights and biases during training
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

    def _scale_noise(self, size):
        # Generate noise using a Gaussian distribution and transform it
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

#  Define the Noisy DQN Architecture
class NoisyDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = NoisyLayer(state_size, 128)
        self.fc2 = NoisyLayer(128, 128)
        self.fc3 = NoisyLayer(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        # Reset noise in each noisy layer
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

# Define experience replay buffer for the storing transitions
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        # Add experience -> state, action, reward, next_state, done to buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        """ Sample a random size of buffer from the experience """
        experiences = random.sample(self.buffer, batch_size)
        return experiences

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)


# Agent class that interacts with the environment
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000) # Set the memory size

        # Initialize online and offline memory
        self.online_net = NoisyDQN(state_size, action_size).to(device)
        self.target_net = NoisyDQN(state_size, action_size).to(device)
        self.update_target_network()

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        """ Copy weights from the online network to target network  """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state):
        """ Select action based on the current policy """
        self.online_net.reset_noise()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.online_net(state)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """ Store experience in the replay buffer """
        self.memory.add((state, action, reward, next_state, done))

    def train_step(self):
        """ Train the agent using the batch of experience """
        if len(self.memory) < BATCH_SIZE:
            return

        experiences = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert experiences to torch tensor
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Reset noise in both networks
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # Current Q-values from the Q-network
        actions = actions.unsqueeze(1)
        q_values = self.online_net(states).gather(1, actions).squeeze(1)

        # Compute target Q-values using Double DQN technique
        next_actions = self.online_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss between current and target Q-values
        loss = self.loss_fn(q_values, target_q_values.detach())

        # Optimize the online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training function to train the agent
def train_agent():
    env = gym.make("LunarLander-v2")
    env.seed(42)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    scores = []
    best_avg_reward = -float("inf")

    # For video recording
    video_path = "./videos/noisy_dqn/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_path, episode_trigger= lambda e: e%100==0)

    # training loop
    for episode in range(1000): # Number of episodes to run for
        state = env.reset()
        total_reward = 0
        done = False

        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action) # Environment responds

            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step() # Train the agent

            state = next_state
            total_reward += reward

            if done:
                break

        # update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()

        scores.append(total_reward) # Track rewards per episode

        # Print progress and save the model
        if episode % 10 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Reward: {avg_score:.2f}")

            # Save the best model if performance improves
            if avg_score > best_avg_reward:
                best_avg_reward = avg_score
                torch.save(agent.online_net.state_dict(), "./models/noisy_dqn_best.pth")

        # Early stopping if solved
        if np.mean(scores[-100:]) >= 200:
            print(f"Solved in episode: {episode}")
            break

    env.close()

    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Episode (Noisy DQN)")
    plt.savefig("./results/noisy_dqn_learning_curve.png")
    plt.show()

# Testing function to evaluate the agent
def test_agent():
    env = gym.make("LunarLander-v2")
    env.seed(42)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    agent.online_net.load_state_dict(torch.load("./models/noisy_dqn_best.pth"))
    # Set the network for the evaluation mode
    agent.online_net.eval()

    total_rewards = []

    # For the video recording
    video_path = "./videos/noisy_dqn_test/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_path)

    # Test loop for multiple steps
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.online_net(state_tensor)
            action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode: {episode}, Total Epsiode: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Test Reward over 10 episodes: {avg_reward}")
    env.close()

if __name__ == "__main__":
    # Create directories for the model, results and videos
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./videos", exist_ok=True)

    # Train the agent
    train_agent()

    # Test the agent
    test_agent()


