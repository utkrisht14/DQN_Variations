import gym
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque
import os
import matplotlib.pyplot as plt

# Set the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set the random seed
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Set the hyperparameters
BATCH_SIZE = 64
GAMMA = 0.995
EPSILON_MIN = 0.01

# Define the DuelingDQN network architecture
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size,128)
        self.fc2 = nn.Linear(128, 128)

        # For value and advantage streams
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Experience replay buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        return experiences

    def __len__(self):
        return len(self.buffer)

# Agent class
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.00
        self.memory = ReplayMemory(100000)  # Set the memory size

        # Initialize networks
        self.online_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net = DuelingDQN(state_size, action_size).to(device)
        self.update_target_net()

        # Define the optimizer and the loss function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def update_target_net(self):
        """ Update the target network """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state):
        """
        Select the action to act in the environment
        """
        # Exploration state
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)  # Add unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_net(state)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        experiences = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert them to the tensor
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)  # Add unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).astype(np.uint8)).to(device)

        # Current Q values
        q_values = self.online_net(states).gather(1, actions).squeeze(1)

        # Double DQN update
        next_actions = self.online_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values.detach())

        # Optimize the online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= 0.995  # Decay the epsilon with constant rate

# Training agent
def train_agent():
    env = gym.make("LunarLander-v2")
    env.seed(42)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    scores = []
    best_avg_reward = -float("inf")

    # Setting the video recording
    video_path = "./videos/dueling_dqn/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: x%100 == 0)

    for episode in range(1000):  # Set the number of episodes
        state = env.reset()
        total_reward = 0
        done = False

        for t in range(1000):  # Set the max number of steps
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_epsilon()

        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_net()

        scores.append(total_reward)

        # Print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Reward: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f} ")

            # Save the best model
            if avg_score > best_avg_reward:
                best_avg_reward = avg_score
                torch.save(agent.online_net.state_dict(), "./models/dueling_dqn_best.pth")

        # Early stopping
        if np.mean(scores[-100:]) >= 200:
            print(f"Solved in episode {episode}")
            break

    env.close()

    # Plotting the learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Episode (Dueling DQN)")
    plt.savefig("./results/dueling_dqn_learning_curve.png")
    plt.show()

# Testing the agent
def test_agent():
    env = gym.make("LunarLander-v2")
    env.seed(42)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    agent.online_net.load_state_dict(torch.load("./models/dueling_dqn_best.pth"))
    agent.online_net.eval()

    total_rewards = []

    # Video recording steps
    video_path = "./videos/dueling_dqn_test/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_path)

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
        print(f"Test Episode {episode}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Test Reward over 10 episodes: {avg_reward}")
    env.close()

if __name__ == "__main__":
    # Create directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./videos', exist_ok=True)

    # Train the agent
    train_agent()

    # Test the agent
    test_agent()
