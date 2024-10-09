# Start by importing necessary libraries
import gym
import numpy as np
import torch
import random
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from collections import deque



# Set the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set the seed for the reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Define some hyeperparameters
BATCH_SIZE = 64
GAMMA = 0.995
EPSILON_MIN = 0.01

# Next we define the DQN network
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

# Define the Experience Replay Buffer
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

# Define the agent class
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.memory = ReplayMemory(100000) # Set the memory size

        # Initialize networks
        self.online_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.update_target_network()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        """ Function to update the target network """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        # Exploitation
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
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

        # Convert them into the tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        # Current Q values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN updates
        next_actions = self.online_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values.detach())

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= 0.995 # Multiply the epsilon by the epsilon decay

# Training function
def train_agent():
    env = gym.make("LunarLander-v2")
    env.seed()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    scores = []
    best_avg_reward = -float("inf")

    # Video recording setup
    video_path = "./videos/dqn/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: x % 100 == 0)

    for episode in range(1000):  # Number of episodes to run for
        state = env.reset()
        total_reward = 0
        done = False

        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        # Update epsilon
        agent.update_epsilon()

        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()

        # Append total reward for this episode
        scores.append(total_reward)

        # Print progress every 10 episodes
        if episode % 10 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Reward: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

            # Save best model
            if avg_score > best_avg_reward:
                best_avg_reward = avg_score
                torch.save(agent.online_net.state_dict(), "./models/dqn_best.pth")

        # Early stopping
        if np.mean(scores[-100:]) >= 200:
            print(f"Solved in episode {episode}")
            break  # Optionally break the loop if solved

    env.close()

    # Plotting learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward vs Episode (DQN)")
    plt.savefig("./results/dqn_learning_curve.png")
    plt.show()


# Define the testing function
def test_agent():
    env = gym.make("LunarLander-v2")
    env.seed(42)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    agent.online_net.load_state_dict(torch.load("./models/dqn_best.pth"))
    agent.online_net.eval()

    total_rewards = []

    os.makedirs("./videos/dqn_test/", exist_ok=True)
    env = gym.wrappers.RecordVideo(env, "./videos/dqn_test/") # Define the environment and the video path

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

        total_reward.append(total_reward)
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















