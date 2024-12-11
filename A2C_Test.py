import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym 
import qwop_gym
import matplotlib.pyplot as plt
import time

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_out = self.shared(state)
        policy = self.actor(shared_out)
        value = self.critic(shared_out)
        return policy, value

class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=5e-3, gamma=0.99, entropy_coeff=0.001):
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.decay_rate = 0.99
        self.final_entropy_coeff = 0.001
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        #lr started as 5e-4

    def compute_returns(self, rewards, dones, last_value):
        """Compute discounted returns."""
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)
    
    # def get_entropy_coeff(self, current_step, final_entropy_coeff, initial_entropy_coeff, decay_rate):
    #     return final_entropy_coeff + (initial_entropy_coeff - final_entropy_coeff) * np.exp(-decay_rate * current_step)

    def update(self, states, actions, rewards, dones, last_value, episode, step):
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        returns = self.compute_returns(rewards, dones, last_value)

        policies, values = self.actor_critic(states)
        values = values.squeeze(-1)  # Remove last dimension

        advantages = returns - values

        # Compute actor loss
        log_probs = torch.log(policies.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Add entropy bonus
        entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=-1).mean()
        # print("entropy: ", entropy)
        # time.sleep(2)
        # self.entropy_coeff = self.get_entropy_coeff(step, 0.001, 0.01, 0.001)
        actor_loss -= self.entropy_coeff * entropy

        # Compute critic loss
        critic_loss = advantages.pow(2).mean()

        # Total loss
        loss = actor_loss + critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return entropy

    

if __name__ == "__main__":

    env = gym.make("QWOP-v1", browser="C:/Program Files/Google/Chrome/Application/chrome.exe", driver="C:/Users/AJ/Documents/RLProject/chromedriver-win64 (1)/chromedriver-win64/chromedriver.exe", 
                   time_cost_mult=0, frames_per_step=1)
    state, _ = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    

    num_runs = 1
    num_episodes = 250
    runDistances = np.zeros((num_runs, num_episodes))
    runTimes = np.zeros((num_runs, num_episodes))
    
    for run in range(num_runs):
        agent = A2C(state_dim, action_dim)
        distancesTraveled = np.zeros(num_episodes)
        times = np.zeros(num_episodes)
        stepCount = 0
        entropies = np.zeros(num_episodes)
        for episode in range(num_episodes):
            episodeStepCount = 0
            state, _ = env.reset()
            done = False
            states, actions, rewards, dones = [], [], [], []
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                policy, value = agent.actor_critic(state_tensor)
                action = torch.multinomial(policy, 1).item()

                next_state, reward, done, trunc, info = env.step(action)
                # print("reward: ", reward)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                state = next_state
                stepCount += 1
                episodeStepCount += 1

            # Compute the last value
            last_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, last_value = agent.actor_critic(last_state_tensor)

            # Update the network
            entropy = agent.update(states, actions, rewards, dones, last_value.item(), episode, stepCount)

            # Log progress
            print(f"Episode {episode + 1}, Most Recent Reward: {rewards[episodeStepCount-1]}", "Distance traveled: ", info["distance"], "Success?", info["is_success"])
            distancesTraveled[episode] = info["distance"]
            times[episode] = info['time']
            entropies[episode] = entropy
            
            # # Every so often log distance traveled
            # if episode % 5 == 0:
            #     print("Distance traveled: ", info["distance"])
        runDistances[run, :] = distancesTraveled
        runTimes[run, :] = times

    env.close()

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_episodes+1), runDistances.mean(axis=0)) #Plot the mean across runs
    err = runDistances.std(axis=0) / np.sqrt(runDistances.shape[0]) * 1.96
    plt.fill_between(range(runDistances.shape[1]), runDistances.mean(axis=0) - err, runDistances.mean(axis=0) + err, alpha=0.3)
    plt.title("Distance Traveled v. Episode Number")
    plt.ylabel("Distance Traveled (meters)")
    plt.xlabel("Episode")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_episodes+1), runTimes.mean(axis=0)) #Plot the mean across runs
    timeErr = runTimes.std(axis=0) / np.sqrt(runTimes.shape[0]) * 1.96
    plt.fill_between(range(runTimes.shape[1]), runTimes.mean(axis=0) - timeErr, runTimes.mean(axis=0) + timeErr, alpha=0.3)
    plt.title("Episode Time vs. Episode Number")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")

    # plt.plot(range(1, num_episodes+1), entropies) #Plot the mean across runs
    # plt.title("Entropy vs. Episode Number")
    # plt.xlabel("Episode")
    # plt.ylabel("Entropy")

    plt.show()


#Entropy coeff was 0.01 to start and used 250 episodes initially
