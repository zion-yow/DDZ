import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from main import DouDiZhuGame
from state_encoder import get_obs
import random
import PCplayer

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 512
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
EPISODES = 10000

class ActionEncoder:
    def __init__(self):
        self.action_dim = 55  # 54 cards + pass
        self.action_mapping = {}

    def init_action_space(self, valid_actions):
        # Create mapping for card combinations and pass
        self.action_mapping = {}
        idx = 0
        for action in valid_actions:
            if action == 'pass':
                self.action_mapping['pass'] = 54
            else:
                for card_idx in action:
                    if card_idx not in self.action_mapping:
                        self.action_mapping[card_idx] = card_idx

    def encode_action(self, action):
        vec = np.zeros(self.action_dim, dtype=np.float32)
        if action == 'pass':
            vec[54] = 1.0
        else:
            for card_idx in action:
                if 0 <= card_idx < 54:
                    vec[card_idx] = 1.0
        return vec

    def get_action_vector(self, valid_actions):
        vec = np.zeros(self.action_dim, dtype=np.float32)
        for action in valid_actions:
            if action == 'pass':
                vec[54] = 1.0
            else:
                for card_idx in action:
                    if 0 <= card_idx < 54:
                        vec[card_idx] = 1.0
        return vec
        
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 55)  # 55-dim output for card presence and pass
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DouDiZhuAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = DQN(input_dim, output_dim).float()
        self.target_net = DQN(input_dim, output_dim).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.action_encoder = ActionEncoder()  # New action encoder component
        self.action_space = []  # To be populated with valid actions
        

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            self.action_encoder.init_action_space(valid_actions)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            
            # Generate action mask from valid actions
            action_mask = self.action_encoder.get_action_vector(valid_actions)
            masked_q = q_values * torch.tensor(action_mask, dtype=torch.float32)
            
            # Select action with highest Q-value among valid actions
            selected_idx = torch.argmax(masked_q).item()
            
            # Map index back to card combination
            if selected_idx == 54:
                return 'pass'
            else:
                # Find valid action containing the selected card index
                for action in valid_actions:
                    if selected_idx in action:
                        return action
                return random.choice(valid_actions)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            # Apply action masking for valid actions
            batch_valid_actions = [exp[4] for exp in batch]
            action_mask = torch.tensor([self.action_encoder.get_action_vector(valid_actions) for valid_actions in batch_valid_actions], dtype=torch.bool)
            all_q_values = self.policy_net(states)
            masked_q_values = torch.where(action_mask, all_q_values, torch.tensor(-float('inf')))
            current_q = masked_q_values.gather(1, actions).squeeze(1)
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

def decode_action_to_cards(action, hand):
    """Convert action indices to actual card objects"""
    if action == 'pass':
        return []
    
    # Convert indices to actual card objects from the hand
    return [hand[idx] for idx in action]

def train_dqn():
    env = DouDiZhuGame(game_mode='ai_vs_ai')
    state_dim = len(get_obs(env)[0])  # Get state dimension from encoder
    action_dim = 1000  # Estimated maximum possible actions
    
    agent = DouDiZhuAgent(state_dim, action_dim)
    rewards_history = []

    for episode in range(EPISODES):
        # Initialize a new game
        env = DouDiZhuGame(game_mode='ai_vs_ai')
        env.shuffle_and_deal()
        env.determine_landlord()
        
        current_player = env.current_player
        state = get_obs(env)[current_player]
        total_reward = 0
        done = False
        
        # Play until game is done
        while not done:
            # Get valid actions for current player
            valid_actions = PCplayer.get_valid_actions(env.players[current_player].hand, env.last_played)
            
            # Add 'pass' option if not first player to play
            if env.last_played:
                valid_actions.append('pass')
                
            agent.action_encoder.init_action_space(valid_actions)
            action = agent.select_action(state, valid_actions)
            
            # Convert action indices to actual card objects for engine validation
            if action != 'pass':
                played_cards = decode_action_to_cards(action, env.players[current_player].hand)
            else:
                played_cards = []
            
            # Execute action and get game feedback
            valid, _ = env.ai_player_playing(env.players[current_player], action=played_cards)
            
            # Check if game ended and calc  ulate reward
            winner = env.check_winner()
            done = winner is not None
            
            if valid:
                # Positive reward for valid moves, bonus for winning
                if done and winner == env.players[current_player]:
                    reward = 10.0  # Big reward for winning
                else:
                    reward = 0.1  # Small reward for valid move
            else:
                # Penalty for invalid moves
                reward = 0.0
            
            # Move to next player if valid move
            if valid and not done:
                env.current_player = (env.current_player + 1) % 3
                
                # Reset last_played if two consecutive passes
                if env.continued_passed_count == 2:
                    env.last_played = []
                    env.continued_passed_count = 0
            
            # Get next state
            next_player = env.current_player
            next_state = get_obs(env)[next_player]
            
            # Store experience in agent's memory
            encoded_action = agent.action_encoder.encode_action(action)
            agent.store_experience(state, encoded_action, reward, next_state, done)
            
            # Train the agent
            agent.train()
            
            # Update for next iteration
            current_player = next_player
            state = next_state
            total_reward += reward

        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        print(f'Episode {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}')
        
        # Save model periodically
        if episode % 100 == 0:
            torch.save({
                'policy_net': agent.policy_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            }, f'doudizhu_dqn_ep{episode}.pth')

    torch.save(agent.policy_net.state_dict(), 'doudizhu_dqn.pth')

if __name__ == '__main__':
    train_dqn()