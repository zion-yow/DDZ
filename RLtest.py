import os
# 設置環境變量以解決 OpenMP 重複初始化問題
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 限制 PyTorch 使用的線程數
os.environ['OMP_NUM_THREADS'] = '1'
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from IPython import display

# 設置 PyTorch 線程數
torch.set_num_threads(4)

# 設置中文字體支持
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題


# 設置超參數
GAMMA = 0.99       # 折扣因子
LR = 1e-2          # 學習率
BATCH_SIZE = 128    # 批次大小
MEMORY_SIZE = 10000 # 記憶回放容量
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.01   # 最低探索率
EPSILON_DECAY = 0.995 # 探索率衰減
TARGET_UPDATE = 5   # 目標網絡更新頻率
EPISODES = 500       # 訓練回合數
RENDER_EVERY = 50    # 每隔多少回合渲染一次環境
# 檢測並設置設備 (CPU 或 GPU)
# 添加更詳細的 GPU 檢測信息

print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("GPU 型號:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("未檢測到 GPU，使用 CPU 進行訓練")
    print("PyTorch 版本:", torch.__version__)
    device = torch.device("cpu")

print(f"使用設備: {device}")

# 檢測並設置設備 (CPU 或 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 🎮 1. 環境封裝
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # 環境的狀態維度 (4)
action_dim = env.action_space.n             # 行動數量 (2)

# 🧠 2. 定義 DQN 神經網絡
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # 第一層隱藏層
        self.fc2 = nn.Linear(128, 128)        # 第二層隱藏層
        self.fc3 = nn.Linear(128, action_dim) # 輸出層：每個動作的 Q 值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 輸出 Q 值

# 🎯 3. DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START  # 初始探索率
        self.device = device  # 使用檢測到的設備

        # 兩個神經網絡（主網絡和目標網絡）- 確保移動到 GPU
        self.policy_net = DQN(state_dim, action_dim).float().to(self.device)
        self.target_net = DQN(state_dim, action_dim).float().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 同步參數

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  # Adam 優化器
        self.memory = deque(maxlen=MEMORY_SIZE)  # 記憶回放緩存

    # 📌 存儲經驗
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 🤖 選擇動作（ε-greedy 策略）
    def select_action(self, state):
        if random.random() < self.epsilon:  # 探索
            return random.randint(0, self.action_dim - 1)
        else:  # 利用 Q 值
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()

    # 🚀 訓練 DQN
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return  # 記憶庫不夠時不訓練

        # 隨機取樣
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 轉換為張量並移動到 GPU
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 計算當前 Q 值
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # 計算目標 Q 值（使用 target_net）在訓練前已采取了行動得到了next_states
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            # dones == 1時gameover，不獲得獎勵，網絡不會更新
            target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

        # 損失函數 & 反向傳播
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # 返回損失值

    # 🔄 更新目標網絡
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # 📉 更新探索率
    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# 🔥 4. 訓練主函數
def train_dqn():
    agent = DQNAgent(state_dim, action_dim)

    # 用於記錄訓練數據的列表
    rewards_history = []
    avg_rewards_history = []
    epsilon_history = []
    losses = []
    
    # 不預先創建渲染環境，而是在需要時創建
    render_env = None

    plt.figure(figsize=(12, 8))

    # 每一局
    for episode in range(EPISODES):
        state = env.reset()[0]  # 取得初始狀態
        total_reward = 0
        episode_losses = []
        
        # 每一幀
        while True:
            action = agent.select_action(state)  # 選擇動作
            next_state, reward, done, _, _ = env.step(action)  # 與環境交互
            agent.store_experience(state, action, reward, next_state, done)  # 存儲經驗

            state = next_state
            total_reward += reward

            # 訓練並記錄損失
            if len(agent.memory) >= BATCH_SIZE:
                loss = agent.train()  # 訓練 DQN，並返回損失值
                if loss is not None:
                    episode_losses.append(loss)

            if done:
                break

        # 每 5 回合更新目標網絡
        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()

        agent.update_epsilon()  # 更新探索率

        # 記錄訓練數據
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])  # 最近100回合的平均獎勵
        avg_rewards_history.append(avg_reward)
        epsilon_history.append(agent.epsilon)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        print(f"Episode {episode}: Total Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # 每隔一定回合數繪製訓練曲線
        if episode % 10 == 0:
            plt.clf()
            
            # 繪製獎勵曲線
            plt.subplot(2, 2, 1)
            plt.plot(rewards_history)
            plt.plot(avg_rewards_history)
            plt.title('獎勵隨回合變化')
            plt.xlabel('回合')
            plt.ylabel('獎勵')
            plt.legend(['每回合獎勵', '平均獎勵'])
            
            # 繪製探索率曲線
            plt.subplot(2, 2, 2)
            plt.plot(epsilon_history)
            plt.title('探索率隨回合變化')
            plt.xlabel('回合')
            plt.ylabel('探索率 (ε)')
            
            # 繪製損失曲線
            if losses:
                plt.subplot(2, 2, 3)
                plt.plot(losses)
                plt.title('損失隨回合變化')
                plt.xlabel('回合')
                plt.ylabel('損失')
            
            plt.tight_layout()
            plt.savefig(f'f:/projects/DouZero/training_progress.png')
            display.clear_output(wait=True)
            display.display(plt.gcf())
        
        # 每隔 RENDER_EVERY 回合渲染一次環境
        if episode % RENDER_EVERY == 0 and episode > 0:
            # 在需要時創建或重新創建渲染環境
            render_env = gym.make("CartPole-v1", render_mode="human")
            visualize_agent(agent, render_env)
            # 使用後立即關閉
            try:
                render_env.close()
            except:
                pass

    plt.close()
    
    # 保存訓練好的模型
    torch.save(agent.policy_net.state_dict(), 'f:/projects/DouZero/dqn_cartpole_model.pth')
    
    # 繪製最終的訓練曲線
    plot_training_results(rewards_history, avg_rewards_history, epsilon_history, losses)


def visualize_agent(agent, render_env, max_steps=500):
    """渲染環境以可視化智能體的表現"""
    try:
        # 重新創建渲染環境以避免 pygame 錯誤
        render_env.close()
        render_env = gym.make("CartPole-v1", render_mode="human")
        
        state = render_env.reset()[0]
        total_reward = 0
        
        for t in range(max_steps):
            # 使用當前策略選擇動作（無探索）- 移動到 GPU
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                action = torch.argmax(agent.policy_net(state_tensor)).item()
            
            next_state, reward, done, _, _ = render_env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"可視化回合獎勵: {total_reward}")
    except Exception as e:
        print(f"渲染過程中發生錯誤: {e}")
    finally:
        # 確保環境被正確關閉
        try:
            render_env.close()
        except:
            pass

# 繪製最終的訓練結果
def plot_training_results(rewards, avg_rewards, epsilons, losses):
    """繪製並保存完整的訓練結果圖表"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.title('訓練獎勵')
    plt.xlabel('回合')
    plt.ylabel('獎勵')
    plt.legend(['每回合獎勵', '平均獎勵'])
    
    plt.subplot(2, 2, 2)
    plt.plot(epsilons)
    plt.title('探索率變化')
    plt.xlabel('回合')
    plt.ylabel('探索率 (ε)')
    
    if losses:
        plt.subplot(2, 2, 3)
        plt.plot(losses)
        plt.title('訓練損失')
        plt.xlabel('回合')
        plt.ylabel('損失')
    
    plt.tight_layout()
    plt.savefig('f:/projects/DouZero/final_training_results.png')
    plt.show()


# 🚀 執行訓練
if __name__ == "__main__":
    train_dqn()
