import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# ===================== 装箱环境 =====================
class BinPack1DEnv:
    def __init__(self, bin_capacity=10, item_num=8):
        self.bin_cap = bin_capacity #每个箱子的容量
        self.item_num = item_num #装箱物品的数量
        self.fixed_obs_len = 20 #NN的输入维度为20的固定长度
        self.reset() #环境初始化

    def reset(self):
        self.items = np.random.randint(low=2, high=6, size=self.item_num).tolist() #设置一个装箱物品的列表
        self.bin_used = [] #使用过的箱子
        self.cur_idx = 0 #当前要放置的物品在物品列表中的下标
        return self._get_obs() #得到的是当前的状态信息

    def _get_obs(self):
        remain_items = self.items[self.cur_idx:] #剩余的没有装箱的物品
        bin_remain = [self.bin_cap - used for used in self.bin_used] #得到的是使用的箱子中剩余的容量
        obs = remain_items + bin_remain #将剩余的物体和剩余的容量构造状态向量
        if len(obs) < self.fixed_obs_len:
            obs += [0.0] * (self.fixed_obs_len - len(obs)) #状态向量比20小，则用0补齐
        else:
            obs = obs[:self.fixed_obs_len] #如果状态向量比20大，则直接截取前20的值
        return torch.tensor(obs, dtype=torch.float32) #将obs生成张量，返回状态的张量

    def step(self, action_bin): #选择一个行为action_bin就是箱子，将对象放在箱子里
        item_size = self.items[self.cur_idx]
        reward = 0.0
        if action_bin < len(self.bin_used):
            if self.bin_used[action_bin] + item_size <= self.bin_cap: #使用当前的箱子装物体
                self.bin_used[action_bin] += item_size
                reward = item_size / self.bin_cap
            else:
                self.bin_used.append(item_size) #新开一个箱子装当前的物体
                reward = item_size / self.bin_cap - 0.2 #当前使用的箱子的剩余空间无法放入新的物体，放入箱子的收益减去新开箱子的花费
        else:
            self.bin_used.append(item_size)
            reward = item_size / self.bin_cap - 0.2 #直接选择一个新开的箱子得到的收益

        self.cur_idx += 1 #当前的物品数加1
        done = (self.cur_idx >= len(self.items)) #如果物品全部放置完
        next_obs = self._get_obs() #放置完当前物品后，重新计算环境的状态
        info = {"bin_count": len(self.bin_used), "items": self.items} #info中指示使用的箱子总数和物品的列表
        return next_obs, reward, done, info

# ===================== Actor-Critic 网络 =====================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x): #x就是输入的状态张量
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feat = self.shared(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value #通过NN输出action的概率分布，和价值函数value

    def select_action(self, obs): #根据概率分布选择一个
        logits, val = self(obs) #调用forward函数，直接生成输出action的概率分布，和价值函数value（val)
        dist = Categorical(logits=logits) #Actor输出的原始打分经过softmax处理，转化成概率值
        action = dist.sample() #基于dist的概率分布，随机选择一个箱子
        log_prob = dist.log_prob(action) #计算这个采用动作对应的对数概率在计算actor loss的时候直接使用
        return action.item(), log_prob, val #输出普通的action的值，不是tensor

    def select_greedy_action(self, obs): #选择一个概率最大的action，并返回action的值
        logits, val = self(obs)
        action = torch.argmax(logits, dim=-1)
        return action.item()

# ===================== 验证评估函数 =====================
def evaluate(net, eval_episodes=50, bin_cap=10):
    env_eval = BinPack1DEnv() #构造评估环境
    net.eval() #把NN切换为评估模型
    total_r = 0.0 #总的收益
    total_bin = 0 #总的箱子数
    total_util = 0.0 #总的效用总数
    with torch.no_grad():
        for _ in range(eval_episodes):
            obs = env_eval.reset() #重新设置环境
            done = False
            ep_r = 0  #一个episode的收益值
            while not done:
                act = net.select_greedy_action(obs) #根据最大概率选择一个行为
                obs, r, done, info = env_eval.step(act) #基于最大概率选择的行为，得到环境的收益
                ep_r += r #episode收益累计
            item_sum = sum(info["items"]) #物体的重量之和
            bin_cnt = info["bin_count"]   #箱子的数目
            util = item_sum / (bin_cnt * bin_cap) #一个episode中单位容量所容纳的物体的数量，体现的是箱子的利用率
            total_r += ep_r #一个episode总的reward
            total_bin += bin_cnt #一个episode总的箱子数加到评估的episode的集合中
            total_util += util #效用累计
    avg_r = total_r / eval_episodes
    avg_bin = total_bin / eval_episodes
    avg_util = total_util / eval_episodes
    net.train() #切换到训练模式
    return avg_r, avg_bin, avg_util

# ===================== A2C 主训练（同步A3C，稳定无nan） =====================
if __name__ == "__main__":
    OBS_DIM = 20 #设置输入状态的维度
    ACT_DIM = 10 #最多有十个箱子
    MAX_EPISODES = 1000 #
    GAMMA = 0.99
    LR = 3e-4

    model = ActorCritic(OBS_DIM, ACT_DIM) #建立A2C智能NN
    optimizer = optim.Adam(model.parameters(), lr=LR) #设置优化器

    for ep in range(1, MAX_EPISODES+1): #设置训练的最大episode
        env = BinPack1DEnv() # 设置环境实例
        obs = env.reset() #初始化环境
        done = False
        log_probs = [] #action的概率分布
        values = [] #用于评价状态值的好坏
        rewards = [] #收益函数
        total_r = 0.0 #总的收益

        while not done:
            action, logp, val = model.select_action(obs) #输入状态值，基于NN生成的action的概率选择一个action输出，同时输出action的概率的对数，状态的价值
            next_obs, r, done, info = env.step(action) #基于选择的action，生成下一个状态值，收益值等
            log_probs.append(logp) #
            values.append(val)
            rewards.append(r)
            obs = next_obs
            total_r += r

        # 计算回报
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R) #返回每一步的收益值，returns是所有step的列表
        returns = torch.tensor(returns, dtype=torch.float32) #改为张量
        log_probs = torch.stack(log_probs) #形成了基于物体数目维的对数概率之和张量
        values = torch.cat(values).squeeze() #形成了基于物体数目维的张量

        advantage = returns - values #两个同维度的张量
        actor_loss = (-log_probs * advantage.detach()).mean() #
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸nan（非常关键！）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        # 验证
        if ep % 50 == 0:
            avg_r, avg_bin, avg_util = evaluate(model, eval_episodes=30, bin_cap=10)
            print(f"===== Eval @ Ep {ep:4d} =====")
            print(f"Train Episode Reward: {total_r:.2f} | Avg Reward: {avg_r:.2f}")
            print(f"Average Bins Used: {avg_bin:.2f} | Avg Utilization: {avg_util:.3f}\n")

    print("\n===== Final Evaluation =====")
    avg_r, avg_bin, avg_util = evaluate(model, eval_episodes=100, bin_cap=10)
    print(f"Final Avg Reward: {avg_r:.2f}")
    print(f"Final Avg Bins Used: {avg_bin:.2f}")
    print(f"Final Avg Utilization: {avg_util:.3f}")
