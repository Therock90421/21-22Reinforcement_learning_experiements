import random
import numpy as np
from fightingice_env import FightingiceEnv
import scipy.io as io

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import collections
import os

class DQN(nn.Module):
    def __init__(self, input_size, output_size, mem_len):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory = collections.deque(maxlen = mem_len)
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Dueling 架构
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, self.output_size)

    def forward(self, input):
        net_output = self.net(input)
        v = self.V(net_output)
        advantage = self.A(net_output)
        advantage = advantage - torch.mean(advantage)
        q_value = v + advantage
        return q_value

    def sample_action(self, inputs, epsilon):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        inputs = inputs.unsqueeze(0)
        q_value = self(inputs)
        action_choice = int(torch.argmax(q_value))
        return action_choice

    def save_trans(self, transition):
        self.memory.append(transition)

    def sample_memory(self, batch_size):
        s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], []
        trans_batch = random.sample(self.memory, batch_size)
        for trans in trans_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        return torch.tensor(s_ls,dtype=torch.float32),\
            torch.tensor(a_ls,dtype=torch.int64),\
            torch.tensor(r_ls,dtype=torch.float32),\
            torch.tensor(s_next_ls,dtype=torch.float32),\
            torch.tensor(done_flag_ls,dtype=torch.float32)


def train_net(Q_net, Q_target, optimizer, losses, loss_list, replay_time, gamma, batch_size):
    s, a, r, s_next, done_flag = Q_net.sample_memory(batch_size)
    # for i in range(replay_time):
    q_value = Q_net(s)
    a = torch.LongTensor(a)
    q_value = torch.gather(q_value, 1, a)

    q_t = Q_net(s_next)
    a_index = torch.argmax(q_t, 1)
    a_index = a_index.reshape((a_index.shape[0], 1))
    # print(a.size())
    # print(a_index.shape)
    q_target = Q_target(s_next)
    q_target = torch.gather(q_target, 1, a_index)
    q_target = r + gamma * q_target * done_flag

    loss = losses(q_target, q_value)
    loss_list.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]#测试时采用此模式
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    #env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]#训练时采用此模式

    gamma=0.95
    # alpha=0.1
    alpha=0.01
    epsilon=0.1
    mem_len = 30000
    load_parameter = True
    learning_rate = 1e-3
    batch_size = 32
    train_begin = 100 #4000



    Q_net = DQN(input_size = 144, output_size = 40, mem_len = mem_len)
    Q_target = DQN(input_size = 144, output_size = 40, mem_len = mem_len)
    Q_target.load_state_dict(Q_net.state_dict())

    if load_parameter:
        print('Load parameter!')
        Q_net = torch.load("./trainLog/DDQNtrainLog/weight_best.pth")
        Q_target.load_state_dict(Q_net.state_dict())

    optimizer = optim.Adam(Q_net.parameters(), lr = learning_rate)
    losses = nn.MSELoss()

    n = 0
    p = 0
    loss_list = []
    reward_list = []
    N = 500
    NumWin = 0
    Best_Score = 0
    abs_score = 0
    step_count = 0

    act = random.randint(0, 39)#初始动作


    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None
        n = n+1
        r = 0
        epsilon = max(0.01, epsilon*0.999)

        while not done:
            act = Q_net.sample_action(obs, epsilon)
            new_obs, reward, done, info = env.step(act)
            obs = new_obs
                
            if not done:
                pass
            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'),'训练局数',n)
                if info[0]>info[1]:
                    NumWin = NumWin + 1
                    abs_score = info[0] - info[1]
            else:
                # java terminates unexpectedly
                pass
        if n == N:
            break

        reward_list.append(r)

    print("finish training")
    print("获胜局数",NumWin)
