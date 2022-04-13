# coding=utf8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import env


# 存储结果
def saveResult(dic_path, Q, angles, total_acts):
    np.save(dic_path + '/SARSAoptimal.npy', Q)
    sequence_path = dic_path + '/SARSA_state_path.txt'
    with open(sequence_path, 'w', encoding='utf8') as f:
        for act in total_acts:
            f.write(str(act) + " ")
        f.write('\n')
        for location in angles:
            f.write(str(location) + ' ')

def show_result(path = './result/SARSA_state_path.txt'):
    with open(path, 'r', encoding='utf8') as f:
        lines = [line for line in f.readlines()]
    angles = lines[-1].strip().split()
    angles = [float(loc) for loc in angles]
    print("最终距离稳定状态角度误差为:%f rad" % np.abs((angles[-1] + np.pi) % (2*np.pi) - np.pi))
    fig = plt.figure()
    plt.title("Inverted Pendulum")
    plt.axis('off')
    ims = []
    for a in angles:
        x = np.array([0, np.sin(a)])
        y = np.array([0, np.cos(a)])
        im = plt.plot(x, y,color='g')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=5, repeat_delay=1000)
    ani.save("SARSA.gif",writer='pillow')

def train(num_alpha, num_diff_alpha):

    #状态限制
    min_alpha, max_alpha = -np.pi, np.pi
    min_diff_alpha, max_diff_alpha = -15 * np.pi, 15*np.pi
    #Q动作状态-价值函数
    Q = np.zeros([3, num_alpha, num_diff_alpha])
    #离散动作
    actions = np.array([-3, 0, 3])
    # 迭代次数
    N = 20000
    #ε-greedy超参
    epsilon = 1.0
    epsilon_limit = 0.01
    # 初始学习率
    lr = 1.0
    #衰减率
    decay = 0.9995
    #折扣因子
    gamma = 0.98
    # 初始状态
    state_alpha, state_diff_alpha = -np.pi, 0
    total_step = 300
    optimal = -1e7
    finalerror = 2*np.pi
    total_angle = np.pi
    # 探索步长限制
    step_control = 300
    #收敛目标控制
    stable_target = 2
    converted_alpha, converted_diff_alpha = 0.05, 0.01
    #迭代
    for i in range(N):
        alpha, diff_alpha, iteration, total_reward = state_alpha, state_diff_alpha, 0, 0
        indice_alpha, indice_diff_alpha = env.get_indice(alpha, diff_alpha, num_alpha, num_diff_alpha)
        greedy_action = env.get_greedy_action(Q, indice_alpha, indice_diff_alpha, epsilon)
        stable_state = 0
        epsilon = max(decay * epsilon, epsilon_limit)
        lr = decay * lr
        angles, total_acts = [], []
        angles.append(alpha)
        error_a = np.abs((alpha-min_alpha) % (max_alpha - min_alpha) + min_alpha)
        while iteration < step_control:
            iteration += 1
            total_acts.append(actions[greedy_action])
            new_alpha, new_diff_alpha = env.get_new_state(alpha, diff_alpha, actions[greedy_action])
            error = np.abs((new_alpha-min_alpha) % (max_alpha - min_alpha) + min_alpha)
            if error < error_a:
                min_angle = new_alpha
            error_a = error
            #收敛控制
            if error_a < converted_alpha and np.abs(diff_alpha) < converted_diff_alpha:
                angles.append(new_alpha)
                stable_state += 1
                print(stable_state)
                if stable_state == stable_target:
                    break
            indice_new_alpha, indice_new_diff_alpha = env.get_indice(new_alpha, new_diff_alpha, num_alpha, num_diff_alpha)
            greedy_new_action = env.get_greedy_action(Q, indice_new_alpha, indice_new_diff_alpha, epsilon)
            reward = env.get_reward(alpha, diff_alpha, actions[greedy_action])
            update_Q = reward + gamma * Q[greedy_new_action][indice_new_alpha][indice_new_diff_alpha] - Q[greedy_action][indice_alpha][indice_diff_alpha]
            Q[greedy_action][indice_alpha][indice_diff_alpha] += lr * update_Q
            alpha, diff_alpha, indice_alpha, indice_diff_alpha, greedy_action = new_alpha, new_diff_alpha, indice_new_alpha, indice_new_diff_alpha, greedy_new_action
            angles.append(alpha)
            total_reward += reward
        total_step = iteration if iteration < total_step else total_step
        min_angle = np.abs((min_angle + np.pi) % (2 * np.pi) - np.pi)
        total_angle = min_angle if min_angle < total_angle else total_angle
        # 若该次迭代到达终点并且误差更小，或者奖励更高，则存储
        if iteration < step_control:
            if error_a < finalerror or(error_a == finalerror and total_reward > optimal):
                optimal, finalerror = total_reward, error_a
            saveResult('./result', Q, angles, total_acts)
        # 打印迭代信息
        if (i+1) % 100 == 0:
            print("第%d 次迭代：搜索步长限制%d 最小距离稳定状态角度误差为:%f rad 最小收敛步长为 %d " % (i+1, step_control, total_angle ,total_step))
            total_step = 300
            total_angle = np.pi
    show_result()


if __name__ == '__main__':
    # 这是最优训练结果
    # show_result('./result/pre_trained_SARSAsequence.txt')

    # num_alpha是摆杆的角度空间划分数量  num_diff_alpha是摆杆角速度空间划分数量 可以修改该参数
    train(num_alpha = 200, num_diff_alpha = 200)

    # 仅显示训练结果
    # show_result()
