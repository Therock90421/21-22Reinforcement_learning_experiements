# coding=utf8
#倒立摆是将一个物体固定在一个圆盘的非中心点位置, 由直流电机驱动将其在
#垂直平面内进行旋转控制的系统 (图 1). 由于输入电压是受限的, 因此电机并不能
#提供足够的动力直接将摆杆推完一圈. 相反, 需要来回摆动收集足够的能量, 然后
#才能将摆杆推起并稳定在最高点.
import numpy as np
import matplotlib.pyplot as plt

#倒立摆系统连续时间动力学模型
def get_diff_diff_alpha(alpha, diff_alpha, u):
    m = 0.055
    g = 9.81
    l = 0.042
    J = 1.91e-4
    b = 3e-6
    K = 0.0536
    R = 9.5
    return (m*g*l*np.sin(alpha) - b*diff_alpha-(K**2)*diff_alpha/R + K*u/R)/J

#倒立摆系统离散时间动力学模型（状态转移）
def get_new_state(old_alpha, old_diff_alpha, action):
    Ts = 0.005
    new_alpha = old_alpha + Ts * old_diff_alpha
    new_diff_alpha = old_diff_alpha + Ts * get_diff_diff_alpha(old_alpha, old_diff_alpha, action)
    #速度限制：[-15pi，15pi]
    max_diff_alpha = 15 * np.pi
    if new_diff_alpha < -max_diff_alpha:
        new_diff_alpha = -max_diff_alpha
    elif new_diff_alpha > max_diff_alpha:
        new_diff_alpha = max_diff_alpha
    return new_alpha, new_diff_alpha

#获取当前状态在离散化状态空间的下标
def get_indice(alpha, diff_alpha, num_alpha, num_diff_alpha):
    max_diff_alpha = 15 * np.pi
    #正则化alpha，范围[-pi，pi)
    norm_alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    #alpha下标范围[0,num_alpha-1]
    indice_alpha = int((norm_alpha + np.pi)/(2*np.pi) * num_alpha)
    #diff_alpha下标范围[0,num_diff_alpha-1]
    indice_diff_alpha = int((diff_alpha + max_diff_alpha)/(2*max_diff_alpha) * num_diff_alpha)
    if indice_diff_alpha == num_diff_alpha: 
        indice_diff_alpha -= 1
    return indice_alpha, indice_diff_alpha

#ε-greedy搜索策略  输入Q动作状态-价值函数和状态下标，返回贪心动作
def get_greedy_action(Q, indice_alpha, indice_diff_alpha, epsilon):
    opt_action = np.argmax(Q[:,indice_alpha,indice_diff_alpha])
    if np.random.random() > epsilon:
        return opt_action
    else:
        return np.random.choice([0,1,2]) #对应[-3，0，3]3个动作

def get_reward(alpha, diff_alpha, action):
    Rrew = 1
    return -(5*alpha**2+0.1*diff_alpha**2)-Rrew*action**2