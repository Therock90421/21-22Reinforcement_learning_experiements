import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


if __name__ == '__main__':
    file = open("/root/RL-homework/inverted-pendulum/result/Q_Learning.log",mode='r')
    data = file.readlines()
    QL = np.zeros(70)
    x_data = np.zeros(70)
    count = 0
    #file.close()
    findword = r"最小距离稳定状态角度误差为:(.+?)rad"
    lastlist = []
    for line in data:
        #result = re.match('[2][6][0]',line)
        pattern = re.compile(findword)
        results = pattern.findall(line)
        if results:
            print(float(results[0]))
            QL[count] = float(results[0])
            x_data[count] = count*100
            count += 1
        if count == 70:
            break
    
    file = open("/root/RL-homework/inverted-pendulum/result/SARSA.log",mode='r')
    data = file.readlines()
    SARSA = np.zeros(70)
    count = 0
    #file.close()
    findword = r"最小距离稳定状态角度误差为:(.+?)rad"
    lastlist = []
    for line in data:
        #result = re.match('[2][6][0]',line)
        pattern = re.compile(findword)
        results = pattern.findall(line)
        if results:
            print(float(results[0]))
            SARSA[count] = float(results[0])
            count += 1
        if count == 70:
            break




    fig = plt.figure(figsize=(8,8))
    plt.plot(x_data, QL, 'r', color='#4169E1', alpha=0.8, linewidth=1, label='Q-Learning')
    plt.plot(x_data, SARSA, 'r', alpha=0.8, linewidth=1, label='SARSA')

    plt.legend(loc="upper right")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('loss.png')


