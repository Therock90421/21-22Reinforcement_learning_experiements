import random
import numpy as np
from fightingice_env import FightingiceEnv
import scipy.io as io

def loadweight(weight_file=None):
    weight= io.loadmat(weight_file)#训练好的权重进行测试
    weight=weight['weight'] #[144*40,1]
    weight = weight.reshape(40,144)
    '''
    np.set_printoptions(threshold=sys.maxsize)
    with open("weight.txt","w") as f:
        f.write(str(weight))
    with open("weight1.txt","w") as f:
        f.write(str(weight.reshape(40,144)))
    '''
    return weight

if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    #env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]#测试时采用此模式
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]#训练时采用此模式


    gamma=0.95
    # alpha=0.1
    alpha=0.01
    epsilon=0.1
    weight1 = loadweight("./trainLog/DoubleQLtrainLog/weight1_best")
    weight2 = loadweight("./trainLog/DoubleQLtrainLog/weight2_best")


    n = 0
    p = 0
    RewardData = []
    N = 500
    NumWin = 0
    Best_Score = 0
    abs_score = 0

    act = random.randint(0, 39)#初始动作

    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None
        n = n+1
        r = 0

        while not done:
            if n<N:
                if np.mod(n,5)==0:
                    string="./trainLog/DoubleQLtrainLog/weight1"+str(n+p)
                    io.savemat(string, {'weight': weight1})
                    string="./trainLog/DoubleQLtrainLog/weight2"+str(n+p)
                    io.savemat(string, {'weight': weight2})
                    string="./trainLog/DoubleQLtrainLog/reward"+str(n+p)
                    io.savemat(string, {'r': RewardData})
                if Best_Score <= abs_score:
                    Best_Score = abs_score
                    string="./trainLog/DoubleQLtrainLog/weight1_best"
                    io.savemat(string, {'weight': weight1})
                    string="./trainLog/DoubleQLtrainLog/weight2_best"
                    io.savemat(string, {'weight': weight2})
                    string="./trainLog/DoubleQLtrainLog/reward_best"
                    io.savemat(string, {'r': RewardData})
            else:
                print("训练结束")
                string="./trainLog/DoubleQLtrainLog/weight1"+str(n+p)
                io.savemat(string, {'weight': weight1})
                string="./trainLog/DoubleQLtrainLog/weight2"+str(n+p)
                io.savemat(string, {'weight': weight2})
                string="./trainLog/DoubleQLtrainLog/reward"+str(n+p)
                io.savemat(string, {'r': RewardData})
                break

            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            # new_obs, reward, done, info = env.step(act)
            act = np.argmax(np.dot(weight1, obs) + np.dot(weight2, obs))
            if random.random() < epsilon:
                act = random.randint(0, 39)
            else:
                pass
            new_obs, reward, done, info = env.step(act)

            if not done:
                # TODO: (main part) learn with data (obs, act, reward, new_obs)
                if random.random() < 0.5:
                    #更新Q1
                    Q1_act = np.argmax(np.dot(weight1, new_obs))
                    delta = reward + gamma * np.dot(new_obs, weight2[Q1_act]) - np.dot(obs, weight1[act])
                    weight1[act] = weight1[act] + alpha * delta * obs
                else:
                    #更新Q2
                    Q2_act = np.argmax(np.dot(weight2, new_obs))
                    delta = reward + gamma * np.dot(new_obs, weight1[Q2_act]) - np.dot(obs, weight2[act])
                    weight2[act] = weight2[act] + alpha * delta * obs

                obs = new_obs
                r = r + reward

            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'),'训练局数',n)
                if info[0]>info[1]:
                    NumWin = NumWin+1
                    abs_score = info[0] - info[1]
            else:
                # java terminates unexpectedly
                pass
        if n == N:
            break

        RewardData.append(r)

    print("finish training")
    print("获胜局数",NumWin)
