'''
2020/5/31
Tong Ru
train —— traditional Q-learning
'''

import random
import numpy as np
from fightingice_env import FightingiceEnv
import scipy.io as io

def loadweight(weight_file=None):
    weight= io.loadmat(weight_file)#训练好的权重进行测试
    weight=weight['w'] #[144*40,1]
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
    weight=np.zeros((40,144))
    weight = loadweight("./trainedWeight/w1686-QL")

    n=0
    p=0
    RewardData=[]
    N=500#训练次数
    NumWin=0
    Best_Score = 0
    abs_score = 0


    act = random.randint(0, 39)#初始动作

    A=np.zeros((40,144*40))
    for i in range(0,40):
        A[i,i*144:(i+1)*144]=1

    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None
        n=n+1#局数+1
        r=0

        while not done:
            if n<N:
                if np.mod(n,5)==0:
                    string="./trainLog/QLtrainLog/weight"+str(n+p)
                    io.savemat(string, {'weight': weight})
                    string="./trainLog/QLtrainLog/reward"+str(n+p)
                    io.savemat(string, {'r': RewardData})
                if Best_Score <= abs_score:
                    Best_Score = abs_score
                    string="./trainLog/QLtrainLog/weight_best"
                    io.savemat(string, {'weight': weight})
                    string="./trainLog/QLtrainLog/reward_best"
                    io.savemat(string, {'r': RewardData})
            else:
                print("训练结束")
                string="./trainLog/QLtrainLog/weight"+str(n+p)
                io.savemat(string, {'weight': weight})
                string="./trainLog/QLtrainLog/reward"+str(n+p)
                io.savemat(string, {'r': RewardData})
                break

            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            # new_obs, reward, done, info = env.step(act)
            '''
            X=np.tile(obs,(40,40))
            X=X*A
            X=X / np.array([X.max(axis=1)]).T
            act=np.argmax(np.dot(X,weight))
            '''
            act = np.argmax(np.dot(weight, obs))
            if random.random()<epsilon:
                act = random.randint(0, 39)
            else:
                pass
            '''
            x=np.zeros((1,144*40))
            x[0,144*act:144*(act+1)]=np.array([obs])
            '''
            new_obs, reward, done, info = env.step(act)

            if not done:
                # TODO: (main part) learn with data (obs, act, reward, new_obs)
                '''
                X=np.tile(new_obs,(40,40))
                X=X*A
                X=X / np.array([X.max(axis=1)]).T
                '''
                delta=reward + gamma * np.max(np.dot(weight,new_obs))-np.dot(obs,weight[act])
                #weight=weight+ alpha*delta[0]* x.T
                weight[act] = weight[act] + alpha*delta*obs

                obs=new_obs
                r=r+reward

            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'),'训练局数',n)
                if info[0]>info[1]:
                    NumWin=NumWin+1
                    abs_score = info[0] - info[1]
            else:
                # java terminates unexpectedly
                pass
        if n==N:
            break

        RewardData.append(r)

    print("finish training")
    print("获胜局数",NumWin)
