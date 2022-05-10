import random
import numpy as np
from fightingice_env import FightingiceEnv
import scipy.io as io
import sys


def loadweight(weight_file=None):
    weight= io.loadmat(weight_file)
    weight=weight['weight']
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

    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]#测试时采用此模式
    # env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]#训练时采用此模式

    weight = loadweight("./trainLog/SARSAtrainLog/weight_best")
    n=0
    Ntest=50
    Win=0

    A=np.zeros((40,144*40))
    act = random.randint(0, 39)
    for i in range(0,40):
        A[i,i*144:(i+1)*144]=1

    while True:
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None
        n=n+1#局数+1
        r=0

        while not done:
            act = np.argmax(np.dot(weight, obs))
            new_obs, reward, done, info = env.step(act)

            if not done:
                obs=new_obs
                r=r+reward

            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'),'测试局数',n)
                print('last reward',reward)
                if info[0]>info[1]:
                    Win=Win+1
            else:
                exit('done')
                pass

        if n==Ntest:
            print("WINNING TIMES", Win)
            break


    print("finish testing")
