import numpy as np
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

Q = np.load('SARSAoptimal.npy')
#Q[Q==0] = -2000
Q = -Q

Q_0 = Q[0, :, :]
Q_1 = Q[1, :, :]
Q_2 = Q[2, :, :]

def action(x,y):
    return np.argmax(Q[:,x,y])

i = 0
j = 0
Q_optimal = np.zeros([200,200])
while i < 200:
    while j < 200:
        Q_optimal[i, j] = action(i,j)
        if Q[0,i,j] == Q[1,i,j] and Q[1,i,j] == Q[2,i,j]:
            Q_optimal[i,j] = -1
        j += 1
    j = 0
    i += 1

Q_shape = Q.shape
print(Q_shape)
print(Q_0.shape)


fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
X = np.arange(0,200,1)
Y = np.arange(0,200,1)
X,Y = np.meshgrid(X,Y)

surf = ax.plot_surface(X,Y,Q_0[X,Y],rstride=1,cstride=1,cmap='rainbow')
levels = np.arange(200, 2200, 200) 
ax.contour(X,Y,Q_0[X,Y], levels, offset=-1000, cmap = 'rainbow')
plt.title("Qvalue-----action=-3V")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel('diff_alpha rad/s')
plt.xlabel('alpha rad')
ax.set_zlabel('Qvalue')
plt.xticks(np.arange(0, 200.01, 50), ['-pi', '-1/2*pi', 0, '1/2*pi', 'pi'])
plt.yticks([0, 33, 66, 99, 133, 167, 200], ['-15pi', '-10*pi', '-5*pi',0, '5*pi', '10*pi', '15pi'])
ax.set_zlim(-1000, 1600)
plt.savefig('SARSA_0.png')

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
X = np.arange(0,200,1)
Y = np.arange(0,200,1)
X,Y = np.meshgrid(X,Y)

surf = ax.plot_surface(X,Y,Q_1[X,Y],rstride=1,cstride=1,cmap='rainbow')
levels = np.arange(200, 2200, 200) 
ax.contour(X,Y,Q_0[X,Y], levels, offset=-1000, cmap = 'rainbow')
plt.title("Qvalue-----action=0V")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel('diff_alpha rad/s')
plt.xlabel('alpha rad')
ax.set_zlabel('Qvalue')
plt.xticks(np.arange(0, 200.01, 50), ['-pi', '-1/2*pi', 0, '1/2*pi', 'pi'])
plt.yticks([0, 33, 66, 99, 133, 167, 200], ['-15pi', '-10*pi', '-5*pi',0, '5*pi', '10*pi', '15pi'])
ax.set_zlim(-1000, 1600)
plt.savefig('SARSA_1.png')

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
X = np.arange(0,200,1)
Y = np.arange(0,200,1)
X,Y = np.meshgrid(X,Y)

surf = ax.plot_surface(X,Y,Q_2[X,Y],rstride=1,cstride=1,cmap='rainbow')
levels = np.arange(200, 2200, 200) 
ax.contour(X,Y,Q_0[X,Y], levels, offset=-1000, cmap = 'rainbow')
plt.title("Qvalue-----action=3V")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel('diff_alpha rad/s')
plt.xlabel('alpha rad')
ax.set_zlabel('Qvalue')
plt.xticks(np.arange(0, 200.01, 50), ['-pi', '-1/2*pi', 0, '1/2*pi', 'pi'])
plt.yticks([0, 33, 66, 99, 133, 167, 200], ['-15pi', '-10*pi', '-5*pi',0, '5*pi', '10*pi', '15pi'])
ax.set_zlim(-1000, 1600)
plt.savefig('SARSA_2.png')


fig = plt.figure(figsize=(16,8))
colormap = colors.ListedColormap(["black","darkblue","lightblue","yellow"])
plt.imshow(Q_optimal,cmap=colormap)
plt.yticks(np.arange(0, 200.01, 50), ['-pi', '-1/2*pi', 0, '1/2*pi', 'pi'])
plt.xticks([0, 33, 66, 99, 133, 167, 200], ['-15pi', '-10*pi', '-5*pi',0, '5*pi', '10*pi', '15pi'])
patches = [ mpatches.Patch(color='black', label="random"), mpatches.Patch(color='darkblue', label="-3V"), mpatches.Patch(color='lightblue', label="0V"), mpatches.Patch(color='yellow', label="3V")]
plt.xlabel('diff_alpha rad/s')
plt.ylabel('alpha rad')
plt.title('Action choice under state(alpha, diff_alpha)')
plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0. )
plt.savefig('SARSA_action.png')
