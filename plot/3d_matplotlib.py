from mpl_toolkits.mplot3d import Axes3D

def plot3d(data, title=None):
    x, y, z = data[:,0], data[:,1], data[:,2]
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z, s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if title is None:
        ax.set_title(title)
    ax.view_init(180, 50)
    plt.show()
