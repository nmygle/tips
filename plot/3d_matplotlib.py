from mpl_toolkits.mplot3d import Axes3D

def plot3d(data, title=None):
    x, y, z = data
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = Axes3D(fig)
    ax.scatter3D(x, y, z, s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if title is None:
        ax.set_title(title)
    ax.view_init(190, 50)
    plt.show()
