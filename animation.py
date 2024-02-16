# プロットを設定
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-15, -15+60)
ax.set_ylim(-30, -30+60)
ax.set_zlim(-25, -25+60)

# ラベルをプロットする関数
def plot_labels(x, y, z, labels):
    for (xi, yi, zi, label) in zip(x, y, z, labels):
        ax.text(xi, yi, zi, label)

# アニメーションの更新関数
def update(frame):
    ax.view_init(30, frame)
    return fig,

# アニメーションの初期化関数
X1 = X_
X2 = points_mean
T1 = np.arange(len(X_))
T2 = points_indices
def init():
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], s=1, color='b', label='X1')
    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], s=1, color='r', label='X2')
    plot_labels(X1[:, 0], X1[:, 1], X1[:, 2], T1)
    plot_labels(X2[:, 0], X2[:, 1], X2[:, 2], T2)
    return fig,

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), init_func=init, blit=False)

# ani.save('3d_animation.mp4', writer='ffmpeg')
ani.save('3d_animation.mp4', writer='ffmpeg', fps=20, extra_args=['-vcodec', 'mpeg4', '-q:v', '5'])

# Jupyter Labでアニメーションを表示
HTML(ani.to_jshtml())
