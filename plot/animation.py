import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def plot_ani_2d_all(df_list, label, x_ax, y_ax, buffer=50):
    nodeids = part2nodeid[label]
    df_mask_list = [df[df["NODE_ID"].isin(nodeids)] for df in df_list]
    
    # プロットオブジェクトを作成
    fig, ax = plt.subplots()
    
    times = df_mask_list[0]["Time"].unique()
    time = times[0]
    scatter_list = []
    x_min_list, x_max_list = [], []
    y_min_list, y_max_list = [], []
    for df_mask in df_mask_list:
        mask = df_mask["Time"] == time
        scatter = ax.scatter(df_mask[mask][x_ax], df_mask[mask][y_ax], s=5)
        scatter_list.append(scatter)
        x_min_list.append(df_mask[x_ax].min())
        x_max_list.append(df_mask[x_ax].max())
        y_min_list.append(df_mask[y_ax].min())
        y_max_list.append(df_mask[y_ax].max())
    x_min, x_max = min(x_min_list), max(x_max_list)
    y_min, y_max = min(y_min_list), max(y_max_list)
    
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    ax.grid()
    ax.set_aspect("equal")
    ax.set_xlabel(x_ax)
    ax.set_ylabel(y_ax)
    
    # アニメーションフレームごとに呼び出される関数
    def update(time):
        plt.title(time)

        for scatter, df_mask in zip(scatter_list, df_mask_list):
            mask = df_mask["Time"] == time
            # scatterプロットを更新
            scatter.set_offsets(np.c_[df_mask[mask][x_ax], df_mask[mask][y_ax]])        
        
        # グラフを再描画
        fig.canvas.draw()
    
    # アニメーションを作成
    animation = FuncAnimation(fig, update, frames=times, interval=200)
    return animation

cls = 0
label = "CTR PLR"
x_ax, y_ax = "y", "z"
animation = plot_ani_2d_all(df_list_cluster[cls], label, x_ax, y_ax)
# animation.save(out_path / f"{label}_{cls}.mp4", writer="ffmpeg", dpi=300)
# 動画の保存(ffmpeg -encodersでコーデック一覧を確認できる)
animation.save("test.mp4", writer='ffmpeg', extra_args=['-vcodec', 'mpeg4'])
# アニメーションをHTML形式に変換（Jupyter上）
HTML(animation.to_jshtml())

