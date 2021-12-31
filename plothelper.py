class PlotHelper:
    def __init__(self, ny, nx, grid=True):
        self.fig = plt.figure(figsize=(4 * nx,3 * ny))
        self.axes = []
        count = 1
        for iy in range(ny):
            for ix in range(nx):
                self.axes.append(plt.subplot(ny, nx, count))
                self.axes[-1].grid()
                count += 1
    
    def plot(self, axes, data, title=None):
        self.axes[axes].plot(data)
        if title is not None:
            self.axes[axes].set_title(title)

    def plots(self, axes, datas, title=None):
        for data in datas:
            self.axes[axes].plot(data)
        if title is not None:
            self.axes[axes].set_title(title)
