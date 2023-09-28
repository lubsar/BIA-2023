import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from common.interval import *

import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)


def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk

def sphere(params):
        sum = 0
        for p in params:
            sum += p**2
        
        return sum

def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines


# Data: 40 random walks as (num_steps, 3) arrays
num_steps = 30
walks = [random_walk(num_steps) for index in range(40)]


class Visualisation3D:
    def __init__(self, antialiasing = False) -> None:
         self.antialiasing = antialiasing
         self.fig = plt.figure()

    def plotSurface(self, viewport : Interval3D, surface):
        axes = self.fig.axes
        ax = None
        if len(axes) < 1:
            ax = self.fig.add_subplot(projection="3d", computed_zorder=False)
        elif len(axes) == 1:
            ax = axes[0]
        else:
            raise RuntimeError("Wrong number of axess")

        startX, endX, stepX = viewport.getXInteval()
        startY, endY, stepY = viewport.getYInteval()
        startZ, endZ, stepZ = viewport.getZInteval()

        ax.set(xlim3d=(startX, endX), xlabel='X')
        ax.set(ylim3d=(startY, endY), ylabel='Y')
        ax.set(zlim3d=(startZ, endZ), zlabel='Z')

        ax.plot_surface(*(surface), cmap=cm.coolwarm, linewidth=0, antialiased=self.antialiasing, zorder=0)
     
    def plot3DFunction(self, viewport : Interval3D, mesh_grid : Interval2D, mesh_function):
        X = np.arange(*mesh_grid.getXInteval())
        Y = np.arange(*mesh_grid.getYInteval())
        Z = np.array([[mesh_function((x, y)) for x in X] for y in Y])
        
        X, Y = np.meshgrid(X, Y)

        self.plotSurface(viewport, (X, Y, Z))
    
    def plotPointsAnimation(self, points, labels = None):
        axes = self.fig.axes
        ax = None
        if len(axes) < 1:
            ax = self.fig.add_subplot(projection="3d", computed_zorder=False)
        elif len(axes) == 1:
            ax = axes[0]
        else:
            raise RuntimeError("Wrong number of axess")
        
        scatter = ax.scatter(points[0], points[1], points[2], marker='o', c = "red", zorder=5)
        texts = []

        def animFunction(frame, points, scatter):
            if labels is not None:
                if frame == 0:
                    for text in texts:
                        text.remove()

                    texts.clear()
                
                texts.append(ax.text(points[0][frame], points[1][frame], points[2][frame], labels[frame], size=12, zorder=10, color="k"))

            X = points[0][:frame + 1]
            Y = points[1][:frame + 1]
            Z = points[2][:frame + 1]

            scatter._offsets3d = (X, Y, Z)
            
        self.anim = animation.FuncAnimation(self.fig, animFunction, len(points[0]), fargs=(points, scatter), interval=200)

    def show(self):
        plt.show(block=True)

    def cleanup(self):
        plt.close()
        plt.clf()