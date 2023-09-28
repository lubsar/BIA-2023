import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from common.interval import *

import matplotlib.animation as animation

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
        
        vertex_colours = [(0,0,0) for _ in range(len(points[0]))]
        vertex_colours[0] = (0,0,1)
        vertex_colours[-1] = (1,0,0)

        scatter = ax.scatter(points[0], points[1], points[2], marker='o', c = vertex_colours, zorder=5)
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
            scatter.set_color(vertex_colours[:frame + 1])
            
        self.anim = animation.FuncAnimation(self.fig, animFunction, len(points[0]), fargs=(points, scatter), interval=200)

    def show(self):
        plt.show(block=True)

    def cleanup(self):
        plt.close()
        plt.clf()