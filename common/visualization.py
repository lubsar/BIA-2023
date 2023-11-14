import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from common.interval import *

import matplotlib.animation as animation

class Visualisation:
    def __init__(self) -> None:
         self.fig = plt.figure()

    def show(self) -> None:
        plt.show(block=True)

    def cleanup(self) -> None:
        plt.close()
        plt.clf()

    def saveFig(self, path : str, file_format="pdf") -> None:
        plt.savefig(path, format=file_format) 

class Visualisation3D(Visualisation):
    def __init__(self, antialiasing = False) -> None:
        super().__init__()
        self.antialiasing = antialiasing

    def plotSurface(self, viewport : Interval3D, surface) -> None:
        axes = self.fig.axes
        ax = None
        if len(axes) < 1:
            ax = self.fig.add_subplot(projection="3d", computed_zorder=False)
        elif len(axes) == 1:
            ax = axes[0]
        else:
            raise RuntimeError("Wrong number of axess")

        startX, endX, _ = viewport.getXInteval()
        startY, endY, _ = viewport.getYInteval()
        startZ, endZ, _ = viewport.getZInteval()

        ax.set(xlim3d=(startX, endX), xlabel='X')
        ax.set(ylim3d=(startY, endY), ylabel='Y')
        ax.set(zlim3d=(startZ, endZ), zlabel='Z')

        ax.plot_surface(*(surface), cmap=cm.coolwarm, linewidth=0, antialiased=self.antialiasing, zorder=0)
     
    def plot3DFunction(self, viewport : Interval3D, mesh_grid : Interval2D, mesh_function) -> None:
        X = np.arange(*mesh_grid.getXInteval())
        Y = np.arange(*mesh_grid.getYInteval())
        Z = np.array([[mesh_function((x, y)) for x in X] for y in Y])
        
        X, Y = np.meshgrid(X, Y)

        self.plotSurface(viewport, (X, Y, Z))
    
    def plotPointsAnimation(self, points, labels = None) -> None:
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

    def plotGenerationsAnimation(self, point_sets) -> None:
        axes = self.fig.axes
        ax = None
        if len(axes) < 1:
            ax = self.fig.add_subplot(projection="3d", computed_zorder=False)
        elif len(axes) == 1:
            ax = axes[0]
        else:
            raise RuntimeError("Wrong number of axess")
        
        vertex_colours = [(0,0,0) for _ in range(len(point_sets[0][0]))]

        scatter = ax.scatter(point_sets[0][0], point_sets[0][1], point_sets[0][2], marker='o', c = vertex_colours, zorder=5)

        def animFunction(frame, points, scatter):
            X = points[frame][0]
            Y = points[frame][1]
            Z = points[frame][2]

            scatter._offsets3d = (X, Y, Z)

            ax.set_title("{0}. generation".format(frame + 1))
            
        self.anim = animation.FuncAnimation(self.fig, animFunction, len(point_sets), fargs=(point_sets, scatter), interval=200)


class Visualisation2D(Visualisation):
    def plotLine(self, values : list[float], y_label : str) -> None:
        ax = self.fig.gca()

        ax.plot([x for x in range(len(values))], values)

        ax.set(xlabel='Generation')
        ax.set(ylabel=y_label)


    def plotPoints(self, area: Interval2D, points : list) -> None:
        ax = self.fig.gca()

        startX, endX, _ = area.getXInteval()
        startY, endY, _ = area.getYInteval()

        ax.set(xlim=(startX, endX), xlabel='X')
        ax.set(ylim=(startY, endY), ylabel='Y')

        ax.scatter(points[0], points[1])
        
    def plotPath(self, area: Interval2D, points : list) -> None:
        ax = self.fig.gca()

        startX, endX, _ = area.getXInteval()
        startY, endY, _ = area.getYInteval()

        ax.set(xlim=(startX, endX), xlabel='X')
        ax.set(ylim=(startY, endY), ylabel='Y')

        ax.scatter(points[0], points[1])

        ax.plot(points[0], points[1])
        ax.plot((points[0][-1], points[0][0]), (points[1][-1], points[1][0]), c="steelblue")

        for point_index in range(len(points[0])):
            ax.text(points[0][point_index], points[1][point_index], str(point_index + 1), size=12, color="k")

    def plotPathsAnimation(self, area: Interval2D, points : list):
        ax = self.fig.gca()

        startX, endX, _ = area.getXInteval()
        startY, endY, _ = area.getYInteval()

        ax.set(xlim=(startX, endX), xlabel='X')
        ax.set(ylim=(startY, endY), ylabel='Y')

        ax.scatter(points[0][0], points[0][1])

        line, = ax.plot(points[0][0], points[0][1])
        last_segment, = ax.plot((points[0][0][-1], points[0][0][0]), (points[0][1][-1], points[0][1][0]), c="steelblue")

        # for point_index in range(len(points[0][0])):
        #     ax.text(points[0][0][point_index], points[0][1][point_index], str(point_index + 1), size=12, color="k")

        def animFunction(frame, points, line, last_segment) -> None:
            line.set_xdata(points[frame][0])
            line.set_ydata(points[frame][1])

            last_segment.set_xdata((points[frame][0][-1], points[frame][0][0]))
            last_segment.set_ydata((points[frame][1][-1], points[frame][1][0]))

            ax.set_title("gen " + str(frame))
            
        self.anim = animation.FuncAnimation(self.fig, animFunction, len(points), fargs=(points, line, last_segment), interval=200)

    def plotPointsAnimation(self, points, labels = None) -> None:
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

        def animFunction(frame, points, scatter) -> None:
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

    def show(self) -> None:
        plt.show(block=True)

    def cleanup(self) -> None:
        plt.close()
        plt.clf()