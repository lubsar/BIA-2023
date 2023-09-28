import numpy as np
import random
import datetime

from common.visualization import *
from common.functions import*

class BlindSearch3D:
    def __init__(self, seed, function : TestFunction):
        self.seed = seed
        self.function = function

    def sample(self, num_samples):
        xMin, xMax = self.function.getXBounds()
        yMin, yMax = self.function.getYBounds()

        points = [(random.random() * (xMax - xMin) + xMin, random.random() * (yMax - yMin) + yMin,) for _ in range(num_samples)]
        samples = [(x[0], x[1], self.function.calculate((x[0], x[1]))) for x in points]

        return samples

    def search(self, num_samples, num_generations):
        random.seed(self.seed)

        best_solutions = []
        best_solutions_labels = []

        #1) first sample
        best_solution = self.sample(1)[0]
        best_solutions.append(best_solution)
        best_solutions_labels.append("gen 0")

        #2) generations
        for gen_number in range(1, num_generations + 1):
            samples = self.sample(num_samples)

            # select first sample
            best_gen_solution = samples[0]

            # compare with rest of samples
            for sample in samples[1:]:
                if sample[2] < best_gen_solution[0]:
                    best_gen_solution = sample

            # compare gen's best with overall best
            if best_gen_solution[2] < best_solution[2]:
                best_solution = best_gen_solution
                best_solutions.append(best_solution)
                best_solutions_labels.append("gen {0}".format(gen_number))

        resultX = []
        resultY = []
        resultZ = []

        for point in best_solutions:
            resultX.append(point[0])
            resultY.append(point[1])
            resultZ.append(point[2])

        return (resultX, resultY, resultZ, best_solutions_labels) 


def test_blindSearch(seed, num_generations : int, num_samples : int, visual_resolution : int):
    print("current seed:", seed)

    functions = Functions()
    for _, function in functions.__dict__.items():
        algo = BlindSearch3D(seed, function)
        visual = Visualisation3D()

        X,Y,Z,labels = algo.search(num_samples, num_generations)

        print("best: ", (labels[-1], X[-1], Y[-1], Z[-1]))

        visual.plot3DFunction(function.viewPort, function.meshInterval(visual_resolution), function.calculate)
        visual.plotPointsAnimation((X, Y, Z), labels)

        visual.show()


seed = datetime.datetime.now().timestamp()
num_generations = 100
num_samples = 100
visual_mesh_resolution = 100

test_blindSearch(seed, num_generations, num_samples, visual_mesh_resolution)