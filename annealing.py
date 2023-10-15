import numpy as np
import math

from common.visualization import *
from common.functions import*

class SimulatedAnnealing3D:
    def init(self, seed, function : TestFunction, deviation : float) -> None:
        self.seed = seed
        self.function = function
        self.sigma = deviation

    def search(self, initial_temperature : float, min_temperature : float, cooling_factor : float):
        self.function.setSamplingSeed(self.seed)

        best_solutions = []
        best_solutions_labels = []

        best_solution = self.function.randomSample()
        best_solutions.append(best_solution)
        best_solutions_labels.append("gen 0")

        temperature = initial_temperature

        while temperature > min_temperature:
            sample = self.function.normalSample(best_solution[:2], self.sigma)

            if sample[2] < best_solution[2]:
                best_solution = sample
                best_solutions.append(best_solution)
                best_solutions_labels.append("Temperature {0}".format(temperature))
            else :
                r = np.random.uniform(0.0, 1.0)
                magic_value = math.e ** ((-1 * (sample[2] - best_solution[2])) / temperature)

                if r < magic_value:
                    best_solution = sample

            temperature *= cooling_factor

        resultX = []
        resultY = []
        resultZ = []

        for point in best_solutions:
            resultX.append(point[0])
            resultY.append(point[1])
            resultZ.append(point[2])

        return (resultX, resultY, resultZ, best_solutions_labels) 

