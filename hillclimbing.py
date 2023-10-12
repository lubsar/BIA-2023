import numpy as np
import random

from common.visualization import *
from common.functions import*

class HillClimbingSearch3D:
    def init(self, seed, function : TestFunction, deviation : float):
        self.seed = seed
        self.function = function
        self.sigma = deviation

    def search(self, num_samples, num_generations):
        self.function.setSamplingSeed(self.seed)

        best_solutions = []
        best_solutions_labels = []

        #1) first sample
        best_solution = self.function.randomSample()
        best_solutions.append(best_solution)
        best_solutions_labels.append("gen 0")

        #2) generations
        for gen_number in range(1, num_generations + 1):
            samples = self.function.normalSamples(best_solution[:2], self.sigma, num_samples)

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