import numpy as np
import random
import datetime

from common.visualization import *
from common.functions import*

class BlindSearch3D:
    def init(self, seed, function : TestFunction, sigma):
        self.seed = seed
        self.function = function

    def search(self, num_samples : int, num_generations : int):
        self.function.setSamplingSeed(self.seed)

        best_solutions = []
        best_solutions_labels = []

        #1) first sample
        best_solution = self.function.randomSample()
        best_solutions.append(best_solution)
        best_solutions_labels.append("gen 0")

        #2) generations
        for gen_number in range(1, num_generations + 1):
            samples = self.function.randomSamples(num_samples)

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