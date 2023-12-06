import numpy as np
import copy

from common.visualization import *
from common.functions import*

class SomaA2O:
    def __init__(self, PRT : float, pathLength: float, step : float) -> None:
        self.PRT = PRT
        self.pathLength = pathLength
        self.step = step

    def init(self, seed, function : TestFunction):
        self.seed = seed
        self.function = function

    def search(self, popSize : int, M_max :int, max_evaluations : int) -> list:
        self.function.setSamplingSeed(self.seed)

        generations = []
        population = self.function.randomSamples(popSize)
        generations.append(population)

        num_evaluations = popSize

        leader = population[0]
        for individual in population:
            if individual[-1] < leader[-1]:
                leader = copy.copy(individual)

        for _ in range(M_max):
            new_population = []
            new_leader = leader
            for  individual in population:
                best_individual = individual
                new_individual = [0] * len(individual)

                for t in np.arange(0.0, self.pathLength, self.step):
                    for i in range(len(self.function.bounds)):
                      new_individual[i] = individual[i] + (leader[i] - individual[i]) * t * (1.0 if np.random.rand() < self.PRT else 0.0)

                    clamped = self.function.preserveBoundsSetAtBorder(new_individual[:-1])
                    new_individual = list(clamped) + [self.function.calculate(clamped)]
                    num_evaluations += 1

                    if(new_individual[-1] < best_individual[-1]):
                        best_individual = copy.copy(new_individual)

                if best_individual[-1] < new_leader[-1]:
                    new_leader = copy.copy(best_individual)

                if(num_evaluations > max_evaluations):
                    return generations

                new_population.append(best_individual)

            population = new_population
            leader = new_leader

            generations.append(population)

        return generations