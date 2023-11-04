import numpy as np
import copy

from common.visualization import *
from common.functions import*

class SomaA2O:
    def init(self, seed, function : TestFunction):
        self.seed = seed
        self.function = function

    def createPopulationVisual(self, population):
        X = []
        Y = []
        Z = []

        for individual in population:
            X.append(individual[0])
            Y.append(individual[1])
            Z.append(individual[2])

        return (X, Y, Z)

    def search(self, popSize : int, PRT : float, pathLength: float, step : float, M_max :int) -> list:
        self.function.setSamplingSeed(self.seed)

        generations = []
        population = self.function.randomSamples(popSize)
        generations.append(self.createPopulationVisual(population))

        leader = population[0]
        for individual in population:
            if individual[2] < leader[2]:
                leader = copy.copy(individual)

        for _ in range(M_max):
            new_population = []
            new_leader = leader
            for  individual in population:
                best_individual = individual
                new_individual = [0] * 3

                for t in np.arange(0.0, pathLength, step):
                    for i in range(2):
                      new_individual[i] = individual[i] + (leader[i] - individual[i]) * t * (1.0 if np.random.rand() < PRT else 0.0)

                    clamped = self.function.preserveBoundsSetAtBorder(new_individual[:-1])

                    new_individual[0] = clamped[0]
                    new_individual[1] = clamped[1]

                    individual_cost = self.function.calculate(new_individual[:-1])
                    new_individual[2] = individual_cost

                    if(individual_cost < best_individual[2]):
                        best_individual = copy.copy(new_individual)

                if best_individual[2] < new_leader[2]:
                    new_leader = copy.copy(best_individual)

                new_population.append(best_individual)

            population = new_population
            leader = new_leader
            generations.append(self.createPopulationVisual(population))

        return generations