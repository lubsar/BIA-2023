import numpy as np
import copy

from common.visualization import *
from common.functions import*

class DifferentialEvolution3D:
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
            Z.append(self.function.calculate(individual))

        return (X, Y, Z)

    def search(self, NP : int, G_maxim : int, F : float, CR : float):
        self.function.setSamplingSeed(self.seed)
        num_dimensions = 2
        num_parents = 3

        generations = []
        population = self.function.randomPoints(NP)
        generations.append(self.createPopulationVisual(population))

        for _ in range(G_maxim):
            new_population = copy.deepcopy(population)

            for individual_index, individual in enumerate(population):
                parents_indicies = []

                for _ in range(num_parents):
                    index = random.randint(0, len(population) - 1)

                    while index in parents_indicies or index == individual_index:
                        index = random.randint(0, len(population) -1)

                    parents_indicies.append(index)

                parents = [population[x] for x in parents_indicies]

                mutation_vector = np.add(F * np.subtract(parents[0], parents[1]), parents[2])
                trial_vector = [0] * num_dimensions

                j_rnd = random.randint(0, num_dimensions - 1)

                for j in range(num_dimensions):
                    if np.random.uniform() < CR or j == j_rnd:
                        trial_vector[j] = mutation_vector[j]
                    else :
                        trial_vector[j] = individual[j]

                trial_vector = self.function.preserveBounds(trial_vector)
                f_u = self.function.calculate(trial_vector)

                if f_u <= self.function.calculate(individual):
                    new_population[individual_index] = trial_vector

            population = new_population
            generations.append(self.createPopulationVisual(population))

        return generations