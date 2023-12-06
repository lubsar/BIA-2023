import numpy as np
import copy

from common.visualization import *
from common.functions import*

class DifferentialEvolution:
    def __init__(self, F : float, CR : float) -> None:
        self.F = F
        self.CR = CR

    def init(self, seed, function : TestFunction):
        self.seed = seed
        self.function = function

    def evaluatedPopulation(self, population : list[tuple[float, ...]]):
        return [(*individual, self.function.calculate(individual)) for individual in population]

    def search(self, NP : int, G_maxim : int, max_evaluations : int) -> list:
        self.function.setSamplingSeed(self.seed)
        num_dimensions = len(self.function.bounds)
        num_parents = 3

        generations = []
        population = self.function.randomPoints(NP)
        num_evaluations = 0
        generations.append(self.evaluatedPopulation(population))

        for _ in range(G_maxim):
            if(num_evaluations >= max_evaluations):
                return generations

            new_population = copy.deepcopy(population)

            for individual_index, individual in enumerate(population):
                parents_indicies = []
                individual_f_u = self.function.calculate(individual)
                num_evaluations += 1

                for _ in range(num_parents):
                    index = random.randint(0, len(population) - 1)

                    while index in parents_indicies or index == individual_index:
                        index = random.randint(0, len(population) -1)

                    parents_indicies.append(index)

                parents = [population[x] for x in parents_indicies]

                mutation_vector = np.add(self.F * np.subtract(parents[0], parents[1]), parents[2])
                trial_vector = [0] * num_dimensions

                j_rnd = random.randint(0, num_dimensions - 1)

                for j in range(num_dimensions):
                    if np.random.uniform() < self.CR or j == j_rnd:
                        trial_vector[j] = mutation_vector[j]
                    else :
                        trial_vector[j] = individual[j]

                trial_vector = self.function.preserveBoundsLoopAround(trial_vector)
                f_u = self.function.calculate(trial_vector)
                num_evaluations += 1

                if f_u <= individual_f_u:
                    new_population[individual_index] = trial_vector
            
            if(num_evaluations > max_evaluations):
                return generations
            
            population = new_population
            generations.append(self.evaluatedPopulation(population))

        return generations