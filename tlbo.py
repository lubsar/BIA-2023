import numpy as np
import copy

from common.visualization import *
from common.functions import*

class TLBO:
    def init(self, seed, function : TestFunction):
        self.seed = seed
        self.function = function

    def search(self, popSize : int, num_generations : int, max_evaluations : int) -> list:
        self.function.setSamplingSeed(self.seed)

        generations = []
        population = self.function.randomSamples(popSize)
        generations.append(population)

        num_evaluations = popSize

        for _ in range(num_generations):
            teacher = copy.copy(population[0])
            teacher_index = 0
            for index, individual in enumerate(population):
                if individual[-1] < teacher[-1]:
                    teacher = copy.copy(individual)
                    teacher_index = index

            population_mean = np.mean(population, axis=0)

            new_population = [teacher]
            
            for index, individual in enumerate(population):
                if index != teacher_index:
                    new_individual = np.random.uniform() * (teacher - np.random.randint(1, 3) * population_mean) 
                    new_individual = self.function.preserveBoundsLoopAround(new_individual[:-1])
                    new_individual = (*new_individual, self.function.calculate(new_individual))

                    num_evaluations += 1
                    
                    if new_individual[-1] < individual[-1]:
                        new_population.append(new_individual)
                    else:
                        new_population.append(individual)

            for index, individual in enumerate(new_population):
                other_student_index = index
                while other_student_index == index:
                    other_student_index = np.random.randint(0, len(population))

                other_student = new_population[other_student_index]

                difference = np.random.uniform() * np.subtract(individual, other_student)
                if individual[-1] > other_student[-1]:
                    difference = np.multiply(-1, difference)
                
                new_individual = np.add(individual, difference)
                new_individual = self.function.preserveBoundsLoopAround(new_individual[:-1])
                new_individual = (*new_individual, self.function.calculate(new_individual))

                num_evaluations += 1

                if new_individual[-1] < individual[-1]:
                    new_population[index] = new_individual

            population = new_population
            if num_evaluations > max_evaluations:
                return generations

            generations.append(population)

        return generations