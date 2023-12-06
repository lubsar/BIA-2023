import numpy as np
import copy

from common.visualization import *
from common.functions import*


class Firefly:
    def __init__(self, position : tuple[float, ...], evaluation : float, function : TestFunction) -> None:
        self.position = position
        self.brightness = 1.0 / evaluation
        self.evaluation = evaluation
        self.function = function

    def setEvaluation(self, evaluation : float) -> None:
        self.evaluation = evaluation
        self.brightness = 1.0 / evaluation

    def distance(self, other):
        return np.sqrt(np.sum(np.power(np.subtract(self.position, other.position), 2.0)))

    def moveRandom(self):
        num_dims = len(self.position)

        random_movement = np.clip(np.random.normal(0.0, 0.3, num_dims), (-1.0) * num_dims, (1.0) * num_dims)
        new_position = self.function.preserveBoundsSetAtBorder(np.add(self.position, random_movement))
        eval = self.function.calculate(new_position)

        if eval < self.evaluation:
            self.position = new_position
            self.evaluation = eval

    def moveToward(self, other):
        if other == self:
            return
        
        num_dims = len(self.position)

        random_movement = np.clip(np.random.normal(0.0, 0.3, num_dims), (-1.0) * num_dims, (1.0) * num_dims)
            
        if self.evaluation > other.evaluation:
            attractiveness = 1 / (1 + self.distance(other))
            attract = np.add(self.position, np.multiply(np.subtract(other.position, self.position), attractiveness))
            new_position = self.function.preserveBoundsSetAtBorder(np.add(attract, random_movement))
        else:
            new_position = self.function.preserveBoundsSetAtBorder(np.add(self.position, random_movement))

        self.position = new_position
        #self.reevaluate()

    def reevaluate(self):
        self.evaluation = self.function.calculate(self.position)
        self.brightness = 1.0 / self.evaluation

class FireflySwarmp:
    def init(self, seed, function : TestFunction):
        self.seed = seed
        self.function = function

    def createGenData(self, population : list[Firefly]):
        data = []

        for individual in population:
            data.append((*individual.position, individual.evaluation))
    
        return data

    def search(self, popSize : int, M_max : int, max_evaluations : int) -> list:
        self.function.setSamplingSeed(self.seed)

        num_evaluations = popSize
        generations = []
        population = [Firefly(sample[:-1], sample[-1], self.function) for sample in self.function.randomSamples(popSize)]
        generations.append(self.createGenData(population))

        leader = population[0]
        for individual in population:
            if individual.evaluation < leader.evaluation:
                leader = individual

        for p in range(M_max):
            for individual1 in population:
                if individual1.evaluation == leader.evaluation:
                    individual1.moveRandom()
                    num_evaluations += 1
                    continue

                for individual2 in population:
                    individual1.moveToward(individual2)
                    #num_evaluations += 1
            
            population[0].reevaluate()
            new_leader = population[0]
            num_evaluations += 1

            for individual in population:
                individual.reevaluate()
                num_evaluations += 1
                if individual.evaluation < new_leader.evaluation:
                    new_leader = individual

            leader = new_leader
            if num_evaluations > max_evaluations:
                return generations

            generations.append(self.createGenData(population))

        return generations