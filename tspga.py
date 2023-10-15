import math
import numpy as np
import random
import copy

class DistanceMatrix:
    def __init__(self, points) -> None:
        self.num_points = len(points)
        self.dimensions = len(points[0])
        self.points = points

        self.distances = [[0.0 for _ in range(self.num_points)] for _ in range(self.num_points)]

        def euclidean_distance(point1, point2) -> float:
            square_sum = 0.0
            for i in range(len(point1)):
                square_sum += (point1[i] - point2[i]) ** 2 

            return math.sqrt(square_sum)

        for i in range(len(points)):
            for o in range(i + 1, len(points)):
                distance = euclidean_distance(points[i], points[o])

                self.distances[i][o] = distance
                self.distances[o][i] = distance
    
    def evaluatePath(self, indicies : list[int]) -> float:
        result = 0.0

        for i in range(len(indicies) - 1):
            result += self.distances[indicies[i]][indicies[i + 1]]

        return result


class Individual:
    def __init__(self, genes : list[int]) -> None:
        self.genes = genes.copy()
        self.path = [0] + genes + [0]
        self.pathLength = None

    def createOffspring(self, other):
        num_first_parent_genes = int(len(self.genes) / 2)

        first_parent_genes = self.genes[:num_first_parent_genes]
        second_parent_genes = other.genes[num_first_parent_genes:]

        if len(first_parent_genes) + len(second_parent_genes) != len(self.genes):
            raise RuntimeError("Wrong resulting genome")

        second_parent_needed_genes = set(self.genes) - set(first_parent_genes)

        new_genes = first_parent_genes + second_parent_genes
        to_replace_indicies = []

        for i in range(num_first_parent_genes, len(self.genes)):
            if new_genes[i] not in second_parent_needed_genes:
                to_replace_indicies.append(i)
            else:
                second_parent_needed_genes.remove(new_genes[i])

            leftover_genes = list(second_parent_needed_genes)
        assert(len(second_parent_needed_genes) == len(to_replace_indicies))

        for o in range(len(to_replace_indicies)):
            new_genes[to_replace_indicies[o]] = leftover_genes[o]
        
        assert(len(set(new_genes)) == len(self.genes))

        return Individual(new_genes)

    def mutate(self) -> None:
        first_gene_index, second_gene_index = random.randint(0, len(self.genes) -1), random.randint(0, len(self.genes) -1)

        while first_gene_index == second_gene_index:
            second_gene_index = random.randint(0, len(self.genes) -1)

        tmp = self.genes[first_gene_index]
        self.genes[first_gene_index] = self.genes[second_gene_index]
        self.genes[second_gene_index] = tmp

        self.path = [0] + self.genes + [0]
        self.pathLength = None

    def getPath(self) -> list[int]:
        return self.path

    def calculatePathLength(self, distance_matrix : DistanceMatrix) -> float:
        self.pathLength = distance_matrix.evaluatePath(self.path)

        return self.pathLength

    def getPathLength(self) -> float:
        if self.pathLength is None:
            raise RuntimeError("Path length was not calculated")
        
        return self.pathLength

class TSPGA:
    def __init__(self, points : list[tuple[float, float]], ) -> None:
        self.points = points
        self.distance_matrix = DistanceMatrix(points)

    def randomPopulation(self, num_individuals: int) -> list[Individual]:
        indicies = [x for x in range(1, len(self.points))]
        population = []

        for _ in range(num_individuals):
            np.random.shuffle(indicies)
            population.append(Individual(indicies))

        return population

    def run(self, num_generations : int, num_individuals : int, seed : int) -> tuple[list, float]:
        np.random.seed(seed)
        random.seed(seed)

        gen_data = []

        population = self.randomPopulation(num_individuals)
        for individual in population:
            individual.calculatePathLength(self.distance_matrix)

        for gen in range(num_generations):
            new_population = copy.deepcopy(population)

            for i in range(num_individuals):
                first_parent = population[i]

                second_parent_index = random.randint(0, num_individuals -1)
                while(second_parent_index) == i:
                    second_parent_index = random.randint(0, num_individuals -1)

                new_individual = first_parent.createOffspring(population[second_parent_index])

                if np.random.uniform() < 0.5:
                    new_individual.mutate()

                new_individual_path_length = new_individual.calculatePathLength(self.distance_matrix)
                parent_path_length = first_parent.getPathLength()

                if  new_individual_path_length < parent_path_length:
                    new_population[i] = new_individual

            population = new_population

            mindist = population[0].getPathLength()
            maxdist = mindist
            for i in range(1, num_individuals):
                mindist = min(mindist, population[i].getPathLength())
                maxdist = max(maxdist, population[i].getPathLength())

            print(gen, mindist, maxdist)
            gen_data.append({"min": mindist, "max": maxdist})

        return population, gen_data