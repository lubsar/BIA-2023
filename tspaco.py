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
    
    def at(self, row: int, column : int):
        return self.distances[row][column]

    def evaluatePath(self, indicies : list[int]) -> float:
        result = 0.0

        for i in range(len(indicies) - 1):
            result += self.distances[indicies[i]][indicies[i + 1]]

        return result


class VisibilityMatrix:
    def __init__(self, distance_matrix : DistanceMatrix) -> None:
        self.num_points = distance_matrix.num_points
        self.visiblities = [[0.0 for _ in range(self.num_points)] for _ in range(self.num_points)]
        self.hidden = set()

        for i in range(self.num_points):
            for o in range(i + 1, self.num_points):
                visiblity = 1.0 / distance_matrix.at(i,o)

                self.visiblities[i][o] = visiblity
                self.visiblities[o][i] = visiblity

    def hideColumn(self, column_index : int):
        self.hidden.add(column_index)

    def unhideColumns(self):
        self.hidden = set()

    def at(self, row : int, column : int):
        return 0.0 if column in self.hidden else self.visiblities[row][column]
    
    def visibleCities(self):
        all_columns = set(range(0, self.num_points))

        return all_columns - self.hidden
    
class PheromoneMatrix:
    def __init__(self, num_points : int) -> None:
        self.num_points = num_points
        self.pheromones = [[1.0 for _ in range(self.num_points)] for _ in range(self.num_points)]
        
        for i in range(self.num_points):
            for o in range(i + 1, self.num_points):
                self.pheromones[i][o] = 1.0
                self.pheromones[o][i] = 1.0

    def evaporate(self, evaporation : float):
        for i in range(self.num_points):
            for o in range(i, self.num_points):
                new_value = self.pheromones[i][o] * (1.0 - evaporation)

                self.pheromones[i][o] = new_value
                self.pheromones[o][i] = new_value

    def deposit(self, path : list[int], value : float):
        prev_node = path[0]
        for next_node in path[1:]:
            self.pheromones[prev_node][next_node] += value
            self.pheromones[next_node][prev_node] += value

            prev_node = next_node

    def at(self, row, column):
        return self.pheromones[row][column]

class TSPACO:
    def __init__(self, points : list[tuple[float, float]], pheromone_importance : float, visibility_importance : float) -> None:
        self.points = points
        self.distance_matrix = DistanceMatrix(points)
        self.visibility_matrix = VisibilityMatrix(self.distance_matrix)
        self.pheromone_matrix = PheromoneMatrix(self.distance_matrix.num_points)
        
        self.pheromone_importance = pheromone_importance
        self.visibility_importance = visibility_importance

    def nextAntCity(self, current_city : int):
        next_city_probabilities = []
        visible_cities = self.visibility_matrix.visibleCities()

        for city in visible_cities:
            city_pheromone = self.pheromone_matrix.at(current_city, city) ** self.pheromone_importance
            city_visibility =  self.visibility_matrix.at(current_city, city) ** self.visibility_importance
            
            city_probability = city_pheromone * city_visibility
            next_city_probabilities.append((city, city_probability))

        probability_sum = sum([prob[1] for prob in next_city_probabilities])
        cummulative = 0.0
        for i in range(len(next_city_probabilities)):
            city, probability = next_city_probabilities[i]

            probability = probability / probability_sum
            cummulative += probability

            next_city_probabilities[i] = (city, cummulative)

        city_selector = random.random()

        next_city = None
        for index, (city, probability) in enumerate(next_city_probabilities):
            if probability >= city_selector:
                next_city = next_city_probabilities[index - 1][0]
                break

        if not next_city:
            next_city = next_city_probabilities[-1][0]

        return next_city

    def nextAnt(self):
        num_cities = self.distance_matrix.num_points
        path = [0]

        self.visibility_matrix.unhideColumns()
        self.visibility_matrix.hideColumn(0)

        for _ in range(num_cities - 1):
            next_city = self.nextAntCity(path[-1])

            path.append(next_city)
            self.visibility_matrix.hideColumn(next_city)

        path.append(0)

        return path

    def run(self, num_generations : int, num_ants : int, seed : int, evaporation : float) -> tuple[list, float]:
        np.random.seed(seed)
        random.seed(seed)

        best_path = None
        best_path_length = None

        gen_data = []
        for gen in range(num_generations):
            ant_paths = []
            
            for i in range(num_ants):
                ant_paths.append(self.nextAnt())

            self.pheromone_matrix.evaporate(evaporation)

            for path in ant_paths:
                path_length = self.distance_matrix.evaluatePath(path)

                if not best_path:
                    best_path = path
                    best_path_length = path_length
                elif path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

                self.pheromone_matrix.deposit(path, 1.0 / path_length)

            gen_data.append(best_path)

        return gen_data