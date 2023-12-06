import numpy as np
import copy

from common.visualization import *
from common.functions import*

class Particle:
    def __init__(self, position, initial_velocity, initial_evaluation) -> None:
        self.position = position
        self.velocity = initial_velocity
        self.evaluation = initial_evaluation
        self.bestPosition = position
        self.bestEvaluation = initial_evaluation

    def update(self, function : TestFunction) -> None:
        self.position = function.preserveBoundsSetAtBorder(np.add(self.position, self.velocity))

        self.evaluation = function.calculate(self.position)
        if self.evaluation < self.bestEvaluation:
            self.bestEvaluation = self.evaluation
            self.bestPosition = self.position
           
    def setVelocity(self, velocity) -> None:
        self.velocity = copy.copy(velocity)

    def getVelocity(self):
        return self.velocity
    
    def getBestPosition(self):
        return self.bestPosition
    
    def __repr__(self) -> str:
        return "({0},{1}) {2}".format(*self.position, self.evaluation)

class Swarm:
    def __init__(self, searchSpace : TestFunction, num_individuals : int, v_mini, v_maxi) -> None:
        velocity_difference = np.subtract(v_mini, v_maxi)

        initial_partcle_positions = searchSpace.randomPoints(num_individuals)
        initial_velocities = [np.add(v_mini, np.multiply(velocity_difference, coef)) for coef in np.random.random(num_individuals)]

        self.particles = [Particle(initial_partcle_positions[pi], initial_velocities[pi], searchSpace.calculate(initial_partcle_positions[pi]) ) for pi in range(num_individuals)]
        self.searchSpace = searchSpace
        self.vMini = v_mini
        self.vMaxi = v_maxi
        self.bestGlobalParticle = copy.deepcopy(self.particles[0])
        self.num_evaluations = num_individuals

    def findBestIndividual(self) -> Particle:
        best_particle = self.particles[0]

        for particle in self.particles:
            if best_particle.evaluation > particle.evaluation:
                best_particle = particle

        return best_particle

    def update(self, inertia_weight : float, c1 : float, c2: float):
        for particle in self.particles:
            r1 = np.random.uniform()

            inertia_velocity = np.multiply(inertia_weight, particle.getVelocity())
            position_learning = r1 * c1 * np.subtract(particle.bestPosition, particle.position)
            global_learning = r1 * c2 * np.subtract(self.bestGlobalParticle.position, particle.position)

            new_velocity = inertia_velocity +  position_learning + global_learning
            abs_velocity = np.abs(new_velocity)

            new_velocity = np.multiply(np.clip(abs_velocity, self.vMini, self.vMaxi), np.divide(new_velocity, abs_velocity))

            particle.setVelocity(new_velocity)
            particle.update(self.searchSpace)
            self.num_evaluations += 1

        gen_best = self.findBestIndividual()
        if gen_best.evaluation < self.bestGlobalParticle.evaluation:
            self.bestGlobalParticle = copy.deepcopy(gen_best)

class ParticleSwarm:
    def __init__(self, c1 : float, 
               c2 :float, vmini_coef : float, vmaxi_coef : float, ws : float, we : float) -> None:
        self.c1 = c1
        self.c2 = c2
        self.vmini_coef = vmini_coef
        self.vmaxi_coef = vmaxi_coef
        self.ws = ws
        self.we = we

    def init(self, seed, function : TestFunction):
        self.seed = seed
        self.function = function

    def createGenData(self, population : Swarm):
        data = []

        for individual in population.particles:
            position = individual.position
            data.append((*position, individual.evaluation))
            
        return data

    def search(self, pop_size : int, M_max : int, max_evaluations : int) -> list:
        self.function.setSamplingSeed(self.seed)

        vmini = np.multiply(self.vmini_coef, self.function.scales)
        vmaxi = np.multiply(self.vmaxi_coef, self.function.scales)

        generations = []
        
        population = Swarm(self.function, pop_size, vmini, vmaxi)
        generations.append(self.createGenData(population))

        for i in range(M_max):
            inertia_weight = self.ws - ((self.ws - self.we) * float(i)) / float(M_max)

            population.update(inertia_weight, self.c1, self.c2)
            if population.num_evaluations > max_evaluations:
                return generations
            
            generations.append(self.createGenData(population))

        return generations