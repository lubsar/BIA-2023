import blindsearch as bs
import hillclimbing as hc
import annealing as an
import tspga as ga
import diffevolution as de
import particleswarm as pso
import somaa2o as soma
import tspaco as aco
import firefly as ff
import tlbo as tl

import datetime

from common.visualization import *
from common.functions import*

import csv

def testAlgo(seed : float, deviation : float, visualResolution : int, algo, searchParams, displayLabels=False):
    print("current seed:", seed)

    functions = Functions3D() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()
        algo.init(seed, function, deviation)
        X, Y, Z, labels = algo.search(*searchParams)

        print("best: ", (labels[-1], X[-1], Y[-1], Z[-1]))

        visual.plot3DFunction(function.viewPort, function.meshInterval(visualResolution), function.calculate)
        visual.plotPointsAnimation((X, Y, Z), labels if displayLabels else None)

        visual.show()

def test_blind_search():
    seed = datetime.datetime.now().timestamp()
    num_generations = 100
    num_samples = 100
    visual_mesh_resolution = 10

    testAlgo(algo=bs.BlindSearch3D(), seed=seed, deviation=0.03, 
             searchParams=(num_samples, num_generations), visualResolution=visual_mesh_resolution)

def test_hill_climbing():
    seed = datetime.datetime.now().timestamp()
    num_generations = 100
    num_samples = 100
    visual_mesh_resolution = 10

    testAlgo(algo=hc.HillClimbingSearch3D(), seed=seed, deviation=0.02, 
             searchParams=(num_samples, num_generations), visualResolution=visual_mesh_resolution)

def test_annealing():
    seed = datetime.datetime.now().timestamp()
    visual_mesh_resolution = 20
    initial_temperature = 200.0
    min_temperature = 0.1
    cooling_factor = 0.9

    testAlgo(algo=an.SimulatedAnnealing3D(), seed=seed, deviation=0.05,
             visualResolution=visual_mesh_resolution, searchParams=(initial_temperature, min_temperature, cooling_factor))

def test_tsp():
    seed = 59794
    num_generations = 1000
    num_individuals = 100
    num_cities = 40

    np.random.seed(seed)

    points = (np.random.uniform(0, 1000, num_cities), np.random.uniform(0, 1000, num_cities))
    area = Interval2D((-100, 1100, 1.0), (-100, 1100, 1.0))
    
    points_list = []
    for i in range(len(points[0])):
        points_list.append((points[0][i], points[1][i]))

    algo = ga.TSPGA(points_list)
    last_gen, gen_data = algo.run(num_generations, num_individuals, seed)

    sorted_generation = sorted(last_gen, key= lambda x : x.getPathLength())
    path_points = ([points[0][x] for x in sorted_generation[0].getPath()[:-1]],
                   [points[1][y] for y in sorted_generation[0].getPath()[:-1]])

    # for individual in sorted_generation:
    #     print(individual.getPath(), individual.getPathLength())

    visual = Visualisation2D()
    visual.plotLine([x["min"] for x in gen_data], "Distance")
    visual.plotLine([x["max"] for x in gen_data], "Distance")
    visual.saveFig("tsp_{0}_cities_{1}_gens_{2}_individuals_seed{3}-progress.pdf".format(num_cities, num_generations, num_individuals, seed))
    visual.show()
    visual.cleanup()

    visual = Visualisation2D()
    visual.plotPath(area, path_points)
    visual.saveFig("tsp_{0}_cities_{1}_gens_{2}_individuals_seed{3}-path.pdf".format(num_cities, num_generations, num_individuals, seed))
    visual.show()

def createPopulationVisual(population):
        X = []
        Y = []
        Z = []

        for individual in population:
            X.append(individual[0])
            Y.append(individual[1])
            Z.append(individual[2])

        return (X, Y, Z)

def test_diff_evolution():
    seed = 59794
    num_generations = 100
    num_individuals = 30

    functions = Functions3D() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()

        algo = de.DifferentialEvolution()
        algo.init(seed, function)

        result = algo.search(num_individuals, num_generations, 0.5, 0.5, 3000)

        visual.plot3DFunction(function.viewPort, function.meshInterval(30), function.calculate)
        visual.plotGenerationsAnimation([createPopulationVisual(gen) for gen in result])

        visual.show()

def test_pso():
    seed = 59794
    num_generations = 50
    num_individuals = 50

    functions = Functions3D() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()

        algo = pso.ParticleSwarm(c1=2.0, c2=2.0, vmaxi_coef=0.02, vmini_coef=0.001,
                             ws=0.9, we=0.4)
        algo.init(seed, function)

        result = algo.search(pop_size=num_individuals, M_max=num_generations, max_evaluations=3000)

        visual.plot3DFunction(function.viewPort, function.meshInterval(30), function.calculate)
        visual.plotGenerationsAnimation([createPopulationVisual(gen) for gen in result])

        visual.show()

def test_soma():
    seed = 59794
    num_generations = 100
    num_individuals = 30
    PRT = 0.4
    pathLength = 3.0
    step = 0.11

    functions = Functions3D() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()

        algo = soma.SomaA2O(PRT, pathLength, step)
        algo.init(seed, function)

        result = algo.search(num_individuals, num_generations, 3000)

        visual.plot3DFunction(function.viewPort, function.meshInterval(30), function.calculate)
        visual.plotGenerationsAnimation([createPopulationVisual(gen) for gen in result])

        visual.show()

def test_tsp_aco():
    seed = 59794
    num_generations = 500
    num_ants = 100
    num_cities = 20

    np.random.seed(seed)

    points = (np.random.uniform(0, 100, num_cities), np.random.uniform(0, 100, num_cities))
    area = Interval2D((-0.1, 100.1, 1.0), (-0.1, 100.1, 1.0))
    
    points_list = []
    for i in range(len(points[0])):
        points_list.append((points[0][i], points[1][i]))

    algo = aco.TSPACO(points_list, 1.0, 2.0)
    paths = algo.run(num_generations, num_ants, seed, 0.02)

    path_points = []

    for path in paths:
        points_x = [points[0][x] for x in path[:-1]]
        points_y = [points[1][y] for y in path[:-1]]

        path_points.append((points_x, points_y))

    # for individual in sorted_generation:
    #     print(individual.getPath(), individual.getPathLength())

    #visual = Visualisation2D()
    # visual.plotLine([x["min"] for x in path_points], "Distance")
    # visual.plotLine([x["max"] for x in path_points], "Distance")
    # visual.saveFig("tsp_{0}_cities_{1}_gens_{2}_ants_seed{3}-progress.pdf".format(num_cities, num_generations, num_ants, seed))
    # visual.show()
    # visual.cleanup()

    visual = Visualisation2D()
    visual.plotPathsAnimation(area, path_points)
    visual.saveFig("tsp_{0}_cities_{1}_gens_{2}_ants_seed{3}-path.pdf".format(num_cities, num_generations, num_ants, seed))
    visual.show()

def test_firefly():
    seed = 59794
    num_generations = 1000
    num_individuals = 20

    functions = Functions3D() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()

        algo = ff.FireflySwarmp()
        algo.init(seed, function)

        result = algo.search(num_individuals, num_generations, 3000)

        visual.plot3DFunction(function.viewPort, function.meshInterval(30), function.calculate)
        visual.plotGenerationsAnimation([createPopulationVisual(gen) for gen in result])

        visual.show()

def test_tlbo():
    seed = 59794
    num_generations = 100
    num_individuals = 20

    functions = Functions3D() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()

        algo = tl.TLBO()
        algo.init(seed, function)

        result = algo.search(num_individuals, num_generations, 3000)

        visual.plot3DFunction(function.viewPort, function.meshInterval(30), function.calculate)
        visual.plotGenerationsAnimation([createPopulationVisual(gen) for gen in result])

        visual.show()

#test_blind_search()
#test_hill_climbing()
#test_annealing()
#test_tsp()
#test_diff_evolution()
#test_pso()
#test_soma()
#test_tsp_aco()
#test_firefly()
test_tlbo()


def experiment(algo, num_repetitions, num_generations, num_individuals, num_evaluations, num_dimensions):
    def runAlgo(algo, num_generations, num_individuals, num_evaluations, num_dimensions):
        functions = Functions(num_dimensions)
        results = {}
        for name, function in functions.__dict__.items():
            algo.init(None, function)
            result =  algo.search(num_individuals, num_generations, num_evaluations)

            results[name] = result
        
        return results
    
    runs = []
    for _ in range(num_repetitions):
        runs.append(runAlgo(algo, num_generations, num_individuals, num_evaluations, num_dimensions))

    return runs

def plotExperimentFunction(experiment_function_data):
    min_experiment = [min(gen, key=lambda x : x[-1])[-1] for gen in experiment_function_data]
    
    plt.plot(min_experiment)
    plt.show(block=True)
            
def plotExperiment(experiment):
    for _, data in experiment.items():
        plotExperimentFunction(data)

def writeResults(results, per_experiment):
    functions = list(results[0].keys())

    with open(per_experiment, "w", newline='') as per_exp_file:
        per_exp_writer = csv.writer(per_exp_file, delimiter=",")
        per_exp_writer.writerow(['experiment', *functions])

        for index, experiment in enumerate(results):
            per_experiment_data = [index]

            for function in functions:
                experiment_function_data = experiment[function]

                gens = [min(gen, key=lambda x : x[-1])[-1] for gen in experiment_function_data]
                min_experiment = min(gens)
                per_experiment_data.append(min_experiment)

            per_exp_writer.writerow(per_experiment_data)

# results = experiment(de.DifferentialEvolution(0.5, 0.5), 30, 30, 30, 3000, 30)
# writeResults(results, "de.csv")
# plotExperiment(results[0])

# results = experiment(soma.SomaA2O(0.4, 3.0, 0.11), 30, 30, 30, 3000, 30)
# writeResults(results, "soma.csv")
# plotExperiment(results[0])

# results = experiment(pso.ParticleSwarm(c1=2.0, c2=2.0, vmaxi_coef=0.1, vmini_coef=0.001,
#                              ws=0.9, we=0.4), 30, 30, 30, 3000, 30)
# writeResults(results, "pso.csv")
# plotExperiment(results[0])

# results = experiment(ff.FireflySwarmp(), 30, 30, 30, 3000, 30)
# writeResults(results, "firefly.csv")
# plotExperiment(results[0])

# results = experiment(tl.TLBO(), 30, 30, 30, 3000, 30)
# writeResults(results, "tlbo.csv")
# plotExperiment(results[0])

pass