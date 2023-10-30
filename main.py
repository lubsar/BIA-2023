import blindsearch as bs
import hillclimbing as hc
import annealing as an
import tspga as ga
import diffevolution as de
import particleswarm as pso

import datetime

from common.visualization import *
from common.functions import*

def testAlgo(seed : float, deviation : float, visualResolution : int, algo, searchParams, displayLabels=False):
    print("current seed:", seed)

    functions = Functions() 
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

def test_diff_evolution():
    seed = 59794
    num_generations = 100
    num_individuals = 100

    functions = Functions() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()

        algo = de.DifferentialEvolution3D()
        algo.init(seed, function)

        result = algo.search(num_individuals, num_generations, 0.5, 0.5)

        visual.plot3DFunction(function.viewPort, function.meshInterval(30), function.calculate)
        visual.plotGenerationsAnimation(result)

        visual.show()

def test_pso():
    seed = 59794
    num_generations = 50
    num_individuals = 50

    functions = Functions() 
    for _, function in functions.__dict__.items():
        visual = Visualisation3D()

        algo = pso.ParticleSwarm3D()
        algo.init(seed, function)

        result = algo.search(pop_size=num_individuals, M_max=num_generations, 
                             c1=2.0, c2=2.0, vmaxi_coef=0.02, vmini_coef=0.001,
                             ws=0.9, we=0.4)

        visual.plot3DFunction(function.viewPort, function.meshInterval(30), function.calculate)
        visual.plotGenerationsAnimation(result)

        visual.show()

#test_blind_search()
#test_hill_climbing()
#test_annealing()
#test_tsp()
#test_diff_evolution()

test_pso()