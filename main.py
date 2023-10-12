import blindsearch as bs
import hillclimbing as hc
import annealing as an

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

    testAlgo(algo=hc.HillClimbingSearch3D(), seed=seed, deviation=0.03, 
             searchParams=(num_samples, num_generations), visualResolution=visual_mesh_resolution)

def test_annealing():
    seed = datetime.datetime.now().timestamp()
    visual_mesh_resolution = 50
    initial_temperature = 200.0
    min_temperature = 0.1
    cooling_factor = 0.9

    testAlgo(algo=an.SimulatedAnnealing3D(), seed=seed, deviation=0.01,
             visualResolution=visual_mesh_resolution, searchParams=(initial_temperature, min_temperature, cooling_factor))


#test_blind_search()
#test_hill_climbing()
test_annealing()
