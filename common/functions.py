import numpy as np
import random
import datetime 

from common.interval import *

class TestFunction:
    def __init__(self, viewPortParams : tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
        self.viewPort = Interval3D((*viewPortParams[0], 0.0), (*viewPortParams[1], 0.0), (*viewPortParams[2], 0.0))

        xMin, xMax = self.getXBounds()
        yMin, yMax = self.getYBounds()

        self.xScale = xMax - xMin
        self.yScale = yMax - yMin

        self.samplingSeed = None

    def meshInterval(self, num_steps : int) -> Interval2D:
        startX, endX, stepX = self.viewPort.getXInteval()
        startY, endY, stepY = self.viewPort.getYInteval()
        
        stepX = abs(endX - startX) / float(num_steps)
        stepY = abs(endY - startY) / float(num_steps)

        return Interval2D((startX, endX, stepX), (startY, endY, stepY))

    def getXBounds(self) -> tuple[float, float]:
        startX, endX, stepX = self.viewPort.getXInteval()
        
        return (startX, endX)
    
    
    def getYBounds(self) -> tuple[float, float]:
        startY, endY, stepY = self.viewPort.getYInteval()

        return (startY, endY)
    
    def setSamplingSeed(self, seed) -> None:
         self.samplingSeed = int(seed)

         random.seed(self.samplingSeed)
         np.random.seed(self.samplingSeed)

    def getSamplingSeed(self) -> int:
        return self.samplingSeed


    def randomSample(self) -> tuple[float, float, float]:
        xMin, xMax = self.getXBounds()
        yMin, yMax = self.getYBounds()

        x = random.random() * (xMax - xMin) + xMin
        y = random.random() *  (yMax - yMin) + yMin

        return (x, y, self.calculate((x,y)))
    
    def randomSamples(self, num_samples) -> list[tuple[float, float, float]]:
        xMin, xMax = self.getXBounds()
        yMin, yMax = self.getYBounds()

        points = [(random.random() * (xMax - xMin) + xMin, random.random() * (yMax - yMin) + yMin) for _ in range(num_samples)]
        samples = [(x[0], x[1], self.calculate((x[0], x[1]))) for x in points]

        return samples

    def randomPoints(self, num_points) -> list[tuple[float, float]]:
        xMin, xMax = self.getXBounds()
        yMin, yMax = self.getYBounds()

        points = [(random.random() * (xMax - xMin) + xMin, random.random() * (yMax - yMin) + yMin) for _ in range(num_points)]
        
        return points

    def normalSample(self, center : tuple[float, float], sigma : float) -> tuple[float, float, float]:
        xSigma = sigma * self.xScale
        ySigma = sigma * self.yScale

        point = np.random.normal(center, (xSigma, ySigma))

        xMin, xMax = self.getXBounds()
        yMin, yMax = self.getYBounds()      

        while point[0] < xMin or point[1] < yMin or point[0] > xMax or point[1] > yMax:
             point = np.random.normal(center, (xSigma, ySigma))

        value = self.calculate((point[0], point[1]))

        return (*point, value)
    
    def normalSamples(self, center : tuple[float, float], sigma : float, num_samples : int) -> list:
        return [self.normalSample(center, sigma) for _ in range(num_samples)]
    
    def preserveBounds(self, point) :
        xMin, xMax = self.getXBounds()
        yMin, yMax = self.getYBounds()   

        xSize = xMax - xMin
        ySize = yMax - yMin

        new_point = point

        if point[0] < xMin:
            new_point[0] += xSize
        elif point[0] > xMax:
            new_point[0] -= xSize

        if point[1] < yMin:
            new_point[1] += ySize
        elif point[1] > yMax:
            new_point[1] -= ySize

        return new_point


class Sphere(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-5.12, 5.12), 
                         (-5.12, 5.12),
                         (0.0, 100.0)))

    def calculate(self, params) -> float:
        sum = 0
        for p in params:
            sum += p**2
            
        return sum

class Ackley(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-32.768, 32.768),
                         (-32.768, 32.768), 
                         (0.0, 25.0)))

    def calculate(self, params) -> float:
        a = 20
        b = 0.2
        c = 2 * np.pi

        one_over_dimension = 1.0 / len(params)

        cos_part = one_over_dimension * sum([np.cos(c * x) for x in params])
        sqrt_part = - b * np.sqrt(one_over_dimension * sum([x * x for x in params]))

        return - a * np.exp(sqrt_part) - np.exp(cos_part) + a + np.exp(1)

class Rastrigin(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-5.12, 5.12), 
                         (-5.12, 5.12),
                         (0.0, 100.0)))

    def calculate(self, params) -> float:
       num_dimensions = len(params)

       return 10 * num_dimensions + sum([(x * x - 10 * np.cos(2 * np.pi * x)) for x in params])

class Rosenbrock(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-10.0, 10), 
                         (-6.0, 6),
                         (0.0, 1000000.0)))

    def calculate(self, params) -> float:
        result = 0.0

        for i in range(len(params) -1):
            x_i = params[i]
            result = result + 100 * ((params[i + 1] - (x_i * x_i)) ** 2) + ((x_i - 1) ** 2)

        return result

class Griewank(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-5.0, 5.0), 
                         (-5.0, 5.0),
                         (0.0, 3.0)))

    def calculate(self, params) -> float:
        sum_result = 0.0
        prod_result = 1.0

        for i,x in enumerate(params):
            sum_result += (x * x)/400.0
            prod_result *= np.cos(x/np.sqrt(float(i + 1))) 

        return sum_result - prod_result + 1.0

class Schwefel(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-500.0, 500.0), 
                         (-500.0, 500.0),
                         (0.0, 2000)))

    def calculate(self, params) -> float:
        return 418.9829 * len(params) - sum([x * np.sin(np.sqrt(abs(x))) for x in params])  

class Levy(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-10.0, 10.0), 
                            (-10.0, 10.0),
                            (0, 100.0)))

    def calculate(self, params) -> float:
        def w_i(x):
            return 1.0 + ((x - 1.0) / 4.0)
        
        term1 = np.sin(np.pi * w_i(params[0])) ** 2
        term2 = sum([(w_i(x) - 1) ** 2  * (1 + 10 * (np.sin(np.pi * w_i(x) + 1.0) ** 2)) for x in params[:-1]])
        term3 = ((w_i(params[-1]) -1) ** 2)*(1 + np.sin(2 * np.pi * w_i(params[-1])))

        return term1 + term2 + term3
        
class Michalewicz(TestFunction):
    def __init__(self) -> None:
        super().__init__(((0, np.pi), 
                            (0, np.pi),
                            (-2.0, 0.0)))

    def calculate(self, params) -> float:
        m = 10

        result = 0.0
        for i in range(1, len(params) + 1):
            x_i = params[i - 1]

            result += np.sin(x_i) * (np.sin((i * x_i * x_i)/np.pi) ** (2 * m))

        return -result

class Zakharov(TestFunction):
    def __init__(self) -> None:
        super().__init__(((-5, 10), 
                            (-5, 10),
                            (0, 100000.0)))

    def calculate(self, params) -> float:
        tmp1 = 0.0
        tmp2 = 0.0

        for i in range(1, len(params) + 1):
            x_i = params[i - 1]

            tmp1 += x_i * x_i
            tmp2 += 0.5 * i * x_i

        return tmp1 + tmp2 ** 2 + tmp2 ** 4

class Functions:
    def __init__(self):
        self.sphere = Sphere() 
        self.ackley = Ackley()
        self.rastrigin = Rastrigin()
        self.rosenbrock = Rosenbrock()
        self.griewank = Griewank()
        self.schwefel = Schwefel()
        self.levy = Levy()
        self.michalewicz = Michalewicz()
        self.zakharov = Zakharov()