import numpy as np
import random


from common.interval import *

class TestFunction:
    def __init__(self, bounds : list[tuple[float, float]]) -> None:
        self.bounds = bounds[:-1]
        self.viewPort = Interval3D((*bounds[0], 0.1), (*bounds[1], 0.1), (*bounds[2], 0.1))
        self.scales = [abs(bound[1] - bound[0]) for bound in self.bounds]

        self.samplingSeed = None

    def meshInterval(self, num_steps : int) -> Interval2D:
        startX, endX = self.bounds[0]
        startY, endY = self.bounds[1]
        
        stepX = abs(endX - startX) / float(num_steps)
        stepY = abs(endY - startY) / float(num_steps)

        return Interval2D((startX, endX, stepX), (startY, endY, stepY))

    def getBounds(self, dim_index : int) -> tuple[float, float]:
        return self.bounds[dim_index]
    
    def setSamplingSeed(self, seed) -> None:
        if seed == None:
            return

        self.samplingSeed = int(seed)

        random.seed(self.samplingSeed)
        np.random.seed(self.samplingSeed)

    def getSamplingSeed(self) -> int:
        return self.samplingSeed

    def randomSample(self) -> tuple[float, ...]:
        point = [random.random() * (mx - mn) + mn for mn, mx in self.bounds]
        return (*point, self.calculate(point))
    
    def randomSamples(self, num_samples: int) -> list[tuple[float, ...]]:
        return [self.randomSample() for _ in range(num_samples)]

    def randomPoints(self, num_points : int) -> list[tuple]:
        return [tuple([random.random() * (mx - mn) + mn for mn, mx in self.bounds]) for _ in range(num_points)]

    def normalSample(self, center : tuple[float, ...], sigma : float) -> tuple[float, ...]:
        scaled_sigmas = [sigma * scale for sigma, scale in self.scales]
        point = np.random.normal(center, scaled_sigmas)  

        while any(point[dim] < self.bounds[dim][0] or point[dim] > self.bounds[dim][1] for dim in range(len(center))):
             point = np.random.normal(center, scaled_sigmas)

        return (*point, self.calculate(point))
    
    def normalSamples(self, center : tuple[float, ...], sigma : float, num_samples : int) -> list:
        return [self.normalSample(center, sigma) for _ in range(num_samples)]
    
    def preserveBoundsLoopAround(self, point : tuple[float, ...]) -> tuple[float, ...] :
        new_point = []

        for dim in range(len(point)):
            coord = point[dim]
            size = self.scales[dim]
            mn, mx = self.bounds[dim]

            new_point.append(coord + size if coord < mn else (coord - size if coord > mx else coord))

        return tuple(new_point)
    
    def preserveBoundsSetAtBorder(self, point : tuple[float, ...]) -> tuple[float, ...]:
        new_point = []

        for dim in range(len(point)):
            coord = point[dim]
            mn, mx = self.bounds[dim]

            new_point.append(mn if coord < mn else (mx if coord > mx else coord))

        return tuple(new_point)
    
    # Implemented by each test function
    def calculate(self, params : tuple[float, ...]) -> float:
        raise NotImplemented("Calculate function is not implemented")

class Sphere(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-5.12, 5.12) for _ in range(dims -1)], (0.0, 100.0)))

    def calculate(self, params) -> float:
        sum = 0
        for p in params:
            sum += p**2
            
        return sum

class Ackley(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-32.768, 32.768) for _ in range(dims -1)], (0.0, 25.0)))

    def calculate(self, params) -> float:
        a = 20
        b = 0.2
        c = 2 * np.pi

        one_over_dimension = 1.0 / len(params)

        cos_part = one_over_dimension * sum([np.cos(c * x) for x in params])
        sqrt_part = - b * np.sqrt(one_over_dimension * sum([x * x for x in params]))

        return - a * np.exp(sqrt_part) - np.exp(cos_part) + a + np.exp(1)

class Rastrigin(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-5.12, 5.12) for _ in range(dims -1)], (0.0, 100.0)))

    def calculate(self, params) -> float:
       num_dimensions = len(params)

       return 10 * num_dimensions + sum([(x * x - 10 * np.cos(2 * np.pi * x)) for x in params])

class Rosenbrock(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-10.0, 10) for _ in range (dims -1)], (0.0, 1000000.0)))

    def calculate(self, params) -> float:
        result = 0.0

        for i in range(len(params) -1):
            x_i = params[i]
            result = result + 100 * ((params[i + 1] - (x_i * x_i)) ** 2) + ((x_i - 1) ** 2)

        return result

class Griewank(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-5.0, 5.0) for _ in range(dims - 1)], (0.0, 3.0)))

    def calculate(self, params) -> float:
        sum_result = 0.0
        prod_result = 1.0

        for i,x in enumerate(params):
            sum_result += (x * x)/400.0
            prod_result *= np.cos(x/np.sqrt(float(i + 1))) 

        return sum_result - prod_result + 1.0

class Schwefel(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-500.0, 500.0) for _ in range(dims -1)], (0.0, 2000)))

    def calculate(self, params) -> float:
        return 418.9829 * len(params) - sum([x * np.sin(np.sqrt(abs(x))) for x in params])  

class Levy(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-10.0, 10.0) for _ in range(dims - 1)],(0, 100.0)))

    def calculate(self, params) -> float:
        def w_i(x):
            return 1.0 + ((x - 1.0) / 4.0)
        
        term1 = np.sin(np.pi * w_i(params[0])) ** 2
        term2 = sum([(w_i(x) - 1) ** 2  * (1 + 10 * (np.sin(np.pi * w_i(x) + 1.0) ** 2)) for x in params[:-1]])
        term3 = ((w_i(params[-1]) -1) ** 2)*(1 + np.sin(2 * np.pi * w_i(params[-1])))

        return term1 + term2 + term3
        
class Michalewicz(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(0, np.pi) for _ in range(dims - 1)], (-2.0, 0.0)))

    def calculate(self, params) -> float:
        m = 10

        result = 0.0
        for i in range(1, len(params) + 1):
            x_i = params[i - 1]

            result += np.sin(x_i) * (np.sin((i * x_i * x_i)/np.pi) ** (2 * m))

        return -result

class Zakharov(TestFunction):
    def __init__(self, dims : int) -> None:
        super().__init__((*[(-5, 10) for _ in range(dims - 1)], (0, 100000.0)))

    def calculate(self, params) -> float:
        tmp1 = 0.0
        tmp2 = 0.0

        for i in range(1, len(params) + 1):
            x_i = params[i - 1]

            tmp1 += x_i * x_i
            tmp2 += 0.5 * i * x_i

        return tmp1 + tmp2 ** 2 + tmp2 ** 4

class Functions:
    def __init__(self, dims : int) :
        self.sphere = Sphere(dims) 
        self.ackley = Ackley(dims)
        self.rastrigin = Rastrigin(dims)
        self.rosenbrock = Rosenbrock(dims)
        self.griewank = Griewank(dims)
        self.schwefel = Schwefel(dims)
        self.levy = Levy(dims)
        self.michalewicz = Michalewicz(dims)
        self.zakharov = Zakharov(dims)

class Functions3D(Functions):
    def __init__(self):
        super().__init__(3)
       
class Functions30D:
    def __init__(self):
        super().__init__(30)