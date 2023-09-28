
class Interval2D:
    def __init__(self, xInterval, yInterval) -> None:
        """
        xInterval: tuple: (start : float, end : float, step : float)
        yInterval: tuple: (start : float, end : float, step : float)
        """
        
        self.x_interval = xInterval
        self.y_interval = yInterval

    def getXInteval(self):
        return self.x_interval
    
    def getYInteval(self):
        return self.y_interval
    
class Interval3D:
    def __init__(self, xInterval, yInterval, zInterval ) -> None:
        """
        xInterval: tuple: (start : float, end : float, step : float)
        yInterval: tuple: (start : float, end : float, step : float)
        zInterval: tuple: (start : float, end : float, step : float)
        """
        self.x_interval = xInterval
        self.y_interval = yInterval
        self.z_interval = zInterval

    def getXInteval(self):
        return self.x_interval
    
    def getYInteval(self):
        return self.y_interval
    
    def getZInteval(self):
        return self.z_interval