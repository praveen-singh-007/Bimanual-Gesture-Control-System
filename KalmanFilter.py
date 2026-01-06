class Kalman1D():
    def __init__(self, process_var = 1e-3, measurement_var = 1.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.state = None
        self.error = 1.0
        self.kalman_gain = 0.0

    def update(self, measurement):
        if self.state is None:
            self.state = measurement
            return self.state
        
        self.error = self.error + self.process_var

        self.kalman_gain = self.error/ (self.error + self.measurement_var)
        self.state = self.state + self.kalman_gain*(measurement - self.state)
        self.error = (1 - self.kalman_gain)*self.error

        return self.state
        
