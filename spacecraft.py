class Spacecraft:
    def __init__(self, J, controller, sensors, actuators):
        """Constructs a Spacecraft object to store system information and objects
        
        Args:
            J (numpy ndarray): the spacecraft's inertia tensor (3x3) (kg * m^2)
            controller ([type]): [description]
            sensors ([type]): [description]
            actuators ([type]): [description]
        """
        self.J = J
        self.controller = controller
        self.sensors = sensors
        self.actuators = actuators
