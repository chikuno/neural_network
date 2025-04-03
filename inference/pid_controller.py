# inference/pid_controller.py

class PIDController:
    """PID Controller for inference stability and parameter adjustment."""
    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.01, setpoint=0.5):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.prev_error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output
