# pid_controller.py

class PIDController:
    """PID Controller for adjusting training parameters dynamically."""

    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.01, setpoint=0.5):
        """
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
        setpoint: Target value (e.g., target loss, target temperature)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.prev_error = 0
        self.integral = 0

    def update(self, current_value):
        """Computes the new control output based on PID formula."""
        error = self.setpoint - current_value  # Difference between target and current value
        self.integral += error  # Cumulative error (integral term)
        derivative = error - self.prev_error  # Change in error (derivative term)

        # PID formula
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        self.prev_error = error  # Store previous error for next iteration
        return output
