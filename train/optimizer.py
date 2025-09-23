# train/optimizer.py

import torch
import torch.optim as optim
from model.pid_controller import PIDController

class MetaOptimizer:
    """
    MetaOptimizer wraps an AdamW optimizer with a StepLR scheduler and a PID controller for adaptive learning rate adjustments.
    """
    def __init__(self, model, base_lr, scheduler_step=10, gamma=0.5, use_pid=True):
        self.model = model
        self.base_lr = base_lr
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
        self.pid_controller = PIDController(Kp=0.05, Ki=0.005, Kd=0.001, setpoint=0.02) if use_pid else None

    def step(self, loss=None):
        """Performs an optimization step with gradient clipping and PID-based LR adjustment."""
        # Backwards-compatible API: if a loss is provided, perform backward here.
        if loss is not None:
            # Caller provided loss -> MetaOptimizer handles backward as before
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.pid_controller:
                try:
                    lr_adjustment = self.pid_controller.update(loss.item())
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = max(1e-6, param_group['lr'] + lr_adjustment)
                except Exception:
                    pass
            self.scheduler.step()
        else:
            # No loss provided -> assume caller already did backward()
            self.optimizer.step()
            # PID update not possible without loss value; still advance scheduler
            self.scheduler.step()

    def zero_grad(self):
        """Expose zero_grad to behave like a regular optimizer."""
        self.optimizer.zero_grad()

    def get_lr(self):
        """Returns the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
