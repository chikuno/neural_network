# train/optimizer.py

import torch
import torch.optim as optim
from model.pid_controller import PIDController

class MetaOptimizer:
    """
    MetaOptimizer wraps an AdamW optimizer with a scheduler + optional PID controller.
    Enhancements:
      - Weight decay configurable (decoupled via AdamW)
      - Exponential Moving Average (EMA) shadow weights for evaluation stability
    """
    def __init__(self, model, base_lr, scheduler_step=10, gamma=0.5, use_pid=True,
                 warmup_steps=0, scheduler_type='step', total_steps=None,
                 weight_decay=0.0, use_ema=False, ema_decay=0.999, ema_eval_only=True,
                 ema_start_step=0):
        self.model = model
        self.base_lr = base_lr
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        # Choose scheduler
        self.scheduler_type = scheduler_type
        if scheduler_type == 'cosine':
            # CosineAnnealingLR requires T_max; approximate if total_steps provided, else fallback to StepLR
            if total_steps is not None and total_steps > 0:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
        elif scheduler_type == 'linear':
            # Implement simple linear decay via LambdaLR
            if total_steps is not None and total_steps > 0:
                def lr_lambda(step):
                    # After warmup, linearly decay to near-zero by total_steps
                    if step <= max(1, warmup_steps):
                        return max(1e-3, step / float(max(1, warmup_steps)))
                    remain = max(1, total_steps - warmup_steps)
                    frac = max(0.0, min(1.0, (step - warmup_steps) / float(remain)))
                    return max(1e-3, 1.0 - frac)
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
        elif scheduler_type == 'plateau':
            # Reduce LR on Plateau will be stepped with validation loss in train loop
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=gamma, patience=max(1, scheduler_step), verbose=False)
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
        self.pid_controller = PIDController(Kp=0.05, Ki=0.005, Kd=0.001, setpoint=0.02) if use_pid else None
        self.warmup_steps = int(max(0, warmup_steps))
        self.step_count = 0
        # --- EMA support ---
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_eval_only = bool(ema_eval_only)
        self.ema_start_step = int(max(0, ema_start_step))
        self._ema_shadow = {}
        self._ema_backup = None  # holds weights when swapping for eval
        if self.use_ema:
            # Initialize shadow with current params (will be updated lazily after start step)
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self._ema_shadow[name] = p.detach().clone()

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
            self._after_step()
        else:
            # No loss provided -> assume caller already did backward()
            self.optimizer.step()
            # PID update not possible without loss value
            self._after_step()
        # After optimizer update, update EMA if enabled
        if self.use_ema and self.step_count >= self.ema_start_step:
            with torch.no_grad():
                d = self.ema_decay
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name not in self._ema_shadow:
                        self._ema_shadow[name] = p.detach().clone()
                    else:
                        self._ema_shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def _after_step(self):
        # Apply warmup for the first N steps: linearly scale lr from small to base
        self.step_count += 1
        if self.warmup_steps > 0 and self.step_count <= self.warmup_steps:
            scale = self.step_count / float(self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(1e-6, self.base_lr * scale)
        else:
            # Advance scheduler automatically for step/linear/cosine; plateau handled with val loss in train loop
            if self.scheduler_type != 'plateau':
                self.scheduler.step()

    def zero_grad(self):
        """Expose zero_grad to behave like a regular optimizer."""
        self.optimizer.zero_grad()

    def get_lr(self):
        """Returns the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    # ---------------- EMA utility methods ----------------
    def swap_to_ema(self):
        """Swap model parameters to EMA shadow (store original)."""
        if not self.use_ema:
            return
        if self._ema_backup is not None:
            return  # already swapped
        self._ema_backup = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name in self._ema_shadow:
                    self._ema_backup[name] = p.detach().clone()
                    p.data.copy_(self._ema_shadow[name])

    def restore_from_ema(self):
        """Restore original parameters after an EMA-evaluated pass."""
        if not self.use_ema or self._ema_backup is None:
            return
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self._ema_backup:
                    p.data.copy_(self._ema_backup[name])
        self._ema_backup = None

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step_count': self.step_count,
            'warmup_steps': self.warmup_steps,
            'base_lr': self.base_lr,
            'scheduler_type': self.scheduler_type,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
            'ema_eval_only': self.ema_eval_only,
            'ema_start_step': self.ema_start_step,
            'ema_shadow': {k: v.cpu() for k, v in self._ema_shadow.items()} if self.use_ema else None,
        }

    def load_state_dict(self, state_dict):
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'scheduler' in state_dict:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
            except Exception:
                pass
        self.step_count = state_dict.get('step_count', 0)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.scheduler_type = state_dict.get('scheduler_type', getattr(self, 'scheduler_type', 'step'))
        # Restore EMA shadow if present
        self.use_ema = state_dict.get('use_ema', self.use_ema)
        if self.use_ema:
            self.ema_decay = state_dict.get('ema_decay', self.ema_decay)
            self.ema_eval_only = state_dict.get('ema_eval_only', self.ema_eval_only)
            self.ema_start_step = state_dict.get('ema_start_step', self.ema_start_step)
            shadow = state_dict.get('ema_shadow', None)
            if shadow:
                # Move to current device
                device = next(self.model.parameters()).device
                self._ema_shadow = {k: v.to(device) for k, v in shadow.items()}
