"""Adam variant matching the Keras update rule, with decoupled weight decay."""
import math

import torch


class KerasAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-7,
                 weight_decay=0.0):
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            states = [self.state[p] for p in params]
            for state, p in zip(states, params):
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
            ms = [s["m"] for s in states]
            vs = [s["v"] for s in states]
            t = states[0]["step"] + 1
            for s in states:
                s["step"] = t
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            if group["weight_decay"]:
                decay = torch._foreach_mul(
                    params, lr * group["weight_decay"]
                )
                torch._foreach_sub_(params, decay)
            torch._foreach_mul_(ms, b1)
            torch._foreach_add_(ms, grads, alpha=1.0 - b1)
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, grads, grads, value=1.0 - b2)
            alpha = lr * math.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)
            denom = torch._foreach_sqrt(vs)
            torch._foreach_add_(denom, group["eps"])
            torch._foreach_addcdiv_(params, ms, denom, value=-alpha)
