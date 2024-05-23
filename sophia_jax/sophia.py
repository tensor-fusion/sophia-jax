import jax
import jax.numpy as jnp
from typing import List, Dict, Any


class SophiaG:
    def __init__(
        self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=1e-1
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.params = params
        self.lr = lr
        self.betas = betas
        self.rho = rho
        self.weight_decay = weight_decay
        self.state = self.init_state(params)

    def init_state(self, params):
        state = []
        for p in params:
            state.append(
                {
                    "step": jnp.array(0.0),
                    "exp_avg": jnp.zeros_like(p),
                    "hessian": jnp.zeros_like(p),
                }
            )
        return state

    def update_hessian(
        self,
        params: List[jnp.ndarray],
        grads: List[jnp.ndarray],
        state: List[Dict[str, jnp.ndarray]],
    ):
        beta2 = self.betas[1]
        new_state = []
        for p, g, s in zip(params, grads, state):
            if g is None:
                new_state.append(s)
                continue

            new_hessian = beta2 * s["hessian"] + (1 - beta2) * g * g
            new_state.append({**s, "hessian": new_hessian})
        return new_state

    def update_exp_avg(
        self,
        params: List[jnp.ndarray],
        grads: List[jnp.ndarray],
        state: List[Dict[str, jnp.ndarray]],
    ):
        beta1 = self.betas[0]
        new_state = []
        for p, g, s in zip(params, grads, state):
            if g is None:
                new_state.append(s)
                continue

            new_exp_avg = beta1 * s["exp_avg"] + (1 - beta1) * g
            new_state.append({**s, "exp_avg": new_exp_avg})
        return new_state

    def _sophiag(
        self,
        params: List[jnp.ndarray],
        grads: List[jnp.ndarray],
        exp_avgs: List[jnp.ndarray],
        hessian: List[jnp.ndarray],
        state_steps: List[jnp.ndarray],
        capturable: bool = False,
        *,
        bs: int,
        beta1: float,
        beta2: float,
        rho: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
    ):
        return self._single_tensor_sophiag(
            params,
            grads,
            exp_avgs,
            hessian,
            state_steps,
            bs=bs,
            beta1=beta1,
            beta2=beta2,
            rho=rho,
            lr=lr,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
        )

    def _single_tensor_sophiag(
        self,
        params: List[jnp.ndarray],
        grads: List[jnp.ndarray],
        exp_avgs: List[jnp.ndarray],
        hessian: List[jnp.ndarray],
        state_steps: List[jnp.ndarray],
        *,
        bs: int,
        beta1: float,
        beta2: float,
        rho: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
        capturable: bool,
    ):
        updated_params = []
        new_state_steps = []
        new_exp_avgs = []
        new_hessians = []

        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            hess = hessian[i]
            step_t = state_steps[i]

            if jnp.iscomplexobj(param):
                grad = jnp.view_as_real(grad)
                exp_avg = jnp.view_as_real(exp_avg)
                hess = jnp.view_as_real(hess)
                param = jnp.view_as_real(param)

            # Update step
            step_t += 1

            # Perform step weight decay
            param = param * (1 - lr * weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad

            # Compute the ratio and apply clipping
            ratio = jnp.clip(jnp.abs(exp_avg) / (rho * hess * bs + 1e-15), a_max=1.0)
            update = -lr * exp_avg * ratio
            param = param + update

            updated_params.append(param)
            new_state_steps.append(step_t)
            new_exp_avgs.append(exp_avg)
            new_hessians.append(hess)

        return updated_params, new_state_steps, new_exp_avgs, new_hessians

    def step(
        self,
        params: List[jnp.ndarray],
        grads: List[jnp.ndarray],
        state: List[Dict[str, jnp.ndarray]],
        bs: int = 5120,
    ):
        """
        Perform a step of the optimizer.
        """
        # Update Hessian and Exponential Average
        new_state = self.update_hessian(params, grads, state)
        new_state = self.update_exp_avg(params, grads, new_state)

        beta1, beta2 = self.betas
        params_with_grad = []
        grads_list = []
        exp_avgs = []
        state_steps = []
        hessians = []

        for p, g, s in zip(params, grads, new_state):
            if g is None:
                continue

            params_with_grad.append(p)
            grads_list.append(g)

            exp_avgs.append(s["exp_avg"])
            state_steps.append(s["step"])
            hessians.append(s["hessian"])

        # Call _sophiag to update parameters
        updated_params, new_state_steps, new_exp_avgs, new_hessians = self._sophiag(
            params_with_grad,
            grads_list,
            exp_avgs,
            hessians,
            state_steps,
            bs=bs,
            beta1=beta1,
            beta2=beta2,
            rho=self.rho,
            lr=self.lr,
            weight_decay=self.weight_decay,
            maximize=False,
            capturable=False,
        )

        # Update the state with new values
        for i, s in enumerate(new_state):
            if grads[i] is None:
                continue

            s["step"] = new_state_steps[i]
            s["exp_avg"] = new_exp_avgs[i]
            s["hessian"] = new_hessians[i]

        return updated_params, new_state
