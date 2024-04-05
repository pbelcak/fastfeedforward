from typing import List, Optional

import torch
from torch import Tensor

class LocalAdam(torch.optim.Optimizer):
    """Implements Adam algorithm with local batch size support.

    All input parameters are as per :class:`torch.optim.Adam` except for the following:
     - `usage`: a tensor of shape (N,) where N is the number of parameter blocks (to have separate gradient accumulations) in the group. If specified, the optimizer will only update parameters where usage[i] >= local_batch_size[i].
     - `local_batch_size`: a tensor of shape (N,) where N is the number of parameter blocks (to have separate gradient accumulations) in the group. If specified, the optimizer will only update parameters where usage[i] >= local_batch_size[i].
     - `differentiable`, `fusion`, and `foreach` from the original implementation are not supported.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *,
                 usage: Optional[torch.Tensor] = None, local_batch_size: Optional[int] = None,
                 maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if usage is not None and not isinstance(usage, torch.Tensor):
            raise ValueError("Parameter usage must be a tensor")
        if local_batch_size is not None and local_batch_size <= 0:
            raise ValueError("Invalid local_batch_size value: {}, must be > 0 or unspecified altogether (None)".format(local_batch_size))

        params = list(params)

        for param in params:
            if not isinstance(param, dict):
                continue

            if 'usage' in param:
                if not isinstance(param['usage'], torch.Tensor):
                    raise ValueError("Parameter usage must be a tensor")
                for p in param['params']:
                    if p.size(0) != param['usage'].size(0):
                        raise ValueError("Every tensor in param['params'] must have the same size(0) as param['usage'].size(0)")

            if 'local_batch_size' in param:
                if not isinstance(param['local_batch_size'], int):
                    raise ValueError("Parameter local_batch_size must be an int")
                if param['local_batch_size'] <= 0:
                    raise ValueError("Parameter local_batch_size must be >0")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize,
                        usage=usage, local_batch_size=local_batch_size,)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('usage', None)
            group.setdefault('local_batch_size', None)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(s['step'], dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        u_list,
        lbs_list,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('LocalAdam does not support sparse gradients; perhaps consider torch.optim.SparseAdam')
                grads.append(p.grad)

                if group['usage'] is not None and group['local_batch_size'] is not None:
                    u_list.append(group['usage'])
                    lbs_list.append(group['local_batch_size'])
                else:
                    u_list.append(None)
                    lbs_list.append(None)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    descent_controlled_by_usage = group['usage'] is not None and group['local_batch_size'] is not None
                    state['step'] = (
                        torch.tensor(0.) if not descent_controlled_by_usage else torch.zeros_like(group['usage'], dtype=torch.float32, memory_format=torch.preserve_format)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state_steps.append(state['step'])

    def zero_grad(self, set_to_none: bool = False):
        usage_zeroing_per_usage_holder = {}
        for group in self.param_groups:
            u = group['usage']
            if u is not None and group['local_batch_size'] is not None:
                where_to_zero = u >= group['local_batch_size']
            else:
                where_to_zero = None

            for p in group['params']:
                if p.grad is not None:
                    if where_to_zero is None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            # DO NOT MODIFY requires_grad even though the base implementation does; ever!
                            p.grad.zero_()
                    else:
                        p.grad[where_to_zero] = 0

            if where_to_zero is not None:
                if u in usage_zeroing_per_usage_holder:
                    usage_zeroing_per_usage_holder[u].logical_or_(where_to_zero)
                else:
                    usage_zeroing_per_usage_holder[u] = where_to_zero.clone().detach()

        # now we can zero out the gradients for each usage holder
        for u, where_to_zero in usage_zeroing_per_usage_holder.items():
            u[where_to_zero] = 0

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            u_list = []
            lbs_list = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            self._init_group(
                group,
                params_with_grad,
                grads,
                u_list,
                lbs_list,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            adam(
                params_with_grad,
                grads,
                u_list,
                lbs_list,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss

def adam(params: List[Tensor],
         grads: List[Tensor],
         u_list: List[Optional[Tensor]],
         lbs_list: List[Optional[Tensor]],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         grad_scale: Optional[Tensor] = None,
         found_inf: Optional[Tensor] = None,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         maximize: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_adam

    func(params,
         grads,
         u_list,
         lbs_list,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         grad_scale=grad_scale,
         found_inf=found_inf)

@torch.no_grad()
def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        u_list: List[Optional[Tensor]],
                        lbs_list: List[Optional[Tensor]],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool
                        ):

    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        u = u_list[i]
        lbs = lbs_list[i]

        # find out where updates are due
        if u is not None and lbs is not None:
            to_update = u >= lbs
            to_update_expanded = to_update.reshape(to_update.shape + (1,) * (len(grad.shape) - len(to_update.shape)))
            to_update_expanded = to_update_expanded.expand_as(grad)
        else:
            to_update = None

        # correct the grad where more accumulations have happened
        if to_update is not None:
            correction = torch.where(to_update, u / lbs, torch.ones_like(u))
            correction_expanded = correction.reshape(correction.shape + (1,) * (len(grad.shape) - len(correction.shape)))
            correction_expanded = correction_expanded.expand_as(grad)
            grad = grad / correction_expanded

        # update the step counter where an update is imminent
        if to_update is not None:
            state_steps[i].copy_(torch.where(to_update, state_steps[i] + 1, state_steps[i]))
        else:
            state_steps[i].add_(1)

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        if to_update is not None:
            exp_avg.copy_(torch.where(to_update_expanded, exp_avg.mul_(beta1).add(grad, alpha=1 - beta1), exp_avg))
            exp_avg_sq.copy_(torch.where(to_update_expanded, exp_avg_sq.mul(beta2).addcmul(grad, grad.conj(), value=1 - beta2), exp_avg_sq))
        else:
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        step = state_steps[i]

        bias_correction1 = torch.where(to_update, 1 - beta1 ** step, torch.ones_like(step))
        bias_correction2 = torch.where(to_update, 1 - beta2 ** step, torch.ones_like(step))
        bias_correction2 = bias_correction2.reshape(bias_correction2.shape + (1,) * (len(exp_avg_sq.shape) - len(bias_correction2.shape)))

        step_size = lr / bias_correction1
        step_size = step_size.reshape(step_size.shape + (1,) * (len(exp_avg.shape) - len(step_size.shape)))

        bias_correction2_sqrt = torch.sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            if to_update is not None:
                torch.maximum(
                    max_exp_avg_sqs[i],
                    torch.where(to_update_expanded, exp_avg_sq, max_exp_avg_sqs[i]),
                    out=max_exp_avg_sqs[i]
                )
            else:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        if to_update is None:
            param.copy_(param + exp_avg / denom * -step_size)
        else:
            param.copy_(torch.where(
                to_update_expanded,
                param + exp_avg / denom * -step_size,
                param
            ))