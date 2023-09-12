import torch
import torch.optim as optim
from typing import Optional, Callable, List
from torch import Tensor

class LocalSGD(optim.Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, usage: Optional[torch.Tensor] = None, local_batch_size: Optional[int] = None, maximize: bool = False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if usage is not None and not isinstance(param['usage'], torch.Tensor):
            raise ValueError("Parameter usage must be a tensor")
        if local_batch_size is not None and local_batch_size <= 0:
            raise ValueError("Invalid local_batch_size value: {}, must be >0".format(local_batch_size))
        
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

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, 
                        usage=usage, local_batch_size=local_batch_size)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('usage', None)
            group.setdefault('local_batch_size', None)

    def _init_group(self, group, params_with_grad, d_p_list, u_list, lbs_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if group['usage'] is not None and group['local_batch_size'] is not None:
                    u_list.append(group['usage'])
                    lbs_list.append(group['local_batch_size'])
                else:
                    u_list.append(None)
                    lbs_list.append(None)
                
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad

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
                            # DO NOT MODIFY requires_grad even though the base implementation does, ever!
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


    def step(self, closure: Optional[Callable] = None, update_locations: Optional[list[tuple[int, torch.ByteTensor]]] = None):
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
            d_p_list = []
            u_list = []
            lbs_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(
                group,
                params_with_grad, d_p_list, u_list, lbs_list,
                momentum_buffer_list
            )

            sgd(params_with_grad,
                d_p_list,
                u_list,
                lbs_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                update_locations=update_locations)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

@torch.no_grad()
def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        u_list: List[Optional[Tensor]],
        lbs_list: List[Optional[Tensor]],
        momentum_buffer_list: List[Optional[Tensor]],
        has_sparse_grad: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        update_locations: Optional[list[tuple[int, torch.ByteTensor]]] = None):

    for i, param in enumerate(params):
        d_p = (d_p_list[i] if not maximize else -d_p_list[i]).clone()
        u = u_list[i]
        lbs = lbs_list[i]

        if u is not None and lbs is not None:
            d_p_filter_based_on_usage = u >= lbs
            correction = lbs / u
            correction = correction.reshape(correction.shape + (1,) * (len(d_p.shape) - len(correction.shape)))
            correction = correction.expand_as(d_p)
        else:
            d_p_filter_based_on_usage = torch.ones_like(u, dtype=torch.bool)
            correction = 1.0

        dpfbou = d_p_filter_based_on_usage
        dpfbou = dpfbou.reshape(dpfbou.shape + (1,) * (len(d_p.shape) - len(dpfbou.shape)))
        dpfbou = dpfbou.expand_as(d_p)
        d_p.mul_(correction)

        # the following is only to compute the stats for logging
        if update_locations is not None:
            indices_of_update = d_p_filter_based_on_usage.nonzero(as_tuple=True)[0] # (num_updates,)
            update_locations.append((i, indices_of_update))

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None: # lazy init
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                momentum_buffer_list[i] = torch.where(dpfbou, buf.mul(momentum).add(d_p, alpha=1 - dampening), buf)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        d_p[~dpfbou] = 0 # ignore the gradients where there hasn't been sufficient usage
        param.add_(d_p, alpha=-lr)