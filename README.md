# fastfeedforward
A repository for fast feedforward (FFF) networks.
Fast feedforward layers can be used in place of vanilla feedforward and mixture-of-expert layers, offering inference time that grows only logarithmically in the training width of the layer.

More information can be found in the paper "Fast Feedforward Networks" ([arXiv](https://arxiv.org/abs/2308.14711)).

## Quickstart
1. Install the package.
```sh
pip install fastfeedforward
```

2. Import the `FFF` layer implementation.
```py
from fastfeedforward import FFF
```

3. Use `FFF` in place of feedforward or mixture-of-experts layers, e.g. instead of
```py
my_ff = torch.nn.Sequential(
    torch.nn.Linear(input_width, hidden_width, bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=dropout),
    torch.nn.Linear(hidden_width, output_width, bias=True)
)
```
use
```py
depth = ... # your choice of the FFF depth
leaf_width = math.ceil(hidden_width / 2**depth)
region_leak # your choice of the region leak probability (0 - 0.5) to encourage generalisation in very deep FFFs

my_ff = FFF(
    input_width,
    leaf_width,
    output_width,
    depth,
    activation=torch.nn.ReLU(),
    dropout=dropout,
    region_leak=region_leak
)
```

Note that in order to get performance equal to that of a vanilla feedforward layer (FF) of width `hidden_width`, you might have to choose `leaf_width` and `depth` such that `2**depth * leaf_width > hidden_width`, i.e. such that the training width of the FFF will be larger than the training width of the FF.


## Documentation
Use `help(fastfeedforward.FFF)` to display the following documentation.

```
class FFF(torch.nn.modules.module.Module)
 |  FFF(input_width: int, leaf_width: int, output_width: int, depth: int, activation=ReLU(), dropout: float = 0.0, train_hardened: bool = False, region_leak: float = 0.0, usage_mode: str = 'none')
 |
 |  An implementation of fast feedforward networks from the paper "Fast Feedforward Networks".
 |
 |  Method resolution order:
 |      FFF
 |      torch.nn.modules.module.Module
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(self, input_width: int, leaf_width: int, output_width: int, depth: int, activation=ReLU(), dropout: float = 0.0, train_hardened: bool = False, region_leak: float = 0.0, usage_mode: str = 'none')
 |      Initializes a fast feedforward network (FFF).
 |
 |      Parameters
 |      ----------
 |      input_width : int
 |              The width of the input, i.e. the size of the last dimension of the tensor passed into `forward()`.
 |      leaf_width : int
 |              The width of each leaf of this FFF.
 |      output_width : int
 |              The width of the output, i.e. the size of the last dimension of the tensor returned by `forward()`.
 |      depth : int
 |              The depth of the FFF tree. Will result to 2**depth leaves.
 |      activation : torch.nn.Module, optional
 |              The activation function to use. Defaults to `torch.nn.ReLU()`.
 |      dropout : float, optional
 |              The probability to use for the dropout at the leaves after the activations have been computed. Defaults to 0.0.
 |              Plays no role if self.training is False.
 |      train_hardened : bool, optional
 |              Whether to use hardened decisions during training. Defaults to False.
 |      region_leak : float, optional
 |              The probability of a region to leak to the next region at each node. Defaults to 0.0.
 |              Plays no role if self.training is False.
 |      usage_mode : str, optional
 |              The mode of recording usage of the leaves and nodes of this FFF.
 |              Must be one of ['hard', 'soft, 'none']. Defaults to 'none'.
 |
 |      Raises
 |      ------
 |      ValueError
 |              - if `input_width`, `leaf_width` or `output_width` are not positive integers
 |              - if `depth` is not a positive integer or 0
 |              - if `dropout` is not in the range [0, 1]
 |              - if `region_leak` is not in the range [0, 1]
 |              - if `usage_mode` is not one of ['hard', 'soft, 'none']
 |
 |      Notes
 |      -----
 |      - The number of leaves of the FFF will be 2**depth.
 |      - The number of nodes of the FFF will be 2**depth - 1.
 |      - The region leak of >0.5 effectively reverses the roles of the left and right child at each node.
 |      - Dropout and region leaks are only applied during training (i.e. model.eval() will disable them).
 |
 |  eval_forward(self, x: torch.Tensor) -> torch.Tensor
 |      Computes the forward pass of this FFF during evaluation (i.e. making hard decisions at each node and traversing the FFF in logarithmic time).
 |
 |      Parameters
 |      ----------
 |      x : torch.Tensor
 |              The input tensor. Must have shape (..., input_width).
 |
 |      Returns
 |      -------
 |      torch.Tensor
 |              The output tensor. Will have shape (..., output_width).
 |
 |      Notes
 |      -----
 |      - Dropout and region leaks are not engaged by this method.
 |
 |  forward(self, x: torch.Tensor, return_entropies: bool = False, use_hard_decisions: Optional[bool] = None)
 |      Computes the forward pass of this FFF.
 |      If `self.training` is True, `training_forward()` will be called, otherwise `eval_forward()` will be called.
 |
 |      Parameters
 |      ----------
 |      x : torch.Tensor
 |              The input tensor. Must have shape (..., input_width).
 |      return_entropies : bool, optional
 |              Whether to return the entropies of the decisions made at each node. Defaults to False.
 |              If True, the mean batch entropies for each node will be returned as a tensor of shape (n_nodes,).
 |      use_hard_decisions : bool, optional
 |              Whether to use hard decisions during the forward pass. Defaults to None.
 |              If None and `self.training` is True, will effectively be False.
 |              If None and `self.training` is False, will effectively be True.
 |              Cannot be set to False if `self.training` is False.
 |
 |
 |      Returns
 |      -------
 |      torch.Tensor
 |              The output tensor. Will have shape (..., output_width).
 |      torch.Tensor, optional
 |              The mean batch entropies for each node. Will be returned with shape (n_nodes,) if `return_entropies` is True.
 |              Will not be returned if `return_entropies` is False.
 |
 |      Raises
 |      ------
 |      ValueError
 |              - if `x` does not have shape (..., input_width)
 |              - if `return_entropies` is True and `self.training` is False
 |              - if `use_hard_decisions` is False and `self.training` is False
 |
 |      See Also
 |      --------
 |      `training_forward()`
 |      `eval_forward()`
 |
 |  get_leaf_param_group(self) -> dict
 |      Returns the parameters of the leaves of this FFF, coupled with their usage tensor.
 |
 |      Returns
 |      -------
 |      dict
 |              The parameters of the leaves of this FFF, coupled with their usage tensor.
 |              Will have the following keys:
 |                      - "params": a list containing the leaf parameters
 |                      - "usage": the node usage tensor
 |
 |  get_node_param_group(self) -> dict
 |      Returns the parameters of the nodes of this FFF, coupled with their usage tensor.
 |
 |      Returns
 |      -------
 |      dict
 |              The parameters of the nodes of this FFF, coupled with their usage tensor.
 |              Will have the following keys:
 |                      - "params": a list containing the node parameters
 |                      - "usage": the node usage tensor
 |
 |  training_forward(self, x: torch.Tensor, return_entropies: bool = False, use_hard_decisions: bool = False)
 |      Computes the forward pass of this FFF during training.
 |
 |      Parameters
 |      ----------
 |      x : torch.Tensor
 |              The input tensor. Must have shape (..., input_width).
 |      return_entropies : bool, optional
 |              Whether to return the entropies of the decisions made at each node. Defaults to False.
 |              If True, the mean batch entropies for each node will be returned as a tensor of shape (n_nodes,).
 |      use_hard_decisions : bool, optional
 |              Whether to use hard decisions during the forward pass. Defaults to False.
 |              If True, the decisions will be rounded to the nearest integer. This will effectively make the FFF tree non-differentiable.
 |
 |      Returns
 |      -------
 |      torch.Tensor
 |              The output tensor. Will have shape (..., output_width).
 |      torch.Tensor, optional
 |              The mean batch entropies for each node. Will be returned with shape (n_nodes,) if `return_entropies` is True.
 |              Will not be returned if `return_entropies` is False.
 |
 |      Notes
 |      -----
 |      - The FFF tree is traversed from the root to the leaves.
 |              At each node, the input is multiplied by the node's weight matrix and added to the node's bias vector.
 |              The result is passed through a sigmoid function to obtain a probability.
 |              The probability is used to modify the mixture of the current batch of inputs.
 |              The modified mixture is passed to the next node.
 |              Finally, the outputs of all leaves are mixed together to obtain the final output.
 |      - If `use_hard_decisions` is True and `return_entropies` is True, the entropies will be computed before the decisions are rounded.
 |      - If self.training is False, region leaks and dropout will not be applied in this function.
 |      - Node usage, when tracked, is computed after node leaks have been applied (but is of course also applied when there is no node leaks).
 |
 |      Raises
 |      ------
 |      ValueError
 |              - if `x` does not have shape (..., input_width)
 |
 |      See Also
 |      --------
 |      `eval_forward()`
 |
 |  ----------------------------------------------------------------------
 |  The rest of the methods are inherited from torch.nn.modules.module.Module.
```