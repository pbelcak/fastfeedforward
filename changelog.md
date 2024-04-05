# Change Log

## v0.2.1
 - one no longer has to specify `use_hard_decisions=True` when running forward under `model.training=False` (`ValueError` will no longer be thrown)
 - fixed a bug that occurred for some PyTorch versions where torch.matmul broadcasting across two-dimensional batch size resulted in greedy memory allocation and unreasonable CUDA OOMs
 - using `index_select` in `forward_eval` was fast but too memory-consuming; reduced it to a for-loop until `.foreach` or custom CUDA implementation is available

## v0.2.0
### FFF
 - renamed the `__init__` parameter `hidden_width` to `leaf_width` (yes, this disrespects semantic versioning a little, but hey ...)
 - introduced the `region_leak` parameter
 - introduced the `usage_mode` parameter
 - added `get_node_param_group` and `get_leaf_param_group` methods to enable connection to `LocalSGD` and `LocalAdam` with ease
### fastfeedforward.optim
 - added as a separate package
 - contains implementations of the `LocalSGD` and `LocalAdam`

## v0.1.0
This is the initial version with a basic implementation of the fast feedforward module and with an example notebook.