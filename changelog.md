# Change Log

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