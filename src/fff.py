import torch

from torch import nn
import math

def compute_entropy_safe(p, minus_p):
	EPSILON = 1e-6
	p = torch.clamp(p, min=EPSILON, max=1-EPSILON)
	minus_p = torch.clamp(minus_p, min=EPSILON, max=1-EPSILON)

	return -p * torch.log(p) - minus_p * torch.log(minus_p)

class FFF(nn.Module):
	"""
	An implementation of fast feedforward networks from the paper "Fast Feedforward Networks".
	"""
	def __init__(self, input_width, hidden_width, output_width, depth, activation=nn.ReLU(), dropout=0.0, train_hardened=False):
		super().__init__()
		self.input_width = input_width
		self.hidden_width = hidden_width
		self.output_width = output_width
		self.dropout = dropout
		self.activation = activation
		self.train_hardened = train_hardened

		self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.long), requires_grad=False)
		self.n_leaves = 2 ** depth
		self.n_nodes = 2 ** depth - 1

		l1_init_factor = 1.0 / math.sqrt(self.input_width)
		self.node_weights = nn.Parameter(torch.empty((self.n_nodes, input_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
		self.node_biases = nn.Parameter(torch.empty((self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

		l2_init_factor = 1.0 / math.sqrt(self.hidden_width)
		self.w1s = nn.Parameter(torch.empty((self.n_leaves, input_width, hidden_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
		self.b1s = nn.Parameter(torch.empty((self.n_leaves, hidden_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
		self.w2s = nn.Parameter(torch.empty((self.n_leaves, hidden_width, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
		self.b2s = nn.Parameter(torch.empty((self.n_leaves, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
		self.leaf_dropout = nn.Dropout(dropout)

	def training_forward(self, x, return_entropies=False, use_hard_decisions=False):
		# x has shape (batch_size, input_width)
		original_shape = x.shape
		x = x.view(-1, x.shape[-1])
		batch_size = x.shape[0]

		hard_decisions = use_hard_decisions or self.train_hardened
		current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
		entropies = None if not return_entropies else torch.zeros((batch_size, self.n_nodes), dtype=torch.float, device=x.device)

		for current_depth in range(self.depth.item()):
			platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
			next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

			n_nodes = 2 ** current_depth
			current_weights = self.node_weights[platform:next_platform]	# (n_nodes, input_width)	
			current_biases = self.node_biases[platform:next_platform]	# (n_nodes, 1)

			boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))		# (batch_size, n_nodes)
			boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)
			boundary_effect = torch.sigmoid(boundary_plane_logits)								# (batch_size, n_nodes)
			not_boundary_effect = 1 - boundary_effect											# (batch_size, n_nodes)

			if return_entropies:
				platform_entropies = compute_entropy_safe(
					boundary_effect, not_boundary_effect
				) # (batch_size, n_nodes)
				entropies[:, platform:next_platform] = platform_entropies	# (batch_size, n_nodes)
				
			if hard_decisions:
				boundary_effect = torch.round(boundary_effect)				# (batch_size, n_nodes)
				not_boundary_effect = 1 - boundary_effect					# (batch_size, n_nodes)
			
			mixture_modifier = torch.cat(
				(not_boundary_effect.unsqueeze(-1), boundary_effect.unsqueeze(-1)),
				dim=-1
			).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)												# (batch_size, n_nodes*2, 1)
			current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes))	# (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
			current_mixture.mul_(mixture_modifier)															# (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
			current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)								# (batch_size, self.n_leaves)
			del mixture_modifier, boundary_effect, not_boundary_effect, boundary_plane_logits, boundary_plane_coeff_scores, current_weights, current_biases

		element_logits = torch.matmul(
			x.unsqueeze(1).unsqueeze(1),		# (batch_size, 1, 1, self.input_width)
			self.w1s.view(1, *self.w1s.shape)	# (1, self.n_leaves, self.input_width, self.hidden_width)
		) # (batch_size, self.n_leaves, 1, self.hidden_width)
		element_logits += self.b1s.view(1, *self.b1s.shape).unsqueeze(-2)					# (batch_size, self.n_leaves, 1, self.hidden_width)
		element_activations = self.activation(element_logits)								# (batch_size, self.n_leaves, 1, self.hidden_width)
		element_activations = self.leaf_dropout(element_activations)						# (batch_size, self.n_leaves, 1, self.hidden_width)
		new_logits = torch.matmul(element_activations, self.w2s.view(1, *self.w2s.shape))	# (batch_size, self.n_leaves, 1, self.output_width)
		new_logits = new_logits.squeeze(-2)													# (batch_size, self.n_leaves, self.output_width)
		new_logits += self.b2s.unsqueeze(0)													# (batch_size, self.n_leaves, self.output_width)

		new_logits *= current_mixture.unsqueeze(-1)			# (batch_size, self.n_leaves, self.output_width)
		final_logits = new_logits.sum(dim=1)				# (batch_size, self.output_width)
		
		final_logits = final_logits.view(*original_shape[:-1], self.output_width)	# (..., self.output_width)

		if not return_entropies:
			return final_logits
		else:
			return final_logits, entropies.mean(dim=0)
		
	def forward(self, x, return_entropies=False):
		if self.training:
			return self.training_forward(x, return_entropies=return_entropies)
		else:
			if return_entropies:
				raise ValueError("Cannot return entropies during evaluation.")
			return self.eval_forward(x)

	def eval_forward(self, x):
		original_shape = x.shape
		x = x.view(-1, x.shape[-1])
		batch_size = x.shape[0]
		# x has shape (batch_size, input_width)

		current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
		for i in range(self.depth.item()):
			plane_coeffs = self.node_weights.index_select(dim=0, index=current_nodes)		# (batch_size, input_width)
			plane_offsets = self.node_biases.index_select(dim=0, index=current_nodes)		# (batch_size, 1)
			plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))		# (batch_size, 1, 1)
			plane_score = plane_coeff_score.squeeze(-1) + plane_offsets						# (batch_size, 1)
			plane_choices = (plane_score.squeeze(-1) >= 0).long()							# (batch_size,)

			platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)			# (batch_size,)
			next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device)	# (batch_size,)
			current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform	# (batch_size,)

		leaves = current_nodes - next_platform				# (batch_size,)
		w1s = self.w1s.index_select(dim=0, index=leaves)	# (batch_size, input_width, hidden_width)
		b1s = self.b1s.index_select(dim=0, index=leaves)	# (batch_size, hidden_width)
		w2s = self.w2s.index_select(dim=0, index=leaves)	# (batch_size, hidden_width, output_width)
		b2s = self.b2s.index_select(dim=0, index=leaves)	# (batch_size, output_width)

		logits = torch.matmul(
			x.unsqueeze(1),		# (batch_size, 1, self.input_width)
			w1s					# (batch_size, self.input_width, self.hidden_width)
		) 										# (batch_size, 1, self.hidden_width)
		logits += b1s.unsqueeze(-2)				# (batch_size, 1, self.hidden_width)
		activations = self.activation(logits)	# (batch_size, 1, self.hidden_width)
		new_logits = torch.matmul(activations, w2s)		# (batch_size, 1, self.output_width)
		new_logits = new_logits.squeeze(-2)				# (batch_size, self.output_width)
		new_logits += b2s								# (batch_size, self.output_width)

		return new_logits.view(*original_shape[:-1], self.output_width)	# (..., self.output_width)

		