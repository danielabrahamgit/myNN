import numpy as np

class myNN:

	"""
	Uses indices to denote a layer
	0 -> input layer
	xk+1 = act(yk), yk = Mk @ xk + bk
	"""
	def __init__(self, dims, act_types, loss_type='MSE', lr=1e-2, batch_size=1, rng=None):
		assert len(dims) >= 2
		assert len(act_types) == len(dims) - 1

		# Save constants
		self.lr = lr 
		self.B = batch_size
		self.N = len(act_types)
		self.input = None
		self.loss, self.dloss = myNN.get_loss(loss_type)

		# Random number gen
		if rng is None:
			self.rng = np.random
		else:
			self.rng = rng

		# Per layer parameters
		self.weights = []
		self.biases = []
		self.acts = []
		self.dacts = []
		self.ys = []
		self.xs = []
		self.grads = []

		# Populate
		for i in range(len(dims) - 1):
			weight_dims = (dims[i+1], dims[i])
			act, dact = myNN.get_activation(act_types[i])
			self.acts.append(act)
			self.dacts.append(dact)
			weights, bias = self.init_weights_and_bias(weight_dims) 
			self.weights.append(weights)
			self.biases.append(bias)
			self.xs.append(None)
			self.ys.append(None)
			self.grads.append(None)
		self.xs.append(None)
		self.grads.append(None)

	def forward(self, x, p=False):
		if len(x.shape) == 1:
			self.xs[0] = x.reshape((-1,self.B))
		else:
			self.xs[0] = x
			b_temp = x.shape[1]
		for i in range(self.N):
			self.ys[i] = self.weights[i] @ self.xs[i] + self.biases[i][:,:b_temp]
			self.xs[i+1] = self.acts[i](self.ys[i])
		return self.xs[self.N]

	def init_weights_and_bias(self, shape):
		assert len(shape) == 2
		L = np.sqrt(6 / (np.sum(shape)))
		weights = self.rng.uniform(-L, L, shape)
		bias = self.rng.uniform(-L, L, (shape[0], 1)).repeat(self.B, axis=1)
		bias = self.rng.uniform(-L, L, (shape[0], 1)).repeat(self.B, axis=1)
		return weights, bias

	def get_loss(loss_type):
		mse = lambda X, Y : np.mean((X - Y) ** 2)
		dmse = lambda X, Y: 2 * (X - Y) / X.shape[0]
		if loss_type == 'MSE':
			return mse, dmse
		else:
			print('Invalid Loss')
			quit()

	def get_activation(act_type):
		relu = lambda x : x * (x > 0)
		drelu = lambda x : 0 + (x > 0)
		sig = lambda x : 1 / (1 + np.exp(-x))
		dsig =  lambda x : sig(x) * (1 - sig(x))
		lin = lambda x : x 
		dlin = lambda x : np.ones(x.shape)
		if   act_type == 'relu':
			return relu, drelu
		elif act_type == 'sig':
			return sig, dsig
		elif act_type == 'lin':
			return lin, dlin
		else:
			print('Invalid Activation Function Keyword')
			quit()

	def compare_loss(self, true_out, back_prop=False):
		out = self.xs[self.N]
		e = self.loss(out, true_out)

		if back_prop:
			dedout = self.dloss(out, true_out)

			self.grads[self.N] = dedout

			for i in range(self.N)[::-1]:
				# print(f'layer={i}')

				# gradients
				# grad = dedx(i+1)
				dedy_i = self.dacts[i](self.ys[i]) * self.grads[i+1]
				# print(dedy_i.shape)
				dW = dedy_i @ self.xs[i].T
				db = dedy_i

				# update weights and biases ith layer
				self.weights[i] -= self.lr * dW / self.B
				self.biases[i] -= self.lr * db / self.B

				# update gradients 
				self.grads[i] = self.weights[i].T @ dedy_i
		return e
