import numpy as np

class myNN:

	"""
	Uses indices to denote a layer
	0 -> input layer
	xk+1 = act(yk), yk = Mk @ xk + bk
	"""
	def __init__(self, dims, act_types, loss_type='MSE', lr=1e-2):
		assert len(dims) >= 2
		assert len(act_types) == len(dims) - 1

		self.weights = []
		self.biases = []
		self.acts = []
		self.dacts = []
		self.ys = []
		self.xs = []
		self.grads = []

		for i in range(len(dims) - 1):
			shape = (dims[i], dims[i+1])
			act, dact = myNN.get_activation(act_types[i])
			self.acts.append(act)
			self.dacts.append(dact)
			self.weights.append(myNN.init_matrix((shape[1], shape[0])))
			self.biases.append(myNN.init_matrix(shape[1]))
			self.xs.append(np.zeros(shape[0]))
			self.ys.append(np.zeros(shape[1]))
			self.grads.append(np.zeros(shape))
		self.xs.append(np.zeros(shape[1]))
		self.grads.append(np.zeros(shape[1]))

		self.loss, self.dloss = myNN.get_loss(loss_type)
		self.N = len(self.weights)
		self.input = None
		self.lr = lr

	def forward(self, x):
		self.xs[0] = x 
		for i in range(self.N):
			self.ys[i] = self.weights[i] @ self.xs[i] + self.biases[i]
			self.xs[i+1] = self.acts[i](self.ys[i])
		return self.xs[self.N]

	def init_matrix(shape):
		return np.random.uniform(-1,1,shape)

	def get_loss(loss_type):
		mse = lambda x, y : np.mean((x - y) ** 2)
		dmse = lambda x, y: 2 * (x - y) / len(x)
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

	def dy_dmat(self, layer_ind):
		# i -> rows
		# j -> cols 
		# k -> out_ind
		ni, nj = self.weights[layer_ind].shape
		nk = len(self.ys[layer_ind])
		deriv = np.zeros((ni, nj, nk))
		i = np.arange(ni)
		deriv[i,:,i] = self.xs[layer_ind]
		return deriv

	def dy_db(self, layer_ind):
		deriv = np.eye(len(self.biases[layer_ind]))
		return deriv

	# If k is layer_ind, then
	# we want dx[k+1]/dy[k]
	def dx_dy(self, layer_ind):
		dgnal = self.dacts[layer_ind](self.ys[layer_ind])
		return np.diag(dgnal)

	# If k is layer_ind, the
	# we want dy[k]/dx[k]
	def dy_dx(self, layer_ind):
		return self.weights[layer_ind]

	def compare_loss(self, true_out, back_prop=False):
		out = self.xs[self.N]
		e = self.loss(out, true_out)

		if back_prop:
			dedout = self.dloss(out, true_out)

			self.grads[self.N] = dedout

			for i in range(self.N)[::-1]:

				# gradients
				dxdy = self.dx_dy(i)
				dydW = self.dy_dmat(i)
				dydb = self.dy_db(i)
				grad = self.grads[i+1]

				dW = dydW @ dxdy @ grad
				db = dydb @ dxdy @ grad

				# update weights and biases ith layer
				self.weights[i] -= self.lr * dW
				self.biases[i] -= self.lr * db

				# print(self.weights[i].shape, dxdy.shape, grad.shape)
				# update gradients 
				self.grads[i] = self.weights[i].T @ dxdy @ grad

		return e


