import numpy as np
import matplotlib.pyplot as plt
from myNN import myNN

# ----------------- Parameters -----------------
n_train = 1000000
n_test = 100
dims = [2,7,7,1]
act_types = ['relu', 'relu', 'sig']
lr = 1e-1
B = 256
n_train = B * (n_train // B)
rng = np.random.default_rng(2012)
rng = np.random
def f_true(x):
	a = 0 + (x[0,:] > 0.5)
	b = 0 + (x[1,:] > 0.5)
	return ((a ^ b) + 0).reshape((-1,B))

# ----------------- Benchmark -----------------
# Create neural net
nn = myNN(
		dims=dims, 
		act_types=act_types, 
		lr=lr,
		batch_size=B,
		rng=rng)

# Generate inputs
inps = rng.binomial(1, 0.5, (2, n_train))

# Train, keep track of mses
mses = []
for i in range(0, n_train, B):
	inp = inps[:,i:i+B]
	nn.forward(inp)
	mse = nn.compare_loss(f_true(inp), back_prop=True)
	mses.append(mse)
mses = np.array(mses)

# Create grid to visualize network
x = np.linspace(-1, 1, n_test)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for r in range(n_test):
	for c in range(n_test):
		arr = np.array([[X[r,c]], [Y[r,c]]])
		output = nn.forward(arr,p=True)
		Z[r,c] = output

# ----------------- Plot -----------------
# plt.rcParams['font.size'] = '20'
plt.figure(figsize=(7,3))
plt.subplot(121)
plt.title('Loss vs MiniBatch')
plt.plot(mses)  
plt.subplot(122)
plt.imshow(Z, origin='lower', extent=[-1, 1] * 2)
plt.show()