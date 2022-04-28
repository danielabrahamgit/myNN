import numpy as np
import matplotlib.pyplot as plt
from myNN import myNN

# ----------------- Parameters -----------------
n_train = 100000
n_test = 100
lr = 1e-3
dims = [2,7,7,3,1]
act_types = ['relu', 'relu', 'relu', 'sig']
def f_true(x):
	# 1 if on disc, 0 else
	r_max = 0.8
	r_min = 0.2 
	r = np.linalg.norm(x) 
	return 0 + (r_min < r) * (r < r_max)

# ----------------- Benchmark -----------------
# Create neural net
nn = myNN(dims, act_types, lr=1e-3)

# Generate inputs
inps = np.random.uniform(-1, 1, (n_train, 2))

# Train, keep track of mses
mses = []
for i, inp in enumerate(inps):
	nn.forward(inp)
	mse = nn.compare_loss(f_true(inp), back_prop=True)
	if i % (n_train//100) == 0:
		mses.append(mse)

# Create grid to visualize network
x = np.linspace(-1, 1, n_test)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for r in range(n_test):
	for c in range(n_test):
		arr = np.array([X[r,c], Y[r,c]])
		Z[r,c] = nn.forward(arr).squeeze()

# ----------------- Plot -----------------
plt.subplot(121)
plt.plot(mses)
plt.subplot(122)
plt.imshow(Z, origin='lower', extent=[-1,1]*2)
plt.show()