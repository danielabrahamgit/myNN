import numpy as np
import matplotlib.pyplot as plt
from myNN import myNN

# ----------------- Parameters -----------------
n_train = 100000
n_test = 100
lr = 1e-3
dims = [2,1]
act_types = ['relu']
def f_true(x):
	a = 0 + (x[0] > 0.5)
	b = 0 + (x[1] > 0.5)
	return 0 + (a & b)

# ----------------- Benchmark -----------------
# Create neural net
nn = myNN(dims, act_types, lr=1e-3)

# Generate inputs
inps = np.random.randint(0, 2, (n_train, 2))

# Train, keep track of mses
mses = []
for i, inp in enumerate(inps):
	nn.forward(inp)
	mse = nn.compare_loss(f_true(inp), back_prop=True)
	if i % 100 == 0:
		mses.append(mse)

# Create grid to visualize network
x = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for r in range(100):
	for c in range(100):
		arr = np.array([X[r,c], Y[r,c]])
		Z[r,c] = nn.forward(arr).squeeze()

# ----------------- Plot -----------------
plt.subplot(121)
plt.plot(mses)
plt.subplot(122)
plt.imshow(Z, origin='lower')
plt.show()