import numpy as np
import matplotlib.pyplot as plt

def frobenius_error(X, W, H):
    WH = W @ H
    return np.sum((X - WH) ** 2)

X = np.array([
    [1,0,1,0,1,0],
    [0,1,0,1,0,0],
    [1,1,0,0,0,0],
    [0,0,0,0,1,1]
], dtype=float)

print("X = \n", X)
print("X shape = ", X.shape)


k = 2

# initialise the W and H matrices
rng = np.random.default_rng(0)
W = rng.random((X.shape[0], k))
H = rng.random((k, X.shape[1]))

eps = 1e-12 # avoid division by zero

print("starting W = \n", W)
print("starting H = \n", H)

print("starting F error = ", frobenius_error(X, W, H))


errors = []

def mur(X, W, H, max_iterations, eps=1e-12):
    for it in range(max_iterations):
        numerator_H = W.T @ X
        denominator_H = (W.T @ W @ H)

        H = H * numerator_H / (denominator_H + eps)

        numerator_W = X @ H.T
        denominator_W = (W @ H @ H.T)

        W = W * numerator_W / (denominator_W + eps)

        errors.append(frobenius_error(X, W, H))

    return W, H

W, H = mur(X, W, H, max_iterations=200)

print("final H = \n", np.round(H, 2))
print("final W = \n", np.round(W, 2))

print("final WH = \n", np.round((W @ H), 2))

print("final F error = ", frobenius_error(X, W, H))

W_norm = W / (W.sum(axis=1, keepdims=True) + eps)

print("W norm = \n", np.round(W_norm, 2))

# plot the errors

plt.plot(errors, marker='o')
plt.xlabel("Iterations")
plt.ylabel("Squared F Error")
plt.title("NMF Convergence")
plt.grid(True)
plt.show()

