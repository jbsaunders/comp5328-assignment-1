import numpy as np
import matplotlib.pyplot as plt

def l21_norm(V, W, H): # Finds the L_2,1 Norm - this goes step by step but is slower
    V_col_count = V.shape[1]
    print("V col count = ", V_col_count)
    total = 0.0
    for j in range(V_col_count):
        v_j = V[:, [j]]
        # r_j = V[j]
        # print("v_j = \n", v_j)
        h_j = H[:, [j]]
        r_j = v_j - W @ h_j
        # print("r_j = \n", r_j)
        norm = np.linalg.norm(r_j)
        # print(norm)
        total += norm
    
    return total

def l21_norm_fast(V, W, H): # this is a faster way of finding the L_2,1 Norm
    R = V - W @ H
    return np.linalg.norm(R, axis=0).sum(0) #axis =0 means we go column by column

V = np.array([
    [1,0,1,0,1,0],
    [0,1,0,1,0,0],
    [1,1,0,0,0,0],
    [0,0,0,0,1,1]
], dtype=float)

print("V = \n", V)
print("V shape = ", V.shape)


k = 2

rng = np.random.default_rng(0)
W = rng.random((V.shape[0], k))
H = rng.random((k, V.shape[1]))

eps = 1e-12 # avoid division by zero
delta = 1e-12

print("starting W = \n", W)
print("starting H = \n", H)


fast_norm = l21_norm_fast(V, W, H)
print("fast norm = ", fast_norm)

def compute_U(V, W, H, eps=1e-12):
    R = V - W @ H
    col_norms = np.linalg.norm(R, axis=0) # gets a vector of all the norms of the columns of R
    u = 1.0 / np.maximum(col_norms, eps) # inverts them
    U = np.diag(u) # converts the vector into a diagonal Matrix
    # print(U)
    return U


errors = []

def l21_norm_mul(V, W, H, max_iterations, eps=1e-12, delta=1e-12):
    for it in range(max_iterations):
        U = compute_U(V, W, H, eps=eps)

        numerator_H = W.T @ (V @ U)
        denominator_H = W.T @ (W @ H @ U) + delta

        H = H * numerator_H / denominator_H
        H = np.maximum(H, 0) # clips it to keep it non-negative

        numerator_W = (V @ U) @ H.T
        denominator_W = (W @ H @ U) @ H.T + delta

        W = W * numerator_W / denominator_W
        W = np.maximum(W, 0) # clips it to keep it non-negative

        errors.append(l21_norm_fast(V, W, H))

    return W, H

W, H = l21_norm_mul(V, W, H, max_iterations=200)


print("final H = \n", np.round(H, 2))
print("final W = \n", np.round(W, 2))

print("final WH = \n", np.round((W @ H), 2))

print("final l21 error = ", l21_norm_fast(V, W, H))

W_norm = W / (W.sum(axis=1, keepdims=True) + eps) # normalises W to make the values easier to interpret

print("W norm = \n", np.round(W_norm, 2))

# plot the errors

plt.plot(errors, marker='o')
plt.xlabel("Iterations")
plt.ylabel("L_2,1 Error")
plt.title("NMF Convergence")
plt.grid(True)
plt.show()

