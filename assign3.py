import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# %% SCRIPT 1: Trust Region (Dogleg) vs Steepest Descent (Quadratic)
# -------------------------------------------------------------------
print("--- Running Script 1: TR (Dogleg) vs Steepest Descent ---")

# %% Given data
Q = np.array([
    [100, 20, 0, 0, 0],
    [20, 50, 10, 0, 0],
    [0, 10, 20, 5, 0],
    [0, 0, 5, 10, 2],
    [0, 0, 0, 2, 5]
])
b = np.array([1, 2, 3, 4, 5])

# Objective function and gradient
def f(x, Q, b):
    return 0.5 * x.T @ Q @ x - b.T @ x

def g(x, Q, b):
    return Q @ x - b

# %% Parameters
x0 = np.zeros(5)
maxIter = 100
tol = 1e-6

# %% ---------------- TRUST REGION (Dogleg) ----------------
Delta = 1.0        # initial trust region radius
Delta_max = 10.0   # maximum radius
eta = 0.1          # acceptance threshold

x = x0.copy()
fvals_TR = []
grad_TR = []
Delta_hist = []
iter_TR = 0

for k in range(maxIter):
    grad = g(x, Q, b)
    B = Q  # Hessian is constant
    
    fvals_TR.append(f(x, Q, b))
    grad_TR.append(np.linalg.norm(grad))
    Delta_hist.append(Delta)
    iter_TR += 1

    if np.linalg.norm(grad) < tol:
        print(f'Trust Region converged at iteration {k+1}')
        break

    # Newton and Cauchy steps
    pN = np.linalg.solve(B, -grad)
    gBg = grad.T @ B @ grad
    if gBg <= 0: # Handle non-positive definiteness (though not for this Q)
        pC = - (Delta / np.linalg.norm(grad)) * grad
    else:
        pC = -((grad.T @ grad) / gBg) * grad


    # Dogleg step selection
    if np.linalg.norm(pN) <= Delta:
        p = pN
    elif np.linalg.norm(pC) >= Delta:
        p = (Delta / np.linalg.norm(pC)) * pC
    else:
        a = pN - pC
        b_vec = pC
        # Solve ||pC + tau*a||^2 = Delta^2 for tau
        coeffs = [a.T @ a, 2 * a.T @ b_vec, b_vec.T @ b_vec - Delta**2]
        tau = np.max(np.roots(coeffs))
        p = pC + tau * a

    # Compute reduction ratio
    pred_red = -(grad.T @ p + 0.5 * p.T @ B @ p)
    f_new = f(x + p, Q, b)
    act_red = f(x, Q, b) - f_new
    
    # Avoid division by zero if pred_red is tiny
    if pred_red < 1e-12:
        rho = 0
    else:
        rho = act_red / pred_red

    # Accept or reject step
    if rho > eta:
        x = x + p

    # Update trust region radius
    if rho < 0.25:
        Delta = 0.25 * Delta
    elif rho > 0.75 and np.abs(np.linalg.norm(p) - Delta) < 1e-8:
        Delta = min(2 * Delta, Delta_max)

x_TR = x
print(f'\nTrust Region completed in {iter_TR} iterations.')

# %% ---------------- STEEPEST DESCENT ----------------
x = x0.copy()
fvals_SD = []
grad_SD = []
iter_SD = 0

for k in range(maxIter):
    grad = g(x, Q, b)
    fvals_SD.append(f(x, Q, b))
    grad_SD.append(np.linalg.norm(grad))
    iter_SD += 1

    if np.linalg.norm(grad) < tol:
        print(f'Steepest Descent converged at iteration {k+1}')
        break

    # Exact line search for quadratic
    alpha = (grad.T @ grad) / (grad.T @ Q @ grad)
    x = x - alpha * grad

x_SD = x
print(f'\nSteepest Descent completed in {iter_SD} iterations.')

# %% ---------------- FINAL VALUES ----------------
f_TR = f(x_TR, Q, b)
f_SD = f(x_SD, Q, b)

print('\nFinal Trust Region x* = \n', np.round(x_TR, 6))
print('Final Steepest Descent x* = \n', np.round(x_SD, 6))

print(f'Final function value (TR) = {f_TR:.6f}')
print(f'Final function value (SD) = {f_SD:.6f}')
print(f'||grad(x_TR)|| = {np.linalg.norm(g(x_TR, Q, b)):.3e}')
print(f'||grad(x_SD)|| = {np.linalg.norm(g(x_SD, Q, b)):.3e}')

print(f'Difference between x_TR and x_SD = {np.linalg.norm(x_TR - x_SD):.3e}')
print(f'Difference between function values = {np.abs(f_TR - f_SD):.3e}\n')

# %% ---------------- PLOTS ----------------
plt.figure(figsize=(12, 10))

# Plot 1: Function Value vs Iteration
plt.subplot(3, 1, 1)
plt.plot(range(1, iter_TR + 1), fvals_TR, '-o', linewidth=1.2, label='Trust Region')
plt.plot(range(1, iter_SD + 1), fvals_SD, '-x', linewidth=1.2, label='Steepest Descent')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Function Value vs Iteration')
plt.legend()
plt.grid(True)

# Plot 2: Gradient Norm vs Iteration
plt.subplot(3, 1, 2)
plt.plot(range(1, iter_TR + 1), grad_TR, '-o', linewidth=1.2, label='Trust Region')
plt.plot(range(1, iter_SD + 1), grad_SD, '-x', linewidth=1.2, label='Steepest Descent')
plt.xlabel('Iteration')
plt.ylabel('||grad||')
plt.title('Gradient Norm vs Iteration')
plt.legend()
plt.grid(True)

# Plot 3: Trust Region Radius vs Iteration
plt.subplot(3, 1, 3)
plt.plot(range(1, iter_TR + 1), Delta_hist, '-o', linewidth=1.2, color='green')
plt.xlabel('Iteration')
plt.ylabel('Trust Region Radius (Δ)')
plt.title('Trust Region Radius vs Iteration')
plt.grid(True)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# %% SCRIPT 2: Basic Trust Region Method (Dogleg, Non-Quadratic)
# -------------------------------------------------------------------
print("\n\n--- Running Script 2: TR (Dogleg) for Non-Quadratic ---")

# --- Define objective function, gradient, and Hessian ---
# f = (x+y-1)^4 + 1/2(x-1)^2 + 1/2(y+2)^2 + 0.3xy
def f_script2(x_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    t = x1 + x2 - 1
    return t**4 + 0.5*(x1-1)**2 + 0.5*(x2+2)**2 + 0.3*x1*x2

def g_script2(x_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    t = x1 + x2 - 1
    grad = np.array([
        4*t**3 + (x1-1) + 0.3*x2,
        4*t**3 + (x2+2) + 0.3*x1
    ])
    return grad

def B_script2(x_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    t = x1 + x2 - 1
    B = 12*t**2 * np.array([[1, 1], [1, 1]]) + np.array([[1, 0.3], [0.3, 1]])
    return B

# --- Parameters ---
x = np.array([-1.2, 1.0])
maxIter = 100
Delta = 2.0           # initial trust region radius
Delta_max = 5.0       # maximum trust region radius
eta = 0.1             # acceptance threshold

for k in range(maxIter):
    # Evaluate function, gradient, Hessian
    f_val = f_script2(x)
    g_val = g_script2(x)
    B_val = B_script2(x)

    # Stop if gradient small
    g_norm = np.linalg.norm(g_val)
    if g_norm < 1e-6:
        print(f'Converged in {k+1} iterations')
        break

    # --- Dogleg Step ---
    try:
        # Newton step
        pN = np.linalg.solve(B_val, -g_val)
    except np.linalg.LinAlgError:
        # Fallback if Hessian is singular
        pN = -g_val 

    # Cauchy (steepest descent) point
    gBg = g_val.T @ B_val @ g_val
    if gBg <= 0:
        # Non-positive definite, just move along gradient
        tau = Delta / g_norm
    else:
        tau = (g_val.T @ g_val) / gBg
    
    pC = -tau * g_val

    # Dogleg step selection
    if np.linalg.norm(pN) <= Delta:
        p = pN
    elif np.linalg.norm(pC) >= Delta:
        p = (Delta / np.linalg.norm(pC)) * pC
    else:
        # interpolate between pC and pN
        a = pN - pC
        b_vec = pC
        coeffs = [a.T @ a, 2 * a.T @ b_vec, b_vec.T @ b_vec - Delta**2]
        roots_t = np.roots(coeffs)
        t = np.max(roots_t.real) # Take the real part of the largest root
        p = pC + t * a
    
    # --- End Dogleg Step ---

    # Predicted reduction
    pred_red = -(g_val.T @ p + 0.5 * p.T @ B_val @ p)

    # Actual reduction
    x_new = x + p
    f_new = f_script2(x_new)
    act_red = f_val - f_new

    # Ratio
    if pred_red < 1e-12:
        rho = 0
    else:
        rho = act_red / pred_red

    # Update x if acceptable
    if rho > eta:
        x = x_new

    # Update trust region radius
    if rho < 0.25:
        Delta = 0.25 * Delta
    elif rho > 0.75 and np.abs(np.linalg.norm(p) - Delta) < 1e-10:
        Delta = min(2 * Delta, Delta_max)
        
    # Ensure Delta is not too small
    Delta = max(Delta, 1e-8) 

    print(f'Iter {k+1:3d}: f = {f_val:+.6f}, ||g||={g_norm:.3e}, Delta={Delta:.2f}, rho={rho:.2f}')

if k == maxIter - 1:
    print(f'Reached max iterations ({maxIter}) without converging.')

print(f'\nFinal solution: x = [{x[0]:.6f}, {x[1]:.6f}]')
print(f'Final f(x) = {f_script2(x):.6f}')
print(f'Final ||g(x)|| = {np.linalg.norm(g_script2(x)):.3e}')