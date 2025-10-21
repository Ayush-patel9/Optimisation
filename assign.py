import numpy as np
import matplotlib.pyplot as plt
from math import isclose
Q = np.array([
    [100., 20., 0., 0., 0.],
    [20., 50., 10., 0., 0.],
    [0., 10., 20., 5., 0.],
    [0., 0., 5., 10., 2.],
    [0., 0., 0., 2., 5.]
])
b = np.array([1., 2., 3., 4., 5.])

#  func given in question
def f(x):
    """Objective function f(x) = 1/2 x^T Q x - b^T x."""
    return 0.5 * x.dot(Q.dot(x)) - b.dot(x)

# defining grad of any func
def grad(x):
    """Gradient ∇f(x) = Qx - b."""
    return Q.dot(x) - b

def newton_step(g):
    # exact Newton step for quadratic
    return np.linalg.solve(Q, -g)

def cauchy_point(g):
    # steepest descent(Cauchy) step direction within trust region
    gQg = g.dot(Q.dot(g))
    if isclose(gQg, 0.0):
        return -g
    alpha = g.dot(g) / gQg
    return -alpha * g

def dogleg_step(g, Delta):
    # compute Dogleg step given gradient and trust region radius
    p_u = cauchy_point(g)
    p_b = newton_step(g)
    norm_pb, norm_pu = np.linalg.norm(p_b), np.linalg.norm(p_u)
    if norm_pb <= Delta:
        return p_b, 'Newton'
    if norm_pu >= Delta:
        return (Delta / norm_pu) * p_u, 'ScaledCauchy'
    # interpolate between p_u and p_b
    pb_minus_pu = p_b - p_u
    a = pb_minus_pu.dot(pb_minus_pu)
    b_q = 2 * p_u.dot(pb_minus_pu)
    c_q = p_u.dot(p_u) - Delta**2
    disc = b_q**2 - 4*a*c_q
    if disc < 0:
        return (Delta / norm_pu) * p_u, 'NumericalFallback'
    sqrt_disc = np.sqrt(disc)
    t1 = (-b_q + sqrt_disc) / (2*a)
    t2 = (-b_q - sqrt_disc) / (2*a)
    t_candidates = [t for t in (t1, t2) if 0 <= t <= 1]
    if not t_candidates:
        return (Delta / norm_pb) * p_b, 'ClampToPB'
    t = max(t_candidates)
    p = p_u + t * pb_minus_pu
    return p, 'Dogleg'


def trust_region_dogleg(x0, Delta0=1.0, Delta_max=100.0, eta1=0.25, eta2=0.75,tol=1e-6, maxiter=200):
    # trust Region (Dogleg) method
    x = x0.copy()
    Delta = Delta0
    hist = {'x': [], 'f': [], 'gnorm': [], 'Delta': [], 'step_norm': [], 'step_type': []}
    for k in range(maxiter):
        g = grad(x)
        gnorm = np.linalg.norm(g)
        fx = f(x)

        hist['x'].append(x.copy())
        hist['f'].append(fx)
        hist['gnorm'].append(gnorm)
        hist['Delta'].append(Delta)
        if gnorm < tol:
            break
        p, stype = dogleg_step(g, Delta)
        pred_red = -g.dot(p) - 0.5 * p.dot(Q.dot(p))
        if pred_red <= 0:
            Delta *= 0.25
            continue
        fx_new = f(x + p)
        act_red = fx - fx_new
        rho = act_red / pred_red
        if rho < eta1:
            Delta *= 0.25
        elif rho > eta2 and np.isclose(np.linalg.norm(p), Delta):
            Delta = min(2.0 * Delta, Delta_max)
        if rho > 0:
            x = x + p
            hist['step_norm'].append(np.linalg.norm(p))
            hist['step_type'].append(stype)
        else:
            hist['step_norm'].append(0.0)
            hist['step_type'].append('Rejected')
    return hist

def steepest_descent(x0, tol=1e-6, maxiter=200):
    # steepest Descent with exact line search for quadratic objective
    x = x0.copy()
    hist = {'x': [], 'f': [], 'gnorm': [], 'step_norm': [], 'alpha': []}
    for k in range(maxiter):
        g = grad(x)
        gnorm = np.linalg.norm(g)
        fx = f(x)
        hist['x'].append(x.copy())
        hist['f'].append(fx)
        hist['gnorm'].append(gnorm)
        if gnorm < tol:
            break
        gQg = g.dot(Q.dot(g))
        alpha = g.dot(g) / gQg if not isclose(gQg, 0.0) else 1.0
        p = -alpha * g
        x = x + p
        hist['step_norm'].append(np.linalg.norm(p))
        hist['alpha'].append(alpha)
    return hist

if __name__ == "__main__":
    x0 = np.zeros(5)
    tr_hist = trust_region_dogleg(x0, Delta0=1.0, Delta_max=100.0, tol=1e-8, maxiter=200)
    sd_hist = steepest_descent(x0, tol=1e-8, maxiter=200)
    tr_iters = np.arange(1, len(tr_hist['f']) + 1)
    sd_iters = np.arange(1, len(sd_hist['f']) + 1)

# func value vs iteration
    plt.figure()
    plt.plot(tr_iters, tr_hist['f'], 'o-', label='Trust Region (Dogleg)')
    plt.plot(sd_iters, sd_hist['f'], 'x-', label='Steepest Descent')
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)  # highlight iteration 1
    plt.text(1.1, min(tr_hist['f']), 'Converged at iter 1', fontsize=9, color='gray', va='bottom')
    plt.xlabel('Iteration')
    plt.ylabel('Function value f(x)')
    plt.title('Function Value vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig("function_values.png")

# Gradient norm (using linear y axis) 
    plt.figure()
    plt.plot(tr_iters, tr_hist['gnorm'], 'o-', label='Trust Region (Dogleg)')
    plt.plot(sd_iters, sd_hist['gnorm'], 'x-', label='Steepest Descent')
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)  # highlight iteration 1
    plt.text(1.1, max(tr_hist['gnorm']), 'Converged at iter 1', fontsize=9, color='gray', va='top')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient norm ||g||')
    plt.title('Gradient Norm vs Iteration (Linear Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig("gradient_norms_linear.png")

# Gradient norm (using (log y axis))
    plt.figure()
    plt.semilogy(tr_iters, tr_hist['gnorm'], 'o-', label='Trust Region (Dogleg)')
    plt.semilogy(sd_iters, sd_hist['gnorm'], 'x-', label='Steepest Descent')
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)  # highlight iteration 1
    plt.text(1.1, max(tr_hist['gnorm']), 'Converged at iter 1', fontsize=9, color='gray', va='top')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient norm ||g|| (log scale)')
    plt.title('Gradient Norm vs Iteration (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig("gradient_norms_log.png")

# trust region
    plt.figure()
    plt.plot(tr_iters, tr_hist['Delta'], 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Trust region radius Δ')
    plt.title('Trust Region Radius vs Iteration')
    plt.grid(True)
    plt.savefig("trust_region_radius.png")

# summary 
    print("\nSummary:->")
    print(f"Trust Region iterations: {len(tr_hist['f'])}")
    print(f"Steepest Descent iterations: {len(sd_hist['f'])}")
    print(f"Final TR f(x): {tr_hist['f'][-1]:.6e}")
    print(f"Final SD f(x): {sd_hist['f'][-1]:.6e}")
    print(f"Final TR ||g||: {tr_hist['gnorm'][-1]:.6e}")
    print(f"Final SD ||g||: {sd_hist['gnorm'][-1]:.6e}")
    print("Saved plots:")
    print(" - function_values.png")
    print(" - gradient_norms_linear.png")
    print(" - gradient_norms_log.png")
    print(" - trust_region_radius.png")
