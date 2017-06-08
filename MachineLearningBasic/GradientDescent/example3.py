# check convergence
def has_converged(theta_new, grad):
	return np.linalg.norm(grad(theta_new)) /
							len(theta_new) < 1e-3

def GD_momentum(theta_init, grad, eta, gamma):
	# Suppose we want to store history of theta
	theta = [theta_init]
	v_old = np.zeros_like(theta_init)
	for it in range(100):
		v_new = gamma * v_old + eta * grad(theta[-1])
		theta_new = theta_new - v_new
		if has_converged(theta_new, grad):
			break
		theta.append(theta_new)
		v_old = v_new
	return theta
def GD_NAG(w_init, grad, eta, gamma):
	w = [w_init]
	v = [np.zeros_like(w_init)]
	for it in range(100):
		v_new = gamma * v[-1] + eta * grad(w[-1] - gamma * v[-1])
		w_new = w[-1] - v_new
		if has_converged(w_new, grad):
			break
		w.append(w_new)
		v.append(v_new)
	return (w, it)