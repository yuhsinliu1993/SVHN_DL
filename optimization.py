import numpy as np


def sgd(w, dw, config=None):

	if config is None:
		config = {}
	
	config.setdefault('learning_rate', 1e-2)

	next_w = w - config['learning_rate'] * dw
	return next_w, config


def adam(w, dw, config=None):
	
	if config is None:
		config = {}

	config.setdefault('learning_rate', 1e-3)
	config.setdefault('beta1', 0.9)
	config.setdefault('beta2', 0.999)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('m', np.zeros_like(w))
	config.setdefault('v', np.zeros_like(w))
	config.setdefault('t', 0)

	beta1, beta2, epsilon, m, v, t = config['beta1'], config['beta2'], config['epsilon'], config['m'], config['v'], config['t']

	t = t + 1
	m = beta1 * m + (1 - beta1) * dw
	v = beta2 * v + (1 - beta2) * (dw**2)

	# bias correction 
	m_b = m / (1 - beta1**t)
	v_b = v / (1 - beta2**t)

	next_x = - config['learning_rate'] * m_b / (np.sqrt(v_b) + epsilon) + w
	config['m'], config['v'], config['t'] = m, v, t

	return next_x, config


