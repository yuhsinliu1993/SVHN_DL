import numpy as np


def affine_forward(x, w, b):
	"""
		Computes the forward pass for an affine (fully-connected) layer.

		Input:
			- x: input data with shape (N, D)   N: number of training data    D: dimensions
			- w: weights with shape (D, M)
			- b: biases with shape (M, )

		Return:
			- out: output matrx with shape (N, M)
			- cache: store the input data (for backprop purpose)
	"""
	out = np.dot(x, w) + b
	cache = (x, w, b)

	return out, cache


def affine_backward(d_out, cache):
	"""
		Computes the backward pass for an affine layer.

		Input:
			d_out: Upstream derivative of shape(N, M)
			cache: (x, w, b) input of this layer
	"""
	x, w, b = cache

	d_x = np.dot(d_out, w.T).reshape(x.shape)   # (N, M) x (M, D)
	d_w = np.dot(x.T, d_out)   # (D, N) x (N, M)
	d_b = np.sum(d_out, axis=0)

	return d_x, d_w, d_b


def relu_forward(x):
	out = np.maximum(x, 0)
	
	return out, x


def relu_backward(d_out, cache):
	x = cache
	d_x = np.array(d_out, copy=True)
	d_x[x <= 0] = 0

	return d_x


def affine_relu_forward(x, w, b):
	a, fc_cache = affine_forward(x, w, b)
	
	# ReLU forward
	out, relu_cache = relu_forward(a)
	cache = (fc_cache, relu_cache)

	return out, cache


def affine_relu_backward(d_out, cache):
	fc_cache, relu_cache = cache
	
	# ReLU backward
	fc_cache, relu_cache = cache
	da = relu_backward(d_out, relu_cache)	
	d_x, d_w, d_b = affine_backward(da, fc_cache)

	return d_x, d_w, d_b


def loss_softmax(x, y):
	N = x.shape[0]

	# numerical stability 
	p = np.exp(x - np.max(x, axis=1, keepdims=True))
	
	p = p / np.sum(p, axis=1, keepdims=True)
	loss = -np.sum(np.log(p[np.arange(N), y])) / N
	
	d_x = p.copy()
	d_x[np.arange(N), y] -= 1
	d_x /= N

	return loss, d_x

