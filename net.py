import numpy as np
from layers import *


class FullyConnectedNet(object):
	"""
		Model: A fully-connected neural network with an arbitrary number of hidden layers, 
		ReLU nonlinearities, and a softmax loss function. 
	"""

	def __init__(self, hidden_dims, input_dim=28*28, num_classes=10, reg=0.0, weight_scale=1e-2, dtype=np.float32):
		"""
		Inputs:
			- hidden_dims: A list of integers giving the size of each hidden layer.
			- input_dim: An integer giving the size of the input.
			- num_classes: An integer giving the number of classes to classify.
			- reg: Scalar giving L2 regularization strength.
			- weight_scale: Scalar giving the standard deviation for random initialization of the weights.
			- dtype: A numpy datatype object; all computations will be performed using this datatype. ( float32 is faster but less accurate than float64 )						 
		"""

		self.reg = reg
		self.num_layers = 1 + len(hidden_dims)
		self.dtype = dtype
		self.params = {}

		dims = [input_dim] + hidden_dims + [num_classes]  
		
		# Initialize weight and bias with weight_scale
		for i in xrange(self.num_layers):
			self.params['b%d' % (i+1)] = np.zeros(dims[i+1])
			self.params['W%d' % (i+1)] = np.random.randn(dims[i], dims[i+1]) * weight_scale
			
		
		# Cast all parameters to the correct datatype
		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)


	def loss(self, X, y=None):
		"""
		The loss function is to compute the total loss and grandient for fully-connected network
		
		if y is None ==> "test" mode, compute the total loss for input X ( minibatch ) through the DNN   

		if y is not None  ==> "train" mode, compute both loss and gradient for a minibatch of data.
			
		"""  

		X = X.astype(self.dtype)

		if y is None:
			mode = 'test'
		else:
			mode = 'train'

		
		layer = {}  # layer store the i-th layer's X, which will be the input of the next layer
		cache_layer = {}  # store input data of that layer for the purpose of computing the gradient by chain rule
		grads = {}	# store the gradient of each parameter (W and b)

		layer[0] = X # Initialization the first input as X (minibatch)
		for i in xrange(1, self.num_layers):
			layer[i], cache_layer[i] = affine_relu_forward(layer[i-1], self.params['W%d' % i], self.params['b%d' % i])

		# forward into last layer
		W_last = 'W%d' % self.num_layers
		b_last = 'b%d' % self.num_layers
		scores, cache_scores = affine_forward(layer[self.num_layers - 1], self.params[W_last], self.params[b_last])


		if mode == 'test':
			return scores


		loss, d_scores = loss_softmax(scores, y)

		# ========= Compute all gradients and add L2 regularization ==========
		for i in xrange(1, self.num_layers + 1): # add regularization loss:
			loss += 0.5 * self.reg * np.sum(self.params['W%d' % i]**2)

		# Backprop into last layer
		d_x = {}
		d_x[self.num_layers], grads[W_last], grads[b_last] = affine_backward(d_scores, cache_scores)
		grads[W_last] += self.reg * self.params[W_last]

		# Backprop into remaining layers
		for i in reversed(xrange(1, self.num_layers)):
			d_x[i], grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(d_x[i+1], cache_layer[i])
			grads['W%d' % i] += self.reg * self.params['W%d' % i]
		
		return loss, grads


