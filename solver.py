from optimization import adam, sgd
import numpy as np


class Solver(object):

	def __init__(self, model, data, update_rule='sgd', batch_size=100, num_epochs=100, optim_configs={}):
		self.model = model
		self.X_train = data['X_train']
		self.y_train = data['y_train']
		self.X_val = data['X_val']
		self.y_val = data['y_val']

		self.update_rule = update_rule
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.optim_configs = optim_configs

		self.epoch = 0
		self.best_val_acc = 0
		self.best_params = {}
		self.loss_history = []
		self.train_err_history = []
		self.val_err_history = []

		# the configs of optimization for each weight and bias
		for p in self.model.params.keys():
			self.optim_configs[p] = {key: value for key, value in self.optim_configs.iteritems()}


	def _step(self):
		# Make a minibatch of training data
		num_train = self.X_train.shape[0]
		batch_mask = np.random.choice(num_train, self.batch_size)
		X_batch = self.X_train[batch_mask]
		y_batch = self.y_train[batch_mask]

		# Compute loss and gradient
		loss, grads = self.model.loss(X_batch, y_batch)
		self.loss_history.append(loss)

		# Perform parameters update
		for key, val in self.model.params.items():
			d_val = grads[key]	 # key: W1, b1, W2...
			config = self.optim_configs[key]

			if self.update_rule == 'sgd':
				next_w, next_config= sgd(val, d_val, config)
			elif self.update_rule == 'adam':
				next_w, next_config = adam(val, d_val, config)
			else:
				raise ValueError("Undefined update rule %s" % self.update_rule)

			self.model.params[key] = next_w   	  # updated weights
			self.optim_configs[key] = next_config   # updated optimization's config


	def train(self):
		num_train = self.X_train.shape[0]
		num_batches = num_train / self.batch_size
		iterations_per_epoch = max(num_batches, 1)  # 1 iter : using one batch to train the model
			  									    # 1 epoch : using number of batches (i.e. num of iterations) to train the model
		num_iterations = self.num_epochs * iterations_per_epoch

		for t in xrange(num_iterations):
			self._step()

			if ((t + 1) % iterations_per_epoch) == 0: 	# Finish an epoch
				self.epoch += 1

				train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)  # to save time, make a sub-sample of X_train
				val_acc = self.check_accuracy(self.X_val, self.y_val)

				# Record the error history for each epoch
				self.train_err_history.append(1.0 - train_acc)
				self.val_err_history.append(1.0 - val_acc)

				print '(Epoch %d / %d) train_acc: %f  val_acc: %f  loss: %f' % (self.epoch, self.num_epochs, train_acc, val_acc, self.loss_history[-1])

				# Keep track of the best model
				if val_acc > self.best_val_acc:
					self.best_val_acc = val_acc
					self.best_params = {}
					for k, v in self.model.params.iteritems():
						self.best_params[k] = v.copy()

		self.model.params = self.best_params


	def check_accuracy(self, X, y, num_samples=None, batch_size=100):
		N = X.shape[0]

		if num_samples is not None and N > num_samples:
			mask = np.random.choice(N, num_samples)
			N = num_samples
			X = X[mask]
			y = y[mask]


		num_batches = N / batch_size
		if N % batch_size != 0:
			num_batches += 1

		y_pred = []
		for i in xrange(num_batches):
			start = i * batch_size
			end = (i + 1) * batch_size
			scores = self.model.loss(X[start: end])
			y_pred.append(np.argmax(scores, axis=1))

		y_pred = np.hstack(y_pred)
		acc = np.mean(y_pred == y)

		return acc
