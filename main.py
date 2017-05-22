from net import FullyConnectedNet
from solver import Solver
import numpy as np
import scipy.io as sio


# Load the data
loaded_data = sio.loadmat('SVHN.mat')

# === Preprocessing the data ===
X_train, y_train, X_test, y_test = loaded_data['train_x'], loaded_data['train_label'].argmax(axis=1), \
								   loaded_data['test_x'], loaded_data['test_label'].argmax(axis=1)

# Split the data into train, validation sets
num_training = 44000
num_validation = 1000

mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

data = {'X_val': X_val , 'X_train': X_train, 'y_val': y_val, 'y_train': y_train}

# ======== Building && Training models ========
solver_0 = FullyConnectedNet(hidden_dims=[100, 100, 50], weight_scale=5e-2)
solver = Solver(model=solver_0, data=data, num_epochs=500, batch_size=200, update_rule='sgd', learning_rate=1e-2)
solver.train()


solver_1 = FullyConnectedNet(hidden_dims=[100, 100, 50], weight_scale=5e-2)
solver = Solver(solver_1, data, num_epochs=500, batch_size=200, update_rule='adam', learning_rate=1e-3)
solver.train()


# # SGD with regularization = 5
# solver_2 = FullyConnectedNet(hidden_dims=[100, 100, 50], weight_scale=5e-2, reg=5)
# solver = Solver(solver_2, data, num_epochs=500, batch_size=200, update_rule='sgd', learning_rate=1e-2)
# solvers['sgd1'] = solver
# solver.train()


# # Adam with regularization = 5
# solver_3 = FullyConnectedNet(hidden_dims=[100, 100], weight_scale=5e-2, reg=5)
# solver = Solver(solver_3, data, num_epochs=500, batch_size=200, update_rule='adam', learning_rate=1e-3)
# solvers['adam1'] = solver
# solver.train()


# ======== Testing Time ========
# SGD without regularization:
y_test_pred = np.argmax(solver_0.loss(X_test), axis=1)
y_val_pred = np.argmax(solver_0.loss(X_val), axis=1)
print 'SGD: validation set accuracy (without regularization): ', (y_val_pred == y_val).mean()
print 'SGD: test set accuracy (without regularization): ', (y_test_pred == y_test).mean()


# Adam without regularization:
y_test_pred = np.argmax(solver_1.loss(X_test), axis=1)
y_val_pred = np.argmax(solver_1.loss(X_val), axis=1)
print 'ADAM: validation set accuracy (without regularization): ', (y_val_pred == y_val).mean()
print 'ADAM: test set accuracy (without regularization): ', (y_test_pred == y_test).mean()

