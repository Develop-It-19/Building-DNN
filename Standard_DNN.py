#Import Dependencies
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
 
#Initialize parameters of DNN
def initialize_parameters_deep(layer_dims):
  parameters = {}
  L = len(layer_dims)
  
  for l in range(1, L):
    parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
  return parameters
  
parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#Sigmoid and Relu Activation
def linear_activation_forward(A_prev, W, b, activation):
  if activation = "sigmoid"
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    A, activation_cache = sigmoid(Z)
    
  elif activation = "relu"
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    A, activation_cache = relu(Z)
    
  assert(A.shape == (W.shape[0], A_prev.shape[1]))
  cache = (linear_cache, activation_cache)
  
  return A, cache

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))

#Forward Propagation
def L_model_forward(X, parameters):
  cache = []
  A = X
  L = len(parameters) // 2       #Floor Division
  
  for l in range(1, L):
    A_prev = A
    A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = 'relu')
    caches.append(cache)
    
  AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
  caches.append(cache)
  
  assert(AL.shape == (1, X.shape[1]))
  return AL, caches
  
X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

#Cost Computation
def compute_cost(AL, Y):
  m = Y.shape[1]
  
  cost = -(1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
  cost = np.squeeze(cost)
  
  assert(cost.shape == ())
  
  return cost
  
Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))

#Back Propagation
def linear_backward(dZ, cache):
  A_prev, W, b = cache
  m = A_prev.shape[1]
  
  dW = np.dot(dZ, cache[0].T) / m
  db = np.squeeze(np.sum(dZ, axis = 1, keepdims = True)) / m
  dA_prev = np.dot(cache[1].T, dZ)
  
  assert(dA_prev.shape == A_prev.shape)
  assert(dW.shape == W.shape)
  assert(isinstance(db, float))
  
  return dA_prev, dW, db
  
dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

def linear_activation_backward(dA, cache, activation):
  linear_cache, activation_cache = cache
  
  if activation == "relu":
    dZ = relu_backward(dA, activation_cache)
    
  elif activation == "sigmoid":
    dZ = sigmoid_backward(dA, activation_cache)
    
  dA_prev, dW, db = linear_backward(dZ, linear_cache)
  
  return dA_prev, dW, db
  
AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
  
def L_model_backward(AL, Y, caches):
  grads = {}
  L = len(caches)
  m = AL.shape[1]
  Y = Y.reshape(AL.shape)
  
  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  
  current_cache = caches[-1]
  grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
  
  for l in reversed(range(L-1)):
    current_cache = cache[l]
    dA_prev_temp, dW_temp, db_temp = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
    grads["dA" + str(l + 1)] = dA_prev_temp
    grads["dW" + str(l + 1)] = dW_temp
    grads["db" + str(l + 1)] = db_temp
  
  return grads

X_assess, Y_assess, AL, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))

def update_parameters(parameters, grads, learning_rate):
  L = len(parameters) // 2
  
  for l in range(L):
    parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
    parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
  
  return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = " + str(parameters["W1"]))
print ("b1 = " + str(parameters["b1"]))
print ("W2 = " + str(parameters["W2"]))
print ("b2 = " + str(parameters["b2"]))
print ("W3 = " + str(parameters["W3"]))
print ("b3 = " + str(parameters["b3"]))
  
  
  
  
  
