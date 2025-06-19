import numpy as np
import os
import matplotlib.pyplot as plt

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)  # Convert data to numpy array
        self.dim = np.shape(self.data) #store the dimensions
        self.grad = np.zeros_like(self.data)  # Initialize gradient with zeros
        self._backward = lambda: None #store the backward function
        self.children = set(_children) #store the set of children
        self.op = _op #store the operation

    def __repr__(self):
        return f"Tensor(dim={self.dim}, data={repr(self.data)})"
    
    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            if len(out.grad.shape) > len(self.grad.shape):
                shape = (1, ) * (len(out.grad.shape)- len(self.grad.shape)) + self.grad.shape
            else:
                shape = self.grad.shape
            # Sum out.grad to the shape of self.grad to handle broadcasting
            self.grad += np.sum(out.grad, axis=tuple(d for d, s in enumerate(shape) if s == 1), keepdims=len(out.grad.shape) == len(self.grad.shape))
            
            if len(out.grad.shape) > len(other.grad.shape):
                shape = (1, ) * (len(out.grad.shape)- len(other.grad.shape)) + other.grad.shape
            else:
                shape = other.grad.shape
            # Sum out.grad to the shape of other.grad to handle broadcasting
            other.grad += np.sum(out.grad, axis=tuple(d for d, s in enumerate(shape) if s == 1), keepdims=len(out.grad.shape) == len(other.grad.shape))
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out  # Return the tensor with the backward function attached
    
    def __truediv__(self, other):
        out = Tensor(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad -= self.data / np.power(other.data, 2) * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        out = Tensor(self.data + other.data, (self, other), '-')
        def _backward():
            if len(out.grad.shape) > len(self.grad.shape):
                shape = (1, ) * (len(out.grad.shape)- len(self.grad.shape)) + self.grad.shape
            else:
                shape = self.grad.shape
            # Sum out.grad to the shape of self.grad to handle broadcasting
            self.grad += np.sum(out.grad, axis=tuple(d for d, s in enumerate(shape) if s == 1), keepdims=len(out.grad.shape) == len(self.grad.shape))
            
            if len(out.grad.shape) > len(other.grad.shape):
                shape = (1, ) * (len(out.grad.shape)- len(other.grad.shape)) + other.grad.shape
            else:
                shape = other.grad.shape
            # Sum out.grad to the shape of other.grad to handle broadcasting
            other.grad -= np.sum(out.grad, axis=tuple(d for d, s in enumerate(shape) if s == 1), keepdims=len(out.grad.shape) == len(other.grad.shape))
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            # For C = A @ B:
            # dC/dA = dC @ B.T
            # dC/dB = A.T @ dC
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out  # Return the tensor with the backward function attached
    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - np.power(out.data, 2)) * out.grad
        out._backward = _backward
        return out
    
    def pow(self, power = 1):
        out = Tensor(np.power(self.data, power), (self,), '**')
        def _backward():
            self.grad += power * np.power(self.data, power-1) * out.grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')
        def _backward():
            # Create a gradient of the correct shape and distribute the incoming gradient
            self.grad += out.grad * np.ones_like(self.data) / self.data.size[axis]
        out._backward = _backward
        return out
    def mean(self, axis=None, keepdims=False):
        """
        Computes the mean of the tensor's data along the specified axis.
        """
        # Forward pass
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            # Determine the number of elements that were averaged
            if axis is None:
                num_elements = self.data.size
            else:
                num_elements = self.data.shape[axis]
            self.grad += out.grad * np.ones_like(self.data) / num_elements
            
        out._backward = _backward
        return out
    def softmax(self, axis = 1):
        exps = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True)) # a trick for numerical stability
        return Tensor(exps / np.sum(exps, axis=axis, keepdims=True), _children=(self,), _op='softmax')

    def cross_entropy(self, y_true):
        # Softmax for numerical stability
        exp_x = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        # Cross-entropy loss calculation
        loss = -np.sum(y_true.data * np.log(softmax_output + 1e-9))/ y_true.data.shape[0] # Add a small epsilon for numerical stability
        
        out = Tensor(loss, (self, ),'cross_entropy')
        
        def _backward():
            # Gradient of cross-entropy with softmax
            grad = softmax_output - y_true.data
            self.grad += out.grad.data * grad / y_true.data.shape[0]

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
    
    