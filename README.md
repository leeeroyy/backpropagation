# Backpropagation through an Affine Linear Layer

This repository contains a `.pdf` document that explains the process of backpropagation through an affine linear layer in neural networks.

## Content 
An affine linear layer takes as input a matrix $X \in \mathbb{R}^{N \times D}$ and produces an output $XW + B$ where $W \in \mathbb{R}^{D \times M}$ is the weight matrix and $B \in \mathbb{R}^{N \times M}$ is the bias matrix consisting of $N$ equal rows $b \in \mathbb{R}^{1 \times M}$. To optimize the parameters of the affine linear layer and potentially the layers prior to the affine layer, the partial derivatives of the loss function $L$ of the neural network with respect to the parameters of the affine layer and its input need to be determined. A Python function implementing the backward pass of an affine linear layer may look as follows:
```python
def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout) 
    self.db = np.sum(dout, axis=0)
    return dx
```
The `.pdf` details the mathematical derivations of the above implemented formulas. 
