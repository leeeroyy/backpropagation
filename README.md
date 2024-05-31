# Backpropagation through an Affine Linear Layer

This repository contains a `.pdf` document that explains the process of backpropagation through an affine linear layer in neural networks. The folder [tex_files](https://github.com/leeeroyy/backpropagation/tree/main/tex_files) contains the corresponding `.tex` documents.
## Content 
An affine linear layer takes as input a matrix $X \in \mathbb{R}^{N \times D}$ and produces an output $XW + B$ where $W \in \mathbb{R}^{D \times M}$ is the weight matrix and $B \in \mathbb{R}^{N \times M}$ is the bias matrix consisting of $N$ equal rows $b \in \mathbb{R}^{1 \times M}$. In order to optimize the parameters of the affine layer and potentially the layers prior to the layer, the partial derivatives of the loss function $L$ of the neural network with respect to the parameters of the affine layer and its input need to be determined. A Python function implementing the backward pass of an affine linear layer may look as follows:
```python
def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout) 
    self.db = np.sum(dout, axis=0)
    return dx
```
The `.pdf` details the mathematical derivations of the above implemented formulas. 

## Usage 

Make sure you have [Git](https://git-scm.com/downloads) installed, then copy and paste the following command in your terminal to copy the contents in your current working directory.

```bash
git clone https://github.com/leeeroyy/backpropagation.git
