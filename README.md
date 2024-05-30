# Backpropagation through an Affine Linear Layer

This repository contains a .pdf document that explains the process of backpropagation through an affine linear layer in neural networks.

## Content 
An affine linear layer takes as input a matrix $X \in \mathbb{R}^{N \times D}$ and produces an output $XW + B$ where $W \in \mathbb{R}^{D \times M}$ is the weight matrix and $B \in \mathbb{R}^{N \times M}$ is a bias matrix containing $N$ equal rows $b \in \mathbb{R}^{1 \times M}$. To optimize the parameters of the affine linear layer and the layers prior to the affine layer, the partial derivatives of the loss function $L$ w.r.t. the parameters of the affine layer and its input need to be determined. A Python function implementing the backward pass of an affine linear layer may look as follows:
```python
def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout) 
    self.db = np.sum(dout, axis=0)
    return dx
```
    
## Content

### Pdf Document

The LaTeX document, `backpropagation_affine_layer.tex`, covers the following topics:

- **Introduction to Affine Linear Layers**: 
  - Explanation of how an affine linear layer operates, including the input and output matrices.
  
- **Derivation of Parameter Update Formulas**:
  - Derivation of the partial derivatives of the loss function with respect to the input matrix, weight matrix, and bias vector.
  - The key formulas derived are:
    \[
    \frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^T
    \]
    \[
    \frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{Y}}
    \]
    \[
    \frac{\partial L}{\partial \mathbf{b}} = \mathbf{1}^T \frac{\partial L}{\partial \mathbf{Y}}
    \]


## Usage

To compile the LaTeX document, you will need a LaTeX distribution installed on your system (e.g., TeX Live, MiKTeX). Compile the `.tex` file with the following command:

```bash
pdflatex backpropagation_affine_layer.tex
