# Backpropagation through an Affine Linear Layer

This repository contains a .pdf document that explains the process of backpropagation through an affine linear layer in neural networks. The formulas derived in the work
´´´math
    \[
    \frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^T
    \]
    \[
    \frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{Y}}
    \]
    \[
    \frac{\partial L}{\partial \mathbf{b}} = \mathbf{1}^T \frac{\partial L}{\partial \mathbf{Y}}
    \]
´´´math
A Python function demonstrating the backward pass method of an affine linear layer:
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

- **Python Implementation Example**:
  - 

## Usage

To compile the LaTeX document, you will need a LaTeX distribution installed on your system (e.g., TeX Live, MiKTeX). Compile the `.tex` file with the following command:

```bash
pdflatex backpropagation_affine_layer.tex
