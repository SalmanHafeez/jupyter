# Project Jupyter Documentation

Welcome to the Project Jupyter documentation site. Jupyter is a large umbrella
project that covers many different software offerings and tools, including the
popular [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/latest/)
and [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) web-based
notebook authoring and editing applications. The Jupyter project and its
subprojects all center around providing tools (and [standards](https://docs.jupyter.org/en/latest/#sub-project-documentation))
for interactive computing with [computational notebooks](#what-is-a-notebook).

(what-is-a-notebook)=
## What is a Notebook?

![jupyterlab.png](_static/_images/jupyterlab.png)

**Pictured:** *A computational notebook document, shown inside JupyterLab*

> 📘 **Note:** Read [What is Jupyter?](what_is_jupyter) for a detailed look at Jupyter and notebooks.

A notebook is a shareable document that combines computer code, plain language
descriptions, data, rich visualizations like 3D models, charts, graphs and
figures, and interactive controls. A notebook, along with an editor (like
JupyterLab), provides a fast interactive environment for prototyping and
explaining code, exploring and visualizing data, and sharing ideas with
others.

## Where do I start?

Most people begin with Jupyter by installing an editing application that fits
their preferences, like [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)
or [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/latest/),
and making their first notebook document:

- Jupyter Notebook offers a simplified, lightweight notebook authoring experience
- JupyterLab offers a feature-rich, tabbed multi-notebook editing environment
  with additional tools like a customizable interface layout and system console
- And more... read about additional notebook interfaces [here](projects/user-interfaces)!

You can also develop your own extensions or applications on top of existing Jupyter
software. Check out the subproject sites below for more information.

## More information

These are a few high-level topics to help you learn more about the Jupyter community and ecosystem.
import numpy as np

def lu(A: np.ndarray):
    n = A.shape[0]
    L = np.eye(n)
    U = A  # Work directly on the same array

    for k in range(n - 1):
        if U[k, k] == 0.0:
            raise ZeroDivisionError("Zero pivot encountered.")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return L, U


def plu(A: np.ndarray):
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A  # Work directly on same matrix

    for k in range(n):
        pivot = k + np.argmax(np.abs(U[k:, k]))
        if np.isclose(U[pivot, k], 0.0):
            raise ZeroDivisionError("Matrix is singular.")

        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        L[k, k] = 1.0
        if k < n - 1:
            L[k+1:, k] = U[k+1:, k] / U[k, k]
            U[k+1:, k:] -= np.outer(L[k+1:, k], U[k, k:])
    return P, L, U
