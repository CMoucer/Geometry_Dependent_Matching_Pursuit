# Geometry-dependent matching pursuit

This project aims at studying greedy first order methods, such as steepest coordinate descent or matching pursuit,
that have become popular for solving sparse optimization problems.  We develop a principled approach to matching pursuit 
adapted to the geometry of problems at hand, offering a novel strategy for regularized problems as well as refined 
convergence guarantees. 

These convergence guarantees let appear parameters characterizing the class of functions and the geometries of
optimization problems, but remain mostly intractable. To this end, we develop a priori estimates of convergence rates
for matching pursuit applied to a particular least-squares problem. To this end, we compute a priori exact estimates
using SDP relaxations, and random approximations using random matrix theory. 

Building on this result, we observe a  phase transition in the convergence of gradient descent and steepest coordinate 
descent, depending on the number of samples and the dimension. We experimentally highlight a transition phase for 
regularized matching pursuit and the proximal gradient method on a LASSO problem, depending on the value of the 
regularization parameter. Finally, we derive an ultimate method converging linearly in the underparametrized and the 
overparametrized regime, but is nonetheless not a matching pursuit algorithm.

This code allows to graw figures from the paper : **Geometry-dependent matching pursuit: a transition phase for
convergence on linear regression and LASSO** (to come) - C. Moucer, A. Taylor, F. Bach


## Getting Started


### Prerequisites
This codes relies on the following python modules, with versions provided in requirements.txt:

- numpy
```
pip install numpy
```
- cvxpy
```
pip install cvxpy
```
- matplotlib
```
pip install matplotlib
```
- scikit-learn
```
pip install -U scikit-learn
```

We recommend to use the following solver:
- solver MOSEK 
```
pip install mosek
```
Otherwise:
- solver SCS (already installed in cvxpy)

## Content of this folder
Our code is divided into four folders:
- **algorithms**: this folder contains files for computing gradient descent and steepest coordinate descent applied on
least-squares, as well as the proximal gradient method, the regularized matching pursuit, and the ultimate method
applied on the LASSO.
- **generate_data**: this file contains files for generating random data, in particular synthetic gaussian random
variables, sparse optimum, loader of real datasets (the Leukemia dataset, breast-cancer), and random features.
- **utils**: this folder contains the code for generating epsilon-curves and computing approximates for smoothness,
strong convexity and Lojasiewicz parameters.
- **figures**: this folder contains all the codes for plotting the figures of the paper.

Finally, the file ``` main.py ``` provide an example of how to use the algorithms on least-squares and the LASSO. The file
``` main_compare_mus.py ``` computes the (possibly approximate) values for smoothness, strong-convexity and Losiewicz parameters.



## Authors
* **Céline MOUCER** 
* **Adrien B. TAYLOR**
* **Francis BACH** 

## References

* [On Matching Pursuit and Coordinate Descent](https://proceedings.mlr.press/v80/locatello18a/locatello18a.pdf) - F. Locatello, A. Raj, S. Karimireddy, G. Raetsch, B. Schölkopf, S. Stich and M. Jaggi
* [Coordinate Descent Converges Faster with the Gauss-Southwell Rule Than Random Selection](https://proceedings.mlr.press/v37/nutini15.pdf) - J. Nutini, M. Schmidt, I. Laradji, M. Friedlander and H. Koepke
* [Geometry-dependent matching pursuit: a transition phase for
convergence on linear regression and LASSO]() - C. Moucer, A. Taylor, F. Bach