
# A Fast Neural Network Tool for Big Data Analysis



# Introduction
 
This package implements a deep neural network (MLP) for big data analysis. The remarkable feature of this package is, it estimates the effect size of each variable in an interpretable way, for example, G(aX+b, G is an activation function, and a is effect size) to represent an nonlinear interpretation of neural networks. It is good at dealing with nonlinear problems, particularly for genomic studies. There are multiple functions that can be extended to other Omics data analysis. This package adopted optimized algorithms such as backpropagation, Rprop, simulated annealing, stochastic gradient and pruning algorithms such as minimum magnitude, Optimal Brain Surgeon based on a fast implementation of compressed neural networks (Grzegorz Klima, 2016).
      
Welcome any [feedback](https://github.com/xinghuq/DNN/issues) and [pull request](https://github.com/xinghuq/DNN/pulls).  


## Install the package
```{R}
## Install from CRAN

# install.packages("DNN")

## Install the developement version from github

devtools::install_github("xinghuq/DNN")

library("DNN")

```

## Citation

Qin. X, Jia. P. 2022. A Fast Neural Network Tool for Big Data Analysis. R package version 1.2.1.

