# Learning the right layers: a data-driven layer-aggregation strategy for semi-supervised learning on multilayer graphs
This repository contains the codes of the paper "Learning the right layers: a data-driven layer-aggregation strategy for semi-supervised learning on multilayer graphs" by Sara Venturini, Andrea Cristofari, Francesco Rinaldi, Francesco Tudisco.

## Learning_the_right_layer
Jupyter notebook divided in sections:
- Accuracy functions: functions to calculate the accuracy of the final partition.
- x_sol - solution lower level problem : function to solve the lower level problem with parametric Label Propagation algorithm.
- ZOFW - Zeroth order Frank Wolfe: function to apply the Frank Wolfe inexact algorithm to solve the upper level problem. 
- Cross_entropy - loss function upper level problem: definitions of the binomial cross-entropy loss and the multiclass cross-entropy loss (optimized in the upper level problem).
- Parallelization functions: functions which are parallelized in the code. cross_entropy_c applies ZOFW to optimize the binomial cross-entropy loss on a single community; multistart perform in parallel cross_entropy_c on each community; multistart_multi applies ZOFW to optimize the  multiclass cross-entropy loss; methods performs the proposed methods. 
- Datasets: synthetic_datasets function to perform tests on synthetic datasets, info_datasets to print the information of the real datsets, real_datasets to perform tests on real datasets, real_datasets_noisy to perform tests on real datasets with adding noisy layers. 
- results_statistics: function to print the average results. 
- Tests: functions to perform the tests in the paper.

## Matlab files
- state_of_art_methods: applies the state-of-the-art methods over to same synthetic datasets.
- state_of_art_methods_real: applies the state-of-the-art methods to the real datasets.
- Utils: contains the functions used to calculate the accuracy of the final partition (confusion_matrix calculates the confusion matrix, reindex_com reindexes communities, wrong counts the number of nodes in the wrong community). 

## Datasets
- Synthetic: contains the synthetic datasets reported in the paper, generated using synthetic_datasets.
- Real: \
From https://github.com/melopeo/PM_SSL/tree/master/realworld_datasets \
P. Mercado, F. Tudisco, and M. Hein, Generalized Matrix Means for Semi-Supervised Learning with Multilayer Graphs. In NeurIPS 2019.
From https://bitbucket.org/uuinfolab/20csur/src/master/ \
Magnani, M., Hanteer, O., Interdonato, R., Rossi, L., & Tagarelli, A. (2021). Community detection in multiplex networks. ACM Computing Surveys (CSUR), 54(3), 1-35.

## State-of-the-art methods 
### Download
- From http://www-als.ics.nitech.ac.jp/~karasuyama/software/SMGI/SMGI.html: Sparse Multiple Graph Integration (SGMI).
- From https://github.com/kylejingli/AMGL-IJCAI16: Auto-weighted Multiple Graph Learning (AGML).
- From https://github.com/egujr001/SMACD: Semi-supervised Multi-Aspect Community Detection (SMACD).
- From https://github.com/melopeo/PM_SSL: Generalized Matrix Means (GMM). 

## Reference paper
"Learning the right layers: a data-driven layer-aggregation strategy for semi-supervised learning on multilayer graphs" by Sara Venturini, Andrea Cristofari, Francesco Rinaldi, Francesco Tudisco.

## Authors
- Sara Venturini (e-mail: sara.venturini@math.unipd.it)
- Andrea Cristofari (e-mail: andrea.cristofari@uniroma2.it)
- Francesco Rinaldi (e-mail: rinaldi@math.unipd.it)
- Francesco Tudisco (e-mail: francesco.tudisco@gssi.it)
