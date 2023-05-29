# Learning the right layers: a data-driven layer-aggregation strategy for semi-supervised learning on multilayer graphs

## Learning_the_right_layers
Jupyter notebook divided in sections:
- Accuracy functions: functions to calculate the accuracy of the final partition.
- x_sol - solution lower level problem : function to solve the lower level problem with parametric Label Propagation algorithm.
- ZOFW - Zeroth order Frank Wolfe: function to apply the Frank Wolfe inexact algorithm to solve the upper level problem. 
- Cross_entropy - loss function upper level problem: definitions of the binomial cross-entropy loss and the multiclass cross-entropy loss (optimized in the upper level problem).
- Parallelization functions: functions which are parallelized in the code. cross_entropy_c applies ZOFW to optimize the binomial cross-entropy loss on a single community; multistart perform in parallel cross_entropy_c on each community; multistart_multi applies ZOFW to optimize the  multiclass cross-entropy loss; methods performs the proposed methods. 
- Datasets: synthetic_datasets function to perform tests on synthetic datasets, info_datasets to print the information of the real datsets, real_datasets to perform tests on real datasets, real_datasets_noisy to perform tests on real datasets with adding noisy layers. 
- Tests: functions to perform the tests in the paper.
- Print results: show the results.
- Computational Analysis: perform the computational analysis in the paper.
- Tables: create the tables in the paper from the results.
- Metrics: calculates the metrics in the paper (APR and AR)


## Matlab files
- state_of_art_methods: applies the state-of-the-art methods over to same synthetic datasets.
- state_of_art_methods_real: applies the state-of-the-art methods to the real datasets.
- state_of_art_methods_exec_times: applies the state-of-the-art methods for the the computational analysis. 
- Utils: contains the functions used to calculate the accuracy of the final partition (confusion_matrix calculates the confusion matrix, reindex_com reindexes communities, wrong counts the number of nodes in the wrong community). 

## Datasets
- Synthetic: contains the synthetic datasets reported in the paper, generated using synthetic_datasets.
- Real: contains the real datasets reported in the paper, generated using real_datasets and real_datasets_noisy.\
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

