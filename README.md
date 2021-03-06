# Organisation of the code 

This repository provides the code for the paper Sampling from Arbitrary Functions via PSD Models (https://hal.inria.fr/hal-03386544v2). 

The necessary dependencies are simply a recent verison of torch. 

### Main file

The file containing the main tools is the file [gaussian_psd_model.py](gaussian_psd_model.py). In particular, the class `GaussianPSDModel1` implements the rank one Gaussian PSD models. Moreover, the following functions are used in order to perform the learning phase through a preconditioned conjugate gradient descent. 

### Computing models 

The files [experiment_MMD_distance_d_2.py](experiment_MMD_distance_d_2.py) computes all the necessary models and metrics in order to analyse the evolution of the MMD distance when trying to sample from a density in dimension $5$; in particular, it computes samples generated by the different methods we are interested in : ours, the gridding method, and the uniform sampling.

The files [experiment_hellinger_distance.py](experiment_hellinger_distance.py) and [experiment_hellinger_distance_hard.py](experiment_hellinger_distance_hard.py) compute all the necessary models and metrics in order to analyse the evolution of the Hellinger distance in two cases : the "simple one" in dimension $10$ but for a density which is a PSD model, and the hard one for a density which is in dimension 2 but not a PSD model.


### Creating figures

The file [experiment_effect_width_sampling_2.py](experiment_effect_width_sampling_2.py) contains all the necessary code to compute the figure on the sampling algorihtm (with hypercubes, for different values of $\rho$.
The file [hellinger_distance_graph.py](hellinger_distance_graph.py) computes the graph which performs an empirical evaluation of the Hellinger distance between the learnt and target density as $n$ increases for different values of $m$.
The file [mmd_distance_graph.py](MMD_distance_graph.py) compiles the main figure in the experiments section, learning a 2D density as well as evaluating the MMD distance.


### Additional code of interest 

The file [experiment_non_smooth.py](experiment_non_smooth.py) creates and samples a PSD model approximating a non-smooth density.

The file [experiment_algorithm_cost.py](experiment_algorithm_cost.py) is a test file to check that the sampling algorithm's bottleneck is indeed the erf and matrix product computations, thus checking the announced complexity.





