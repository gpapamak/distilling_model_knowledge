# Distilling Model Knowledge

This project includes all the code from my MSc thesis at the University of Edinburgh:

> G. Papamakarios. _Distilling Model Knowledge_. MSc by Research Thesis, Centre for Doctoral Training in Data Science, University of Edinburgh. 2015.
[[pdf]](https://arxiv.org/pdf/1510.02437.pdf) [[bibtex]](http://homepages.inf.ed.ac.uk/s1459647/bibtex/distilling_model_knowledge.bib)

## How to get started

* In the main folder, run `install.m` to add all necessary paths to the matlab path.
* The various scripts in the folder `knowledge_distillation` run the experiments and visualize the results.

## How the code is organized

The thesis contains three main chapters, each showcasing a different application of knowledge distillation.

### Chapter 2: Model Compression

This chapter shows how to use knowledge distillation to compress large discriminative models, such as ensembles of neural nets, into smaller models that perform almost as well. The code for this chapter is in the folder `knowledge_distillation/neural_nets`.

### Chapter 3: Compact Predictive Distributions

This chapter uses knowledge distillation on MCMC-based Bayesian inference. It shows how to distill an MCMC chain into a small set of representative samples. The code for this chapter is in the folder `knowledge_distillation/bags_of_samples`.

### Chapter 4: Intractable Generative Models

This chapter shows how to distill generative models with intractable likelihoods, such as the Restricted Boltzmann Machine, into models with tractable likelihoods, such as the Neural Autoregressive Distribution Estimator. The code for this chapter is in the folder `knowledge_distillation/nade`.

Note that this chapter has been published separately as:

> G. Papamakarios and I. Murray. _Distilling Intractable Generative Models_. NIPS workshop on Probabilistic Integration, 2015.
[[pdf]](http://homepages.inf.ed.ac.uk/s1459647/papers/distilling_generative_models.pdf) [[bibtex]](http://homepages.inf.ed.ac.uk/s1459647/bibtex/distilling_generative_models.bib)

There is also a dedicated github project for the code of this paper [here](https://github.com/gpapamak/distilling_intractable_generative_models). Please refer to this if you are only interested in this chapter.



