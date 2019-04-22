============
Introduction
============


Estimagic is a Python package that helps to build high-quality and user friendly implementations of (structural) econometric models.

It is designed with large structural models in mind. However, it is also useful for any other estimator that numerically minimizes or maximizes a criterion function (Extremum Estimator). Examples are maximum likelihood estimation, generalized method of moments, method of simulated moments and indirect inference.

The highlights of estimagic are:

Estimation
==========

Collect a wide range methods for global and local optimization with a unified interface. This interface makes it easy to specify complex termination criteria and common types of constraints in an intuitive way - even for optimization algorithms that are originally unconstrained.

On top of the optimizer collection we provide methods to diagnose if a (global) optimization was successful. Moreover, we make it easy to monitor or pause and restart long running optimizations and to save intermediate results.


Inference
=========

Calculate several types of standard errors and confidence intervals for any extremum estimator.

1) asymptotic standard errors based on gradients and hessians.
2) computationally feasible bootstrap methods such as the wild score boostrap.
3) confidence intervals based on the profile-likelihood method.



Automate the boring stuff
=========================

Provide re-usable code blocks that make it faster to write structural models. These include but are not limited to helpers for the following tasks:

- Building criterion functions out of moment conditions
- Process model specifications
- Writing LaTeX tables
- Specify and estimate common auxiliary models for indirect inference
- Model diagnostics

Give you freedom
================

Using estimagic for estimation and inference only requires criterion functions with
a compatible interface. In contrast to other packages there are no further requirements.
For example, we do not ask you to subclass anything or to follow a certain programming
style.

Given these conditions, users can fully abstract from the details of optimization and numerical differentiation and be sure that they are using state of the art methods. Of course, it is also possible to configure every detail of the algorithms if desired.








