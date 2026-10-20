# Weibull Survival Regression from First Principles

A from-scratch implementation of Weibull regression for right-censored lifetime data, applied to predictive maintenance on the AI4I 2020 dataset. Every likelihood, gradient, and Hessian entry is derived analytically and implemented in NumPy, with no autograd or survival analysis libraries.

## Motivation

In a fleet of 10,000 machines observed over a study window, only 339 fail. A standard regression on failure times would either discard the 9,661 machines still running (selection bias) or treat them as failures (underestimated lifetimes). Survival analysis handles this correctly by writing a likelihood that uses exactly what was observed: an exact failure time for some machines, and a lower bound on lifetime for the rest.

## Model

Each observed time follows a Weibull distribution whose shape parameter depends on sensor readings through an exponential link:

- $t_i \sim \text{Weibull}(\alpha, \beta_i)$
- $\beta_i = \exp(\gamma^\top x_i)$

The scale $\alpha$ acts as the fleet baseline lifetime, while $\beta_i$ captures the machine's aging regime: decreasing hazard ($\beta < 1$), constant hazard ($\beta = 1$), or wear-out ($\beta > 1$). Each coefficient $\gamma_j$ has a direct interpretation: a unit increase in sensor $j$ multiplies $\beta_i$ by $e^{\gamma_j}$.

## What's Implemented

- **Censored negative log-likelihood** combining failure densities and survival probabilities via a censoring indicator
- **Analytical gradient and Hessian**, derived by hand and translated line by line into code
- **Newton-Raphson optimizer** using `np.linalg.solve` for numerical stability
- **Gradient descent baseline** for comparison

## Key Results

- **Convergence:** Newton-Raphson converges in under 10 iterations, while gradient descent needs thousands to reach the same tolerance. Both reach the same estimates, as expected from the strict convexity of the likelihood under a full-rank design matrix.
- **Hessian singularity:** Including a global intercept alongside fully one-hot encoded categorical variables makes the design matrix rank-deficient, producing a singular Hessian and inflated standard errors. Since the scale parameter $\alpha$ already anchors the model, the intercept is redundant. Dropping it restores full rank and yields stable, significant estimates.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020): 10,000 machines with sensor readings including air temperature, process temperature, rotational speed, and torque.

## Usage

```python
theta_hat = newton_raphson(X, t, delta)
```

- `X`: (n, p) sensor matrix, **no intercept column**
- `t`: (n,) observed failure or censoring times
- `delta`: (n,) failure indicator (1 = failure, 0 = censored)
- `theta_hat`: estimated parameters $[\alpha, \gamma_1, \dots, \gamma_p]$

## Report

The full derivation, including every gradient and Hessian entry and the convergence analysis, is available in the accompanying report.

## Author

Chahat Ahuja, Summer Research Intern at Chennai Mathematical Institute under Prof. Sourish Das
