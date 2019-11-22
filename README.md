# Semi-parametric Learning for Heteroscedastic Robust Regression 

## General Framework
1. Given noisy data X, we first optimize 2xN free parameters for alpha and c
    - TODO: Experiment with different training schemes
        - N alpha, 1 c
        - 1 alpha, N c
        - N alpha, N c
        - EM-like procedure where we update alternatively
2. Fit models for alpha and c (this process works somewhat like a regularization)
    - Based on prior knowledge about heteroscedasticity, we decide either parametric models or non-parametric models for alpha and c
        - TODO: show trade-off (model misspecification vs variance)
        - TODO: 3 baseline cases: GLM (exponential), Spline GLM, and non-parametric
3. Simulate posterior with MCMC using the general robust kernel but locally differentiating parameters for alpha from 2
    - If alpha and c is from non-parametric model in the previous step, we sample recursively
    - Otherwise we simulate Q(y|x) = P(y|x, alpha(x)) <- since we can easily get Q(y|x), this becomes regular posterior estimate.
 