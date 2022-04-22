# skorchNAM
Neural Additive Model implemented in pytorch wrapped by skorch as a sklearn estimator (classifier)


* Generalized Additive Models (GAM) are a class of models where the contribution of individual features are additive. Logistic regression, for example, is a GAM. GAMs are relatively interpretable as the contribution of individual features to the output can be easily quantified and compared.
* [Neural Additive Models (NAM)](https://neural-additive-models.github.io/) are a type of GAM where the contribution of individual features on the outcome is each modeled using a deep neural network.
* [sklearn](https://scikit-learn.org/stable/index.html) is the premier, batteries-included python machine learning package with useful features such as those for hyperparameter optimization or pipeline construction.
* [skorch](https://github.com/skorch-dev/skorch) is a package for wrapping pytorch neural network models as sklearn estimators

See: [Jupyter Notebook](https://github.com/abraxasyu/skorchNAM/blob/main/NAM_testing.ipynb)

TODO:
* I excluded ExU from the NAM paper for simplicity, but it may be worth adding and experimenting
* Make the NAMRegressor version of NAMClassifier
* See if there's a way to simplify the loss function - triple wrapping caused by the requirement for the lossfunc to be a callable, and for binary classification to require 2 columns rather than just 1
