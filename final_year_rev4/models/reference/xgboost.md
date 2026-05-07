# Theory and Inner Workings of the XGBoost Model

Extreme Gradient Boosting (XGBoost) is a highly efficient and scalable implementation of the gradient boosted decision trees algorithm. Unlike LSTMs or GRUs, which are neural networks designed for sequential data, XGBoost is an ensemble learning method based on tree architecture.

## Core Concepts
XGBoost builds an ensemble of "weak learners"—typically decision trees—sequentially. Each new tree attempts to correct the residual errors made by the combination of all previous trees.

1. **Decision Trees**: A flowchart-like structure where internal nodes test a feature (e.g., Temperature > 25°C), branches represent the outcome, and leaves represent the predicted value.
2. **Gradient Boosting**: An optimization algorithm that minimizes a loss function by adding weak learners using a gradient descent-like procedure. "Boosting" means that errors of previous models are given more weight by the next models.

## Mathematical Formulation

Let the dataset have $n$ examples and $m$ features. A tree ensemble model uses $K$ additive functions (trees) to predict the output:
$$\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)$$
Where $f_k$ is an independent regression tree.

### 1. Objective Function
To learn the set of functions, XGBoost minimizes the following regularized objective function:
$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
- $l$: Differentiable convex loss function that measures the difference between prediction $\hat{y}_i$ and target $y_i$ (e.g., Mean Squared Error).
- $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \|w\|^2$: The regularization term that penalizes the complexity of the model to prevent overfitting. $T$ is the number of leaves in the tree, and $w$ represents the leaf weights.

### 2. Taylor Approximation
XGBoost uses a second-order Taylor approximation to quickly optimize the objective function:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
Where:
- $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ (First derivative / gradient)
- $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$ (Second derivative / hessian)

### 3. Tree Building (Splitting)
To evaluate the quality of a split in a tree node, XGBoost calculates the gain based on the gradients and hessians of the instances in the left ($L$) and right ($R$) child nodes:
$$Gain = \frac{1}{2} \left[ \frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda} + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$
The algorithm scans for splits that maximize this Gain. If Gain < 0, the node is not split (pruning).

## How It Works in Our Context
1. **Flattening Sequences**: Since XGBoost does not natively handle 3D sequence tensors `(samples, time_steps, features)` like RNNs do, the dataset sequences are flattened into 2D arrays `(samples, time_steps * features)`. A 24-hour sequence with 6 features becomes a single row with 144 independent features.
2. **Multi-Output Strategy**: Our implementation uses `MultiOutputRegressor(XGBRegressor)`. Since standard XGBoost predicts one target variable, this wrapper trains $N$ separate XGBoost models (one for each of our 6 targets: Solar MW, Wind MW, etc.).
3. **Training**: Each of the 6 models builds sequential trees to minimize predictions error for its specific load or generation target.