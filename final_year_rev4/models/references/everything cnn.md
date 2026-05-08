# CNN Reference (Everything You Might Be Asked)

## 1) What is a CNN?
A convolutional neural network (CNN) is a neural network that uses convolution layers to learn spatial or temporal patterns with shared weights and local connectivity. It is widely used for images, time series, audio, and text.

## 2) Core CNN building blocks
- Convolution layer
- Nonlinearity (e.g., ReLU)
- Pooling (optional)
- Normalization (optional: BatchNorm/LayerNorm)
- Dropout (optional)
- Fully connected (dense) layers

## 3) Most important formulas

### 3.1 Convolution (2D)
For input $x$ and kernel $w$:

$$
(y_{k})(i, j) = \sum_{c=1}^{C_{in}} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} w_{k,c}(m,n)\, x_c(i+m, j+n) + b_k
$$

Where $k$ is the output channel index, $C_{in}$ is number of input channels, and $b_k$ is bias.

### 3.2 Output size (2D)
For input height/width $H \times W$:

$$
H_{out} = \left\lfloor \frac{H + 2P_h - D_h\,(K_h-1) - 1}{S_h} + 1 \right\rfloor,
\quad
W_{out} = \left\lfloor \frac{W + 2P_w - D_w\,(K_w-1) - 1}{S_w} + 1 \right\rfloor
$$

Where $K$ is kernel size, $S$ is stride, $P$ is padding, $D$ is dilation.

### 3.3 Receptive field (approx)
For stacked convs with stride $S_l$ and kernel $K_l$ at layer $l$:

$$
R_l = R_{l-1} + (K_l - 1) \prod_{i=1}^{l-1} S_i
$$

### 3.4 Pooling (max/avg)

$$
\text{MaxPool}(x) = \max_{(i,j) \in \Omega} x(i,j),
\quad
\text{AvgPool}(x) = \frac{1}{|\Omega|}\sum_{(i,j) \in \Omega} x(i,j)
$$

### 3.5 Common loss functions
- MSE: $\text{MSE} = \frac{1}{N}\sum (y - \hat{y})^2$
- Cross-entropy (multi-class): $\mathcal{L} = -\sum y\log(\hat{y})$

## 4) Key concepts to explain
- **Local connectivity:** filters look at small regions.
- **Parameter sharing:** same filter used across the input.
- **Translation equivariance:** shifting input shifts feature maps.
- **Hierarchical features:** early layers detect edges; deeper layers detect patterns.

## 5) Stride, padding, and dilation
- **Stride:** step size of the filter; increases downsampling.
- **Padding:** zeros around borders; keeps spatial size or controls it.
- **Dilation:** inserts gaps in filter; increases receptive field.

## 6) 1D vs 2D vs 3D CNN
- **1D:** time series or signals (shape: $L \times C$).
- **2D:** images (shape: $H \times W \times C$).
- **3D:** video/volumetric data (shape: $D \times H \times W \times C$).

## 7) Common architectures
- LeNet, AlexNet, VGG, GoogLeNet/Inception, ResNet, DenseNet, MobileNet, EfficientNet.
- **ResNet idea:** skip connections, $y = F(x) + x$.

## 7.1) CNN in our project (microgrid time-series forecasting)
We use a 1D CNN as a temporal feature extractor on multivariate sequences, then feed the learned features to an LSTM. This is a CNN-LSTM hybrid for short-term forecasting of solar, wind, and load values.

### 7.1.1 Input and shape in our pipeline
- Each sample is a 24-hour window of multivariate inputs.
- Input shape to the CNN: $[\text{batch}, \text{channels}, \text{time}]$ after permutation.
- Channels correspond to features and past targets; time length is 24.

### 7.1.2 Why CNN before LSTM
- CNN learns local temporal motifs (e.g., short-term ramps, spikes).
- It reduces noise and builds higher-level features.
- LSTM then models longer temporal dependencies using these features.

### 7.1.3 1D convolution formula
For 1D input $x$ with kernel $w$:

$$
y_k(t) = \sum_{c=1}^{C_{in}} \sum_{m=0}^{K-1} w_{k,c}(m)\, x_c(t+m) + b_k
$$

### 7.1.4 1D output length
For input length $L$:

$$
L_{out} = \left\lfloor \frac{L + 2P - D\,(K-1) - 1}{S} + 1 \right\rfloor
$$

### 7.1.5 Key hyperparameters (project context)
- Kernel size: small (e.g., 3) to capture local trends.
- Padding: set to keep the time length stable.
- Stride: 1 to preserve resolution.
- Filters: increase channels to enrich learned features.

### 7.1.6 How to explain the CNN block in our report
"We apply a 1D convolution over 24-hour input windows to capture local temporal patterns across all variables. The CNN acts as a feature extractor, producing a richer sequence representation that the LSTM uses to model longer-range dependencies and generate multi-target forecasts."

### 7.1.7 Common questions specific to our CNN usage
- Why not pure LSTM? CNN helps denoise and detect local patterns before sequence modeling.
- Why 1D instead of 2D? The data is temporal; channels represent variables, not spatial axes.
- What does padding do here? Keeps output length consistent for the LSTM.

## 8) Normalization and regularization
- **BatchNorm:** stabilizes training by normalizing activations.
- **Dropout:** randomly drops units to reduce overfitting.
- **Weight decay (L2):** penalizes large weights.
- **Data augmentation:** flips, crops, jitter, noise.

## 9) Training essentials
- Initialize weights (He/Kaiming for ReLU).
- Use learning rate schedules.
- Monitor train/val loss for overfitting.
- Use early stopping if needed.

## 10) CNN pros/cons
**Pros**
- Efficient parameter usage.
- Strong inductive bias for spatial data.
- High accuracy in vision tasks.

**Cons**
- Less effective for very long-range dependencies.
- Can be data-hungry without augmentation.

## 11) Interview-style Q&A cues
- Why CNNs over MLPs for images? Fewer parameters, locality, translation equivariance.
- What does padding do? Controls output size, keeps border info.
- What is stride? Downsampling by skipping positions.
- What is receptive field? Input region affecting an output unit.
- How do skip connections help? Reduce vanishing gradients, enable deeper nets.
- How to prevent overfitting? Dropout, augmentation, weight decay, early stopping.

## 12) Quick reference cheat sheet
- Conv output size: $\left\lfloor \frac{H + 2P - D(K-1) - 1}{S} + 1 \right\rfloor$
- 95% confidence band (if asked): $\hat{y} \pm 1.96\sigma$
- ResNet block: $y = F(x) + x$

## 13) Common pitfalls
- Mismatched tensor shapes.
- Incorrect padding/stride causing size errors.
- Overfitting due to limited data.
- Too large learning rate causing divergence.

## 14) Practical tips
- Start simple (few layers), then scale.
- Use BatchNorm for stability.
- Visualize feature maps when debugging.
- Track GPU memory (batch size vs model size).
