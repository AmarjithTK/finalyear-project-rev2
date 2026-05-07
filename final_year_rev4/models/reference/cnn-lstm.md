# Theory and Inner Workings of the CNN-LSTM Hybrid Model

A CNN-LSTM is a hybrid neural network architecture that combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. This architecture is designed to capture both the spatial (or cross-feature local) structures and the temporal (sequential) dependencies within time-series data.

## Core Concepts

1. **CNN (Feature Extractor)**: A 1-dimensional convolutional layer moves across the temporal dimension. Instead of viewing raw data sequentially, the CNN correlates nearby timestamps and identifies local patterns across different weather and load features simultaneously (e.g., recognizing that high solar irradiance and high temperature happen together in a block of a few hours). 
2. **LSTM (Sequence Modeler)**: The filtered and transformed features from the CNN are fed into the LSTM layer. The LSTM's job is to read these "high-level" patterns over time and learn the long-term temporal dependencies (e.g., daily seasonality or multi-day trends).

## Mathematical Formulation

Let the input sequence be $X = [x_1, x_2, \dots, x_T]$, where each $x_t \in \mathbb{R}^d$ and $d$ is the number of features.

### 1. 1D Convolutional Layer
A 1D convolutional filter $W_c$ of size $k$ slides over the input sequence. For a given time window from $t$ to $t+k-1$, the convolution operation produces a feature map element $c_t$:
$$c_t = \text{ReLU}\left( \sum_{i=0}^{k-1} W_c \cdot x_{t+i} + b_c \right)$$
- The ReLU activation function introduces non-linearity: $\text{ReLU}(z) = \max(0, z)$.
- The output of the CNN is a new sequence of feature maps $C = [c_1, c_2, \dots, c_{T'}]$ which acts as an abstract representation of local events.

### 2. LSTM Layer
The sequence $C$ is passed step-by-step into the LSTM. At each sequence step $t$, the LSTM updates its hidden state $h_t$ and cell state $c^{lstm}_t$ based on the new feature map $c_t$ and the previous hidden state $h_{t-1}$.

The LSTM operates using gates:
- **Forget Gate ($f_t$)**: $f_t = \sigma(W_f \cdot [h_{t-1}, c_t] + b_f)$
- **Input Gate ($i_t$)**: $i_t = \sigma(W_i \cdot [h_{t-1}, c_t] + b_i)$
- **Candidate Cell State ($\tilde{c}^{lstm}_t$)**: $\tilde{c}^{lstm}_t = \tanh(W_c \cdot [h_{t-1}, c_t] + b_c)$
- **Update Cell State**: $c^{lstm}_t = f_t * c^{lstm}_{t-1} + i_t * \tilde{c}^{lstm}_t$
- **Output Gate ($o_t$)**: $o_t = \sigma(W_o \cdot [h_{t-1}, c_t] + b_o)$
- **Hidden State**: $h_t = o_t * \tanh(c^{lstm}_t)$

### 3. Fully Connected Layer
After processing the entire sequence, we take the LSTM's final hidden state $h_{T'}$ representing the summary of the 24-hour sequence.
The output values (predictions $\hat{y}$) are calculated using a linear transformation:
$$\hat{y} = W_{fc} \cdot h_{T'} + b_{fc}$$

## How It Works in Our Context
1. **Conv1D Filter**: A sequence of 24 hours of multi-variate data is passed. The `nn.Conv1d` layer acts as a local feature scanner. It smooths out noise and emphasizes sharp changes across multiple factors like temperature and load simultaneously.
2. **Sequence Processing**: The LSTM processes the CNN's feature map. It remembers how yesterday morning's load peak connects to today's expected behavior.
3. **Final Linear Layer**: The network takes the final summarized "memory state" and regresses it directly into predicting the target MW values for our Loads and Generation.
