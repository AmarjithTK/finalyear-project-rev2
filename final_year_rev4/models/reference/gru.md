# Theory and Inner Workings of the GRU Model

The Gated Recurrent Unit (GRU) is a streamlined variant of the LSTM network. It aims to solve the vanishing gradient problem using a gating mechanism but uses fewer parameters and lacks a separate cell state, making it computationally more efficient than LSTM while often achieving similar performance.

## Core Concepts
Instead of maintaining a separate memory cell ($c_t$) and a hidden state ($h_t$), the GRU uses only the hidden state ($h_t$) to transfer information. It combines the forget and input gates of an LSTM into a single "update gate."

## Mathematical Formulation

At each time step $t$, the GRU takes an input $x_t$ and the previous hidden state $h_{t-1}$.

### 1. Update Gate ($z_t$)
The update gate decides how much of the past information (from previous time steps) needs to be passed along to the future. It operates similarly to the combination of the forget and input gates in the LSTM.
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

### 2. Reset Gate ($r_t$)
The reset gate determines how much of the past information to forget. If $r_t$ is close to zero, the network effectively "forgets" the previous state and acts as if reading the first symbol of an input sequence.
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

### 3. Current Memory Content ($\tilde{h}_t$)
This calculates a candidate hidden state based on the current input and the previous hidden state (modulated by the reset gate).
$$\tilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1}, x_t] + b_h)$$

### 4. Final Hidden State ($h_t$)
The final hidden state represents a linear interpolation between the previous hidden state ($h_{t-1}$) and the current memory content ($\tilde{h}_t$), governed by the update gate ($z_t$).
$$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$

## How It Works in Our Context
1. **Processing Sequence**: For a 24-hour sequence, the GRU observes the multi-variate data step-by-step. At each hour, it decides how much of the historical load/weather context to keep via the update gate and how much to drop via the reset gate.
2. **Simplified Memory**: Since GRU doesn't have a separate cell state, memory modifications are applied directly to the hidden state, leading to faster training times per epoch compared to LSTM.
3. **Prediction**: The ultimate hidden state $h_{24}$ captures the essence of the input window. A dense linear layer projects this representation to predict the next hour's loads and generation.
