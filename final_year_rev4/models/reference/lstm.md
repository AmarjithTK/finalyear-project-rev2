# Theory and Inner Workings of the LSTM Model

The Long Short-Term Memory (LSTM) network is a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies. It mitigates the vanishing gradient problem common in vanilla RNNs through a system of gating mechanisms.

## Core Concepts
Unlike a standard feedforward network, an LSTM maintains a hidden state ($h_t$) and a cell state ($c_t$) over time.

- **Cell State ($c_t$)**: The "long-term memory" of the network. It runs horizontally down the entire chain, with only minor linear interactions, allowing information to flow unchanged.
- **Gates**: Neural network layers with a sigmoid activation function (outputting between 0 and 1) that optionally let information through. 0 means "let nothing through," and 1 means "let everything through."

## Mathematical Formulation

At each time step $t$, the LSTM takes an input $x_t$ and the previous hidden state $h_{t-1}$.

### 1. Forget Gate ($f_t$)
Decides what information to throw away from the cell state.
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

### 2. Input Gate ($i_t$) and Candidate Cell State ($\tilde{c}_t$)
Decides what new information to store in the cell state.
The input gate layer decides which values to update:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
A tanh layer creates a vector of new candidate values:
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

### 3. Update Cell State ($c_t$)
Updates the old cell state, $c_{t-1}$, into the new cell state $c_t$.
$$c_t = f_t * c_{t-1} + i_t * \tilde{c}_t$$

### 4. Output Gate ($o_t$) and Hidden State ($h_t$)
Decides what to output based on the filtered cell state.
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * \tanh(c_t)$$

## How It Works in Our Context
1. **Input Sequence**: For a given sequence length (e.g., 24 hours), the LSTM processes the 12 features at time step 1, updating its cell and hidden states.
2. **Temporal Context**: It moves to step 2, combining the new input with the context (hidden state) from step 1.
3. **Prediction**: After observing all 24 hours, the final hidden state $h_{24}$ contains a summary of the temporal dynamics. A final fully connected (Linear) layer maps this $h_{24}$ representation to the final predictions for the 6 target variables.
