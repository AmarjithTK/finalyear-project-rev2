# Viva Questions: CNN-LSTM Hybrid Model

## Model Summary (Project Context)

- **Goal**: Forecast future microgrid load/generation from multivariate time-series data.
- **Input shape**: `[batch, sequence_length, features]` is permuted for `Conv1d` and then permuted back for `LSTM`.
- **Data flow**: `Conv1d` -> `ReLU` -> `LSTM` -> last hidden state -> `Linear` regression head.
- **Training**: Sliding-window sequences, scaled inputs, chronological train/test split, regression loss (e.g., MSE/MAE).
- **Why hybrid**: CNN captures local cross-feature patterns; LSTM captures longer temporal dependencies.

## Viva Questions

1. **What is a CNN-LSTM hybrid model?**
2. **Why use a CNN-LSTM for time-series forecasting instead of a simple LSTM?**
3. **What is the purpose of the 1D Convolutional layer in our model?**
4. **How does a `Conv1d` layer process multivariate time-series data?**
5. **What is the role of the LSTM layer following the CNN?**
6. **How do you handle the tensor shape mismatch between PyTorch's `Conv1d` and `LSTM` layers?**
7. **What is the purpose of the `permute` method when transitioning from PyTorch's Conv1d to LSTM?**
8. **Why do we use the ReLU activation function after the convolutional layer?**
9. **In the LSTM layer, how does the model "remember" past inputs?**
10. **Explain the purpose of the Forget Gate in an LSTM.**
11. **What does the Input Gate do in an LSTM node?**
12. **What does the Output Gate control in the LSTM?**
13. **Why do we only take the last time step output `lstm_out[:, -1, :]` from the LSTM before passing it to the fully connected layer?**
14. **What is the purpose of the Fully Connected (`nn.Linear`) layer at the end of the network?**
15. **How does the CNN-LSTM handle multiple input features (like temperature, solar irradiance, and historical load) simultaneously?**
16. **Why do we apply `MinMaxScaler` or standard scaling to the data before feeding it to the CNN-LSTM?**
17. **What does the `hidden_size` hyperparameter in the PyTorch LSTM signify?**
18. **How does the kernel size in `Conv1d` affect the model's feature extraction?**
19. **What is the effect of the sequence length (lookback window) on model performance?**
20. **Why use `batch_first=True` when defining the PyTorch LSTM?**
21. **What loss function is commonly used for regression tasks like load forecasting, and why?**
22. **Explain how backpropagation through time (BPTT) works in this hybrid model.**
23. **What is the exploding/vanishing gradient problem, and how does the LSTM architecture help mitigate it?**
24. **How would you interpret the `out_channels` parameter in our `Conv1d` layer?**
25. **If we wanted to predict the next 24 hours instead of just the next hour, how would the architecture change?**
26. **What is the purpose of adding an Attention Mechanism to the CNN-LSTM model?**
27. **How does the Attention mechanism calculate the context vector?**
28. **In testing, we use `DataLoader` with `shuffle=False`. Why is shuffling disabled for the test set?**
29. **What metrics (MAE, RMSE, R2) are most informative for evaluating our microgrid CNN-LSTM model and why?**
30. **How can you prevent the CNN-LSTM model from overfitting on the training data?**
31. **What is the difference between univariate and multivariate time-series forecasting in this model?**
32. **What is data leakage, and how do we avoid it when scaling time-series data?**
33. **Why is a chronological train/test split necessary for time-series models?**
34. **Why is `shuffle=True` typically used in the training `DataLoader` for windowed sequences?**
35. **What does the `num_layers` parameter in the LSTM control?**
36. **What is gradient clipping, and why might it be helpful for LSTM training?**
37. **What is the receptive field in a 1D CNN, and why does it matter for time-series?**
38. **How do you decide on a suitable learning rate for this model?**
39. **How do you convert predictions back to original units after scaling?**
40. **What is the difference between one-step and multi-step forecasting?**
41. **What does `output_size` represent in the final linear layer?**
42. **How does dropout help in a CNN-LSTM model?**
43. **Why use early stopping during training?**
44. **How do you handle missing timestamps or NaN values in the input series?**
45. **What is concept drift in energy time-series data, and how can we respond to it?**
46. **Why is `Conv1d` appropriate here instead of `Conv2d`?**
47. **How does batch size affect convergence and generalization?**
48. **What is the role of a validation set in training this model?**
49. **How do MAE and RMSE differ in their sensitivity to outliers?**
50. **How can learning curves indicate overfitting?**

---

## Answers

**1. What is a CNN-LSTM hybrid model?**
It is a neural network architecture that combines Convolutional Neural Networks (CNNs) to extract local spatial or cross-feature patterns, and Long Short-Term Memory (LSTM) layers to model sequential/temporal dependencies over time.

**2. Why use a CNN-LSTM for time-series forecasting instead of a simple LSTM?**
While LSTMs are great for long-term temporal dependencies, they can struggle to filter out noise or identify local, short-term correlations across multiple input features. The CNN acts as a feature extractor/filter, preprocessing the raw data so the LSTM receives cleaner, higher-level representations.

**3. What is the purpose of the 1D Convolutional layer in our model?**
The 1D CNN moves across the time axis to capture local, short-term patterns (e.g., sudden spikes or correlations between temperature and load over a few hours) across all input variables simultaneously.

**4. How does a `Conv1d` layer process multivariate time-series data?**
It applies learnable filters (kernels) across the temporal dimension. The `in_channels` correspond to the different features (weather, load), and the kernel slides over the time steps to produce `out_channels` (feature maps) that represent complex local correlations.

**5. What is the role of the LSTM layer following the CNN?**
The LSTM takes the feature maps generated by the CNN at each time step and models the long-term temporal dependencies, learning how the extracted features evolve over the entire input sequence.

**6. How do you handle the tensor shape mismatch between PyTorch's `Conv1d` and `LSTM` layers?**
PyTorch's `Conv1d` expects the shape `[batch_size, features, sequence_length]`, while `LSTM` (with `batch_first=True`) expects `[batch_size, sequence_length, features]`. We resolve this by permuting the tensor dimensions before and after the CNN layer.

**7. What is the purpose of the `permute` method when transitioning from PyTorch's Conv1d to LSTM?**
`permute(0, 2, 1)` swaps the feature/channel and sequence length dimensions. We do this to convert the CNN output from `[batch, filters, sequence]` to `[batch, sequence, filters]` so the LSTM receives data sequentially.

**8. Why do we use the ReLU activation function after the convolutional layer?**
ReLU (Rectified Linear Unit) introduces non-linearity, allowing the network to learn complex patterns and preventing the vanishing gradient problem during backpropagation.

**9. In the LSTM layer, how does the model "remember" past inputs?**
Through its inner "cell state," which runs straight down the entire chain with minimal linear interactions, governed by specialized gates that add or remove information passing through.

**10. Explain the purpose of the Forget Gate in an LSTM.**
It decides what information from the previous cell state should be discarded or kept, based on the current input and the previous hidden state. It outputs a value between 0 and 1.

**11. What does the Input Gate do in an LSTM node?**
It identifies what new information from the current time step's input should be added to the cell state.

**12. What does the Output Gate control in the LSTM?**
It determines what the next hidden state should be by filtering the newly updated cell state. This hidden state contains information on previous inputs and is used for predictions.

**13. Why do we only take the last time step output `lstm_out[:, -1, :]` from the LSTM before passing it to the fully connected layer?**
Because the last time step's hidden state contains the summarized context and accumulated memory of the entire input sequence (e.g., the past 24 hours), which is what we need to predict the next future value.

**14. What is the purpose of the Fully Connected (`nn.Linear`) layer at the end of the network?**
It maps the high-dimensional hidden state vector generated by the LSTM into the actual target values (e.g., the forecasted MW load or generation) we want to predict.

**15. How does the CNN-LSTM handle multiple input features simultaneously?**
The `in_channels` of the Conv1D layer accepts all features stacked together at each time step. The convolution filters correlate these variables locally before passing the multivariate feature maps to the LSTM.

**16. Why do we apply `MinMaxScaler` or standard scaling to the data before feeding it to the CNN-LSTM?**
Neural networks train better and converge faster when inputs are on a similar scale (e.g., 0 to 1). If variables (like irradiance in thousands vs. temperature in tens) have drastically different scales, the network will disproportionately bias towards features with larger magnitudes and gradients may explode.

**17. What does the `hidden_size` hyperparameter in the PyTorch LSTM signify?**
It defines the number of features in the hidden state and cell state for each LSTM block. A larger hidden size allows the model to memorize more complex long-term dependencies, but increases the risk of overfitting and computational cost.

**18. How does the kernel size in `Conv1d` affect the model's feature extraction?**
The kernel size determines the local receptive field or "window" of time steps the filter looks at simultaneously. A kernel size of 3 means the CNN evaluates 3 consecutive hours at a time to find local patterns.

**19. What is the effect of the sequence length (lookback window) on model performance?**
A longer sequence provides more historical context for the model to understand trends and seasonality (e.g., 24 hours captures daily cycles). Too long, and the model may struggle with vanishing gradients or irrelevant distant data. Too short, and the model loses crucial temporal context.

**20. Why use `batch_first=True` when defining the PyTorch LSTM?**
By default, PyTorch LSTMs expect inputs of shape `[sequence_length, batch_size, features]`. Setting `batch_first=True` aligns it with the more intuitive and common shape of `[batch_size, sequence_length, features]`.

**21. What loss function is commonly used for regression tasks like load forecasting, and why?**
Mean Squared Error (MSE) or Mean Absolute Error (MAE). MSE penalizes large errors heavier (useful if large prediction misses are costly), while MAE provides a linear penalty and is more robust to outliers.

**22. Explain how backpropagation through time (BPTT) works in this hybrid model.**
The error (loss) is calculated at the final output. The gradients are propagated backward through the linear layer, unrolled backward through each time step of the LSTM, and finally backward through the Convolutional layer, updating the weights in all layers.

**23. What is the exploding/vanishing gradient problem, and how does the LSTM architecture help mitigate it?**
When training deep networks on long sequences, gradients can become infinitesimally small or massively large. LSTMs mitigate this through their additive cell state pathway and gating mechanisms (especially the forget gate), which allow error signals to flow backward smoothly.

**24. How would you interpret the `out_channels` parameter in our `Conv1d` layer?**
It represents the number of different filters applied to the input. Each filter learns a distinct feature map or specific local pattern from the input data. It dictates the dimension of the feature vector passed to the LSTM at each time step.

**25. If we wanted to predict the next 24 hours instead of just the next hour, how would the architecture change?**
The output dimension of the final fully connected layer (`nn.Linear`) would be changed from 1 to 24 (or the number of target variables times 24). Alternatively, an encoder-decoder (seq2seq) architecture could be used.

**26. What is the purpose of adding an Attention Mechanism to the CNN-LSTM model?**
Instead of relying solely on the final LSTM time step, Attention allows the model to look at the hidden states of all past time steps and dynamically weight them based on their importance for the target prediction.

**27. How does the Attention mechanism calculate the context vector?**
It calculates an alignment/attention score for each time-step's hidden state, applies a softmax function to get a probability distribution (weights), and then computes a time-step-weighted sum of the LSTM hidden states.

**28. In testing, we use `DataLoader` with `shuffle=False`. Why is shuffling disabled for the test set?**
To preserve chronological order. In time-series forecasting, we need to evaluate the model sequentially to accurately plot and compare the continuous forecasted line against the true chronological timeline.

**29. What metrics (MAE, RMSE, R2) are most informative for evaluating our microgrid CNN-LSTM model and why?**
RMSE penalizes large deviations, which is bad for power grids. MAE gives average expected error in actual units (MW). R2 indicates how much of the variance in the generated/consumed power is explained by the model.

**30. How can you prevent the CNN-LSTM model from overfitting on the training data?**
By using dropout layers (e.g., `nn.Dropout`), introducing L2 regularization (weight decay) in the optimizer, applying early stopping based on a validation set, or reducing model complexity (fewer filters, smaller hidden state).

**31. What is the difference between univariate and multivariate time-series forecasting in this model?**
Univariate forecasting uses a single feature as input, while multivariate forecasting uses multiple features (e.g., temperature, irradiance, past load). In our model, multivariate inputs are handled as channels for `Conv1d`.

**32. What is data leakage, and how do we avoid it when scaling time-series data?**
Data leakage occurs when information from the test set influences training. Avoid it by fitting the scaler on training data only, then applying the same transform to validation/test data.

**33. Why is a chronological train/test split necessary for time-series models?**
Random splits can leak future information into training. A chronological split respects time order and simulates real-world forecasting conditions.

**34. Why is `shuffle=True` typically used in the training `DataLoader` for windowed sequences?**
Shuffling reduces correlation between adjacent windows in a batch, stabilizes gradient updates, and improves convergence. For evaluation, we keep ordering intact.

**35. What does the `num_layers` parameter in the LSTM control?**
It controls the number of stacked LSTM layers. More layers can model richer temporal patterns but increase training cost and overfitting risk.

**36. What is gradient clipping, and why might it be helpful for LSTM training?**
Gradient clipping limits the gradient magnitude to prevent exploding gradients, improving training stability on long sequences.

**37. What is the receptive field in a 1D CNN, and why does it matter for time-series?**
It is the number of time steps that influence a CNN output at a given position. A larger receptive field captures broader local context; kernel size and depth determine it.

**38. How do you decide on a suitable learning rate for this model?**
Start with a common value (e.g., 1e-3 for Adam), monitor loss curves, and tune using validation performance or a learning-rate finder.

**39. How do you convert predictions back to original units after scaling?**
Apply the inverse transform of the scaler fitted on the training data to map normalized predictions back to MW or original units.

**40. What is the difference between one-step and multi-step forecasting?**
One-step predicts the next time point only. Multi-step predicts a horizon (e.g., next 24 hours) using a vector output or recursive predictions.

**41. What does `output_size` represent in the final linear layer?**
It represents the number of predicted targets per time step (and optionally the forecast horizon if using a vector output).

**42. How does dropout help in a CNN-LSTM model?**
Dropout randomly deactivates neurons during training, reducing co-adaptation and improving generalization.

**43. Why use early stopping during training?**
It halts training when validation loss stops improving, preventing overfitting and saving compute.

**44. How do you handle missing timestamps or NaN values in the input series?**
Use interpolation, forward/backward fill, or model-based imputation; ensure consistent time intervals before sequence creation.

**45. What is concept drift in energy time-series data, and how can we respond to it?**
Concept drift is a change in underlying patterns (e.g., seasonal shifts or demand changes). Respond by retraining periodically or using sliding-window updates.

**46. Why is `Conv1d` appropriate here instead of `Conv2d`?**
The data is a 1D temporal sequence with multiple channels. `Conv1d` slides along time, while `Conv2d` is suited to spatial grids or images.

**47. How does batch size affect convergence and generalization?**
Larger batches yield smoother gradients and faster throughput but can generalize worse; smaller batches add noise that can improve generalization but train slower.

**48. What is the role of a validation set in training this model?**
It provides unbiased feedback for hyperparameter tuning and early stopping without touching the test set.

**49. How do MAE and RMSE differ in their sensitivity to outliers?**
RMSE penalizes larger errors more heavily due to squaring, while MAE treats all errors linearly and is more robust to outliers.

**50. How can learning curves indicate overfitting?**
If training loss keeps decreasing but validation loss starts increasing or plateaus, the model is overfitting.