# AI and Deep Learning Fundamentals

Artificial Intelligence (AI) is the broad field of creating systems that perform tasks requiring human-like intelligence. Machine Learning (ML) is a subset of AI where systems learn patterns from data. Deep Learning (DL) is a subset of ML that uses multi-layer neural networks to learn hierarchical representations.

## Core Ideas

- **AI vs ML vs DL**: AI is the goal, ML is the approach, DL is a modern, neural-network-based approach.
- **Representation learning**: Deep models learn features automatically, reducing manual feature engineering.
- **Hierarchy of features**: Lower layers learn simple patterns; higher layers combine them into complex concepts.
- **Non-linearity**: Activation functions allow neural networks to model complex relationships.

## Feedforward Neural Networks (FNN)

A feedforward network sends information in one direction: input -> hidden layers -> output. The basic neuron computes a weighted sum and applies an activation:
$$y = f(Wx + b)$$
where $W$ are weights, $b$ is a bias, and $f$ is a non-linear activation function.

## Why Traditional ML Often Struggled

- **Manual feature engineering**: Classic ML relied on hand-crafted features, which is slow and brittle.
- **Complex, non-linear relationships**: Linear models and shallow methods often miss high-order interactions.
- **Unstructured data**: Images, audio, and text require spatial or sequential structure that classic models do not capture well.
- **Scalability**: Performance often saturates without large feature sets and careful tuning.

## Activation Functions (Purpose and Usage)

- **ReLU**: $\max(0, x)$; fast, avoids saturation for positive values, widely used in hidden layers.
- **Sigmoid**: Outputs in $(0,1)$; used for binary outputs or gates.
- **Tanh**: Outputs in $(-1,1)$; often used in recurrent networks.
- **Softmax**: Converts logits to class probabilities in multi-class classification.

## Fully Connected (Dense) Layers

A fully connected layer connects every input to every output. It learns global combinations of features and is commonly used near the end of a network for classification or regression.

## Common Network Families

- **CNNs**: Exploit local patterns and weight sharing, ideal for images and spatial data.
- **RNNs (LSTM/GRU)**: Capture sequential dependencies using hidden states.
- **Transformers**: Use attention to model long-range dependencies without recurrence.

## Viva Questions

1. **What is the difference between AI, ML, and DL?**
2. **What is representation learning?**
3. **What is a feedforward neural network (FNN)?**
4. **What does a neuron compute in a neural network?**
5. **Why are activation functions needed?**
6. **What is the ReLU activation, and why is it popular?**
7. **When would you use a sigmoid activation?**
8. **What is the role of the softmax function?**
9. **What is a fully connected (dense) layer?**
10. **How does depth help a neural network?**
11. **What is the difference between depth and width?**
12. **What is a hidden layer?**
13. **What is meant by non-linearity in neural networks?**
14. **Why did traditional ML often fail on unstructured data?**
15. **What does it mean to say deep models learn hierarchical features?**
16. **What is feature engineering, and how does DL reduce it?**
17. **What is an embedding in deep learning?**
18. **What is a decision boundary?**
19. **What is the universal approximation theorem (in simple terms)?**
20. **What is overfitting in a conceptual sense?**
21. **What is underfitting?**
22. **What is the bias-variance tradeoff?**
23. **What is a convolution, conceptually?**
24. **Why do CNNs use weight sharing?**
25. **What is a receptive field?**
26. **What is the key idea behind RNNs?**
27. **How do LSTM and GRU differ at a high level?**
28. **What is attention in deep learning?**
29. **Why do transformers not need recurrence?**
30. **What is a residual connection and why is it useful?**
31. **What is activation saturation and why is it a problem?**
32. **What is the difference between linear and non-linear models?**
33. **Why do deeper networks typically need non-linear activations between layers?**
34. **What is a parameter in a neural network?**
35. **What is the difference between parameters and hyperparameters?**
36. **What does it mean for a model to generalize?**
37. **Why are deep networks good at capturing high-order feature interactions?**
38. **What is dimensionality in the context of features?**
39. **How do embeddings help with sparse or categorical data?**
40. **What is a probability distribution in the context of softmax outputs?**
41. **What is a logits vector?**
42. **Why are hidden states used in sequence models?**
43. **What is the difference between sequence modeling and static modeling?**
44. **What is the idea of inductive bias?**
45. **Why do CNNs perform well on images?**
46. **Why do RNNs struggle with very long sequences without gating?**
47. **What is the intuition behind gating mechanisms?**
48. **What is the role of a final linear layer in a regression network?**
49. **What does it mean for a model to be end-to-end?**
50. **What is the main reason deep learning became practical in recent years?**

## Answers

**1. What is the difference between AI, ML, and DL?**
AI is the broad goal of intelligent behavior, ML is learning from data, and DL is ML using deep neural networks.

**2. What is representation learning?**
It is the ability of a model to automatically learn useful features from raw data instead of relying on manual feature engineering.

**3. What is a feedforward neural network (FNN)?**
An FNN passes information in one direction from input to output through hidden layers, without loops or memory.

**4. What does a neuron compute in a neural network?**
A weighted sum of inputs plus a bias, followed by a non-linear activation: $y = f(Wx + b)$.

**5. Why are activation functions needed?**
Without non-linear activations, multiple layers collapse into a single linear transformation and cannot model complex patterns.

**6. What is the ReLU activation, and why is it popular?**
ReLU is $\max(0, x)$; it is simple, fast, and avoids saturation for positive values, helping gradients flow.

**7. When would you use a sigmoid activation?**
For outputs that represent probabilities between 0 and 1, such as binary classification or gating.

**8. What is the role of the softmax function?**
It converts a vector of logits into a probability distribution over multiple classes.

**9. What is a fully connected (dense) layer?**
A layer where every input connects to every output, enabling global combinations of features.

**10. How does depth help a neural network?**
Deeper networks build hierarchical features, combining simple patterns into complex concepts.

**11. What is the difference between depth and width?**
Depth is the number of layers, width is the number of neurons per layer.

**12. What is a hidden layer?**
A layer between input and output that learns intermediate feature representations.

**13. What is meant by non-linearity in neural networks?**
It is the use of non-linear activations to model relationships that are not straight-line (linear) in the input space.

**14. Why did traditional ML often fail on unstructured data?**
It depends heavily on manual feature engineering and lacks built-in structure for images, text, or sequences.

**15. What does it mean to say deep models learn hierarchical features?**
Lower layers learn basic patterns; higher layers combine them into more abstract representations.

**16. What is feature engineering, and how does DL reduce it?**
Feature engineering is manual creation of input features; DL learns features automatically from raw data.

**17. What is an embedding in deep learning?**
A dense vector representation of categorical data or tokens that captures semantic similarity.

**18. What is a decision boundary?**
A surface that separates classes or output regions in the feature space.

**19. What is the universal approximation theorem (in simple terms)?**
A sufficiently large neural network can approximate any continuous function on a bounded domain.

**20. What is overfitting in a conceptual sense?**
The model learns noise or specific details of the data and fails to generalize to new inputs.

**21. What is underfitting?**
The model is too simple to capture the underlying patterns, leading to poor performance.

**22. What is the bias-variance tradeoff?**
High bias means oversimplification; high variance means sensitivity to noise. Good models balance both.

**23. What is a convolution, conceptually?**
A sliding filter that detects local patterns by combining nearby inputs with shared weights.

**24. Why do CNNs use weight sharing?**
It reduces parameters and captures the idea that the same pattern can appear at different positions.

**25. What is a receptive field?**
The region of the input that influences a particular output unit.

**26. What is the key idea behind RNNs?**
They maintain a hidden state that carries information from previous time steps.

**27. How do LSTM and GRU differ at a high level?**
LSTM uses separate cell and hidden states with multiple gates; GRU uses fewer gates and no separate cell state.

**28. What is attention in deep learning?**
A mechanism that lets the model focus on the most relevant parts of an input when producing an output.

**29. Why do transformers not need recurrence?**
Attention layers directly connect all positions, capturing long-range dependencies without sequential processing.

**30. What is a residual connection and why is it useful?**
It adds the input to the output of a layer, helping gradients flow and enabling deeper networks.

**31. What is activation saturation and why is it a problem?**
When activations are stuck near their limits (e.g., sigmoid near 0 or 1), gradients become small and learning slows.

**32. What is the difference between linear and non-linear models?**
Linear models produce outputs that are linear combinations of inputs; non-linear models can represent complex patterns.

**33. Why do deeper networks typically need non-linear activations between layers?**
Without non-linearity, stacking layers does not increase modeling power.

**34. What is a parameter in a neural network?**
A learnable value such as a weight or bias that defines the model's behavior.

**35. What is the difference between parameters and hyperparameters?**
Parameters are learned from data; hyperparameters are set by the designer (e.g., layer sizes).

**36. What does it mean for a model to generalize?**
It performs well on unseen data, not just the data it was exposed to.

**37. Why are deep networks good at capturing high-order feature interactions?**
Multiple layers successively combine features, creating complex interactions naturally.

**38. What is dimensionality in the context of features?**
The number of independent input variables or representation dimensions.

**39. How do embeddings help with sparse or categorical data?**
They map sparse symbols into dense vectors where similarity is meaningful and computation is efficient.

**40. What is a probability distribution in the context of softmax outputs?**
A set of non-negative values that sum to 1, representing class probabilities.

**41. What is a logits vector?**
The raw, unnormalized scores output by a model before softmax.

**42. Why are hidden states used in sequence models?**
They store context from previous time steps to inform current predictions.

**43. What is the difference between sequence modeling and static modeling?**
Sequence modeling uses order and temporal context, while static modeling treats inputs as independent.

**44. What is the idea of inductive bias?**
Assumptions built into a model that guide learning toward certain patterns (e.g., locality in CNNs).

**45. Why do CNNs perform well on images?**
Images have local spatial structure, and CNNs exploit locality and translation invariance.

**46. Why do RNNs struggle with very long sequences without gating?**
Gradients can vanish or explode, making it hard to preserve information over long time spans.

**47. What is the intuition behind gating mechanisms?**
Gates control the flow of information, deciding what to keep, update, or forget.

**48. What is the role of a final linear layer in a regression network?**
It maps the learned features to continuous output values.

**49. What does it mean for a model to be end-to-end?**
It learns directly from raw inputs to outputs without hand-designed feature pipelines.

**50. What is the main reason deep learning became practical in recent years?**
Large datasets, improved algorithms, and powerful hardware (GPUs) made training deep models feasible.
