# CNN from Scratch with NumPy

This project demonstrates a complete Convolutional Neural Network (CNN) implementation using only NumPy, applied to the *MNIST* handwritten digit dataset. This CNN includes core layers—Convolution, Max Pooling, and Softmax—and is trained using backpropagation with cross-entropy loss, fully coded from scratch.

## Project Overview

The project aims to build a functional CNN model to classify handwritten digits from the *MNIST* dataset. Each layer, including forward and backward propagation (backpropagation), has been implemented with NumPy.

### Key Features
- **Custom CNN Implementation**: Convolutional, Max Pooling, and Softmax layers.
- **Handcrafted Backpropagation**: Implemented for each layer to optimize performance.
- **Training and Testing**: Model trains and tests on the *MNIST* dataset for digit recognition.
- **Result Visualization**: Graphs depicting accuracy and loss across different configurations.

## Project Structure
```plaintext
|-- main.py               # Main script for training and evaluating the model
|-- convolution.py        # Custom Convolution layer implementation
|-- max_pool.py           # Max Pooling layer implementation
|-- softmax.py            # Softmax layer with cross-entropy loss
|-- comparison.png        # plots comparing accuracy and loss across different configurations
```
## Model Details

### Layers Implemented
- **Convolutional Layer**: 3x3 filter, applied to 2D grayscale images.
- **Max Pooling Layer**: 2x2 pooling for down-sampling and dimensionality reduction.
- **Softmax Layer**: Maps output to a 10-class distribution using cross-entropy loss.

### Backpropagation
Each layer is equipped with backpropagation logic implemented from scratch to calculate gradients and update weights during training.

## Testing Configurations

Multiple CNN configurations were tested by varying the number and arrangement of convolutional and pooling layers. Each configuration’s impact on the training and testing performance was visualized in two graphs:

- **Accuracy vs. Configuration**
- **Loss vs. Configuration**

## Testing Results
![Accuracy vs. Configuration](comparison.png)

## Observations and Reasonings

- The configurations `c16pc16` and `c8pc16` exhibit better performance  
   - More filters in convolutional layers allow the model to capture a greater variety of features in the input data. This ability to learn complex patterns typically leads to improved classification performance, especially in datasets like *MNIST*, where digit shapes and styles can vary significantly
   - Deeper architectures tend to represent more abstract features of the input data. For example, the first layer might learn simple edges, while deeper layers can learn more complex shapes and patterns

- Configurations like `c8pc8p` and others that include multiple pooling layers appear to hinder performance  
  -  Pooling layers are meant to reduce dimensionality and retain the most important information. However, excessive pooling can lead to loss of critical spatial information, which is vital for tasks such as image classification. In the case of *MNIST*, the digits are not overly complex, so too much pooling can discard important features needed for accurate classification  

## Conclusion

- The observation that `c16pc16` performed the best reinforces the principle of using deeper architectures with more filters in tasks involving complex data. This model likely achieved better performance because it could learn and generalize well from the training data, resulting in improved performance on unseen test data
- The analysis suggests that finding an optimal balance between convolutional and pooling layers is crucial. While pooling layers are necessary for down-sampling and controlling overfitting, too many can be counterproductive.

