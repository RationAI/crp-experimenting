# Scrutinizing Concept Attribution in Neural Networks

In this presentation, we will explore the applications and limits of using Concept Relevance Propagation for explainablity in general, and with respect to medical data.

We will start with the well-known *ImageNet dataset* and a *VGG16*-based classification model.
ImageNet is a dataset consisting of several pictures of animals, fruits and other objects.
All we need to about the model is that it is large neural network with several convolutional (feature) layers. 

Suppose now that we would like to classify the following image.

![Lizard](lizard/lizard.png)

The model says that this image depicts an American chameleon with 86 % certainty.
The natural question is, **why** does the model say that.

## Layer-Wise Relevance Propagation (LRP)

Of course, answering the question *why* in general is a million dollar question.
A more narrow question is which **components of the model are most relevant** to the classification of a single image.
The method of Layer-wise Relevance Propagation (LRP) works in backpropagation-ish way.
LRP says that in the last layer, only the output neuron corresponding to the resulting class is relevant.
For this neuron, the relevant neurons in the layer below are those that contribute most significantly to its activation level.
Various methods exist to decide how exactly the contribution should be defined, but in general it depens on weights, activations and depend on the output function.

By backpropagating all the way to the input, we can visualize which parts of the original image are most relevant for the output.


