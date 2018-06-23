# Further exploring the classification of images of handwritten digits on the EMNIST dataset (Learning rules, BatchNorm, and ConvNets)
In this report, I investigate and further explore the performance the classification of images of handwritten digits using Deep Neural Network (DNN) and Convolutional Neural Network (CNN) trained on the EMNIST Balanced dataset. I firstly implement the experiments on the DNN. After many comparisons, I find that when the activation function is Relu and the learning rate is 0.001, the model performs best on the dataset. In addition, add the batch normalization layer and dropout layer can improve the performance of the model. At the second stage, I mainly focus on the investigation of the Convolutional Neural Network. I construct and train different networks containing one and two convolutional layers and max-pooling layers. After a very long time running, the experiment has the results that the two-layer CNN has better performances than one-layer CNN no matter the accuracy rate or the error rate. Finally, I use the test set to access the accuracy of the DNN and CNN and find that the accuracy rate of the CNN is higher than the DNN, especially the error rate of CNN is much lower than DNN.

## About the Dataset - EMNIST
In this project, I use the EMNIST (Extended MNIST) Balanced dataset, [https://www.nist.gov/itl/iad/image-group/emnist-dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset). EMNIST extends MNIST by including images of handwritten
letters (upper and lower case) as well as handwritten digits. Both EMNIST and MNIST are extracted from
the same underlying dataset, referred to as NIST Special Database 19. Both use the same conversion process
resulting in centred images of dimension 28*28.

Although there are 62 potential classes for EMNIST (10 digits, 26 lower case letters, and 26 upper case letters)
I will use a reduced label set of 47 different labels. This is because of confusions which arise when trying to
discriminate upper-case and lower-case versions of the same letter, following the data conversion process. In the
47 label set, upper- and lower-case labels are merged for the following letters: C, I, J, K, L, M, O, P, S, U, V, W,
X, Y and Z.

** The details of this project is shown in [the PDF file in the same repository](https://github.com/Mandy001/LR_BatchNorm_ConvNets/blob/master/Further%20exploring%20the%20classification%20of%20images%20of%20handwritten%20digits.pdf). **
