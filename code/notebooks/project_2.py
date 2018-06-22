
# coding: utf-8

# # Coursework 2
# 
# This notebook is intended to be used as a starting point for your experiments. The instructions can be found in the instructions file located under spec/coursework2.pdf. The methods provided here are just helper functions. If you want more complex graphs such as side by side comparisons of different experiments you should learn more about matplotlib and implement them. Before each experiment remember to re-initialize neural network weights and reset the data providers so you get a properly initialized experiment. For each experiment try to keep most hyperparameters the same except the one under investigation so you can understand what the effects of each are.

# In[1]:


import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')


def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    
    return stats, keys


# In[10]:


# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
import os
os.environ["MLP_DATA_DIR"] = "../data"

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
#create the test date
test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)


# ## PartA 1.   Perform baseline experiments using DNNs trained on EMNIST Balanced.
# 
# ### different activation layers such as  SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer....

# In[3]:

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)


model_SigmoidLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_ReluLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_LeakyReluLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_ELULayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ELULayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ELULayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_SELULayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    SELULayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    SELULayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])


# model = MultipleLayerModel([
#     AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
#     ReluLayer(),
#     AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
#     ReluLayer(),
#     AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
# ])

error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('SigmoidLayer\n\n')
[stats_SigmoidLayer, keys_SigmoidLayer] = train_model_and_plot_stats(
    model_SigmoidLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('ReluLayer\n\n')
[stats_ReluLayer, keys_ReluLayer] = train_model_and_plot_stats(
    model_ReluLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('LeakyReluLayer\n\n')
[stats_LeakyReluLayer, keys_LeakyReluLayer] = train_model_and_plot_stats(
    model_LeakyReluLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('ELULayer\n\n')
[stats_ELULayer, keys_ELULayer] = train_model_and_plot_stats(
    model_ELULayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('SELULayer\n\n')
[stats_SELULayer, keys_SELULayer] = train_model_and_plot_stats(
    model_SELULayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_1.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_1.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_1.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different activation layers  in the training dataset')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('PartA1_activation_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_2.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_2.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_2.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different activation layers  in the valid dataset')
fig_2.tight_layout() # This minimises whitespace around the axes.
fig_2.savefig('PartA1_activation_valid_error.pdf') # Save figure to current directory in PDF format
#


fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_3.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_3.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_3.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different activation layers  in the training dataset')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('PartA1_activation_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_4.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_4.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_4.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different activation layers  in the valid dataset')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('PartA1_activation_valid_acc.pdf') # Save figure to current directory in PDF format


# # PartA 1. Perform baseline experiments using DNNs trained on EMNIST Balanced.
# ## explore the performance of different learning rate on the selected activatoion function Relu.
# I change the learning rate to 0.01 and 0.001

# In[8]:


from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

#learning_rate = 0.1
model_ReluLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

#learning_rate = 0.01
model_ReluLayer_changeLearnRate = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_ReluLayer_changeLearnRate01 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])
# model = MultipleLayerModel([
#     AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
#     ReluLayer(),
#     AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
#     ReluLayer(),
#     AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
# ])

error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=0.1)
learning_rule_change = GradientDescentLearningRule(learning_rate=0.01)
learning_rule_change01 = GradientDescentLearningRule(learning_rate=0.001)

#Remember to use notebook=False when you write a script to be run in a terminal
print('ReluLayer learning rate 0.1\n\n')
[stats_ReluLayer, keys_ReluLayer] = train_model_and_plot_stats(
    model_ReluLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('ReluLayer learning rate 0.01\n\n')
[stats_ReluLayerLearnRate, keys_ReluLayerLearnRate] = train_model_and_plot_stats(
    model_ReluLayer_changeLearnRate, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('ReluLayer learning rate 0.001\n\n')
[stats_ReluLayerLearnRate01, keys_ReluLayerLearnRate01] = train_model_and_plot_stats(
    model_ReluLayer_changeLearnRate01, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='Learning_rate:0.1')
    ax_1.plot(np.arange(1, stats_ReluLayerLearnRate.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate[1:, keys_ReluLayer[k]], label='Learning_rate:0.01')
    ax_1.plot(np.arange(1, stats_ReluLayerLearnRate01.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate01[1:, keys_ReluLayer[k]], label='Learning_rate:0.001')

ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different learning rate of ReluLayer on training set')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('PartA1_ReluLearningRate_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='Learning_rate:0.1')
    ax_2.plot(np.arange(1, stats_ReluLayerLearnRate.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate[1:, keys_ReluLayer[k]], label='Learning_rate:0.01')
    ax_2.plot(np.arange(1, stats_ReluLayerLearnRate01.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate01[1:, keys_ReluLayer[k]], label='Learning_rate:0.001')

ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different learning rate of ReluLayer on valid dataset')
fig_2.tight_layout()  # This minimises whitespace around the axes.
fig_2.savefig('PartA1_ReluLearningRate_valid_error.pdf')  # Save figure to current directory in PDF format
#

fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='Learning_rate:0.1')
    ax_3.plot(np.arange(1, stats_ReluLayerLearnRate.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate[1:, keys_ReluLayer[k]], label='Learning_rate:0.01')
    ax_3.plot(np.arange(1, stats_ReluLayerLearnRate01.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate01[1:, keys_ReluLayer[k]], label='Learning_rate:0.001')
ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different learning rate of ReluLayer on training dataset')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('PartA1_ReluLearningRate_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='Learning_rate:0.1')
    ax_4.plot(np.arange(1, stats_ReluLayerLearnRate.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate[1:, keys_ReluLayer[k]], label='Learning_rate:0.01')
    ax_4.plot(np.arange(1, stats_ReluLayerLearnRate01.shape[0]) * stats_interval,
              stats_ReluLayerLearnRate01[1:, keys_ReluLayer[k]], label='Learning_rate:0.001')
ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different learning rate of ReluLayer on valid dataset')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('PartA1_ReluLearningRate_valid_acc.pdf') # Save figure to current directory in PDF format


# # PartA 1. Perform baseline experiments using DNNs trained on EMNIST Balanced.
# ## add the dropoutlayer (pro=0.4),activation function is still Relu, and learning rate is 0.1

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer,DropoutLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
incl_prob = 0.4
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

model_ReluLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_ReluLayer_addDropout = MultipleLayerModel([
    DropoutLayer(rng, incl_prob),
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng,incl_prob),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    DropoutLayer(rng, incl_prob),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# model = MultipleLayerModel([
#     AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
#     ReluLayer(),
#     AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
#     ReluLayer(),
#     AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
# ])

error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('ReluLayer\n\n')
[stats_ReluLayer, keys_ReluLayer] = train_model_and_plot_stats(
    model_ReluLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('ReluLayer add DropoutLayer\n\n')
[stats_ReluLayerDropout, keys_ReluLayerDropout] = train_model_and_plot_stats(
    model_ReluLayer_addDropout, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')
    ax_1.plot(np.arange(1, stats_ReluLayerDropout.shape[0]) * stats_interval,
              stats_ReluLayerDropout[1:, keys_ReluLayerDropout[k]], label='ReluLayerAddDropout')

ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of ReluLayer add DropoutLayer on training dataset')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('PartA1_ReluAddDropout_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_1.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')
    ax_1.plot(np.arange(1, stats_ReluLayerDropout.shape[0]) * stats_interval,
              stats_ReluLayerDropout[1:, keys_ReluLayerDropout[k]], label='ReluLayerAddDropout')

ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of ReluLayer add DropoutLayer on valid dataset')
fig_2.tight_layout()  # This minimises whitespace around the axes.
fig_2.savefig('PartA1_ReluAddDropout_valid_error.pdf')  # Save figure to current directory in PDF format
#

fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_1.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')
    ax_1.plot(np.arange(1, stats_ReluLayerDropout.shape[0]) * stats_interval,
              stats_ReluLayerDropout[1:, keys_ReluLayerDropout[k]], label='ReluLayerAddDropout')
ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of ReluLayer add DropoutLayer on training dataset')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('PartA1_ReluAddDropout_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_1.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
              stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')
    ax_1.plot(np.arange(1, stats_ReluLayerDropout.shape[0]) * stats_interval,
              stats_ReluLayerDropout[1:, keys_ReluLayerDropout[k]], label='ReluLayerAddDropout')
ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of ReluLayer add DropoutLayer on valid dataset')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('PartA1_ReluAddDropout_valid_acc.pdf') # Save figure to current directory in PDF format






# ## PartA 3. Perform experiments to compare stochastic gradient descent, RMSProp, and Adam for deep neural network,training on EMNIST Balanced, building on your earlier baseline experiments. In this experiment, I set the learning rate to 0.1.
# 

# In[4]:


from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,RMSPropLearningRule,AdamLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)


model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])



error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule_Gradient = GradientDescentLearningRule(learning_rate=learning_rate)
learning_rule_RMS = RMSPropLearningRule(learning_rate=learning_rate)
learning_rule_Adam = AdamLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('learning_rule_Gradient\n\n')
[stats_Gradient, keys_Gradient] = train_model_and_plot_stats(
    model, error, learning_rule_Gradient, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('learning_rule_RMS\n\n')
[stats_RMS, keys_RMS] = train_model_and_plot_stats(
    model, error, learning_rule_RMS, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('learning_rule_Adam\n\n')
[stats_Adam, keys_Adam] = train_model_and_plot_stats(
    model, error, learning_rule_Adam, train_data, valid_data, num_epochs, stats_interval, notebook=True)



 # Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_1.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_1.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')

ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different learning rules in the training dataset (learning rate is 0.1) ')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('PartA3_learning_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_2.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_2.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')


ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different learning rules in the training dataset (learning rate is 0.1) ')
fig_2.tight_layout() # This minimises whitespace around the axes.
fig_2.savefig('PartA3_learning_valid_error.pdf') # Save figure to current directory in PDF format
#


fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_3.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_3.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')


ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different learning rules in the training dataset (learning rate is 0.1) ')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('PartA3_learning_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_4.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_4.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')

ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different learning rules in the training dataset (learning rate is 0.1) ')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('PartA3_learning_valid_acc.pdf') # Save figure to current directory in PDF format


# ## PartA 3.  In this experiment, I set the learning rate to 0.01.
# 

# In[5]:


from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,RMSPropLearningRule,AdamLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.01
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)


model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])



error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule_Gradient = GradientDescentLearningRule(learning_rate=learning_rate)
learning_rule_RMS = RMSPropLearningRule(learning_rate=learning_rate)
learning_rule_Adam = AdamLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('learning_rule_Gradient\n\n')
[stats_Gradient, keys_Gradient] = train_model_and_plot_stats(
    model, error, learning_rule_Gradient, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('learning_rule_RMS\n\n')
[stats_RMS, keys_RMS] = train_model_and_plot_stats(
    model, error, learning_rule_RMS, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('learning_rule_Adam\n\n')
[stats_Adam, keys_Adam] = train_model_and_plot_stats(
    model, error, learning_rule_Adam, train_data, valid_data, num_epochs, stats_interval, notebook=True)



 # Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_1.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_1.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')

ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different learning rules in the training dataset (learning rate is 0.01) ')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('PartA3_learning_train_error_001.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_2.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_2.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')


ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different learning rules in the training dataset (learning rate is 0.01) ')
fig_2.tight_layout() # This minimises whitespace around the axes.
fig_2.savefig('PartA3_learning_valid_error_001.pdf') # Save figure to current directory in PDF format
#


fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_3.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_3.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')


ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different learning rules in the training dataset (learning rate is 0.01) ')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('PartA3_learning_train_acc_001.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats_Gradient.shape[0]) * stats_interval,
            stats_Gradient[1:, keys_Gradient[k]], label='learning_rule_Gradient')
    ax_4.plot(np.arange(1, stats_RMS.shape[0]) * stats_interval,
            stats_RMS[1:, keys_RMS[k]], label='learning_rule_RMS')

    ax_4.plot(np.arange(1, stats_Adam.shape[0]) * stats_interval,
            stats_Adam[1:, keys_Adam[k]], label='learning_rule_Adam')

ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different learning rules in the training dataset (learning rate is 0.01) ')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('PartA3_learning_valid_acc_001.pdf') # Save figure to current directory in PDF format
#


# ## PartA 7. Perform experiments on EMNIST Balanced to investigate the impact of using batch normalisation in deep neural networks, building on your earlier experiments.

# In[6]:


from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,RMSPropLearningRule,AdamLearningRule
from mlp.optimisers import Optimiser
from mlp.layers import BatchNormalizationLayer
#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)



model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),

    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),

    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_BatchNormalizationLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    BatchNormalizationLayer(hidden_dim),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    BatchNormalizationLayer(hidden_dim),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])



error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule_Gradient = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('NoBatchNormalization_rule_Gradient\n\n')
[stats, keys] = train_model_and_plot_stats(
    model, error, learning_rule_Gradient, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('BatchNormalization_rule_Gradient\n\n')
[stats_BatchNormalization, keys_BatchNormalization] = train_model_and_plot_stats(
    model, error, learning_rule_Gradient, train_data, valid_data, num_epochs, stats_interval, notebook=True)


 # Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval,
            stats[1:, keys[k]], label='NoBatchNormalization_rule_Gradient')
    ax_1.plot(np.arange(1, stats_BatchNormalization.shape[0]) * stats_interval,
            stats_BatchNormalization[1:, keys_BatchNormalization[k]], label='BatchNormalization_rule_Gradient')


ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different BatchNormalizations in the training dataset ')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('PartA7_BatchNormalizations_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval,
            stats[1:, keys[k]], label='NoBatchNormalization_rule_Gradient')
    ax_2.plot(np.arange(1, stats_BatchNormalization.shape[0]) * stats_interval,
            stats_BatchNormalization[1:, keys_BatchNormalization[k]], label='BatchNormalization_rule_Gradient')



ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different BatchNormalizations in the training dataset')
fig_2.tight_layout() # This minimises whitespace around the axes.
fig_2.savefig('PartA7_BatchNormalizations_valid_error.pdf') # Save figure to current directory in PDF format
#


fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats.shape[0]) * stats_interval,
            stats[1:, keys[k]], label='NoBatchNormalization_rule_Gradient')
    ax_3.plot(np.arange(1, stats_BatchNormalization.shape[0]) * stats_interval,
            stats_BatchNormalization[1:, keys_BatchNormalization[k]], label='BatchNormalization_rule_Gradient')



ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different BatchNormalizations in the training dataset ')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('PartA7_BatchNormalizations_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats.shape[0]) * stats_interval,
            stats[1:, keys[k]], label='NoBatchNormalization_rule_Gradient')
    ax_4.plot(np.arange(1, stats_BatchNormalization.shape[0]) * stats_interval,
            stats_BatchNormalization[1:, keys_BatchNormalization[k]], label='BatchNormalization_rule_Gradient')


ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different BatchNormalizations in the training dataset ')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('PartA7_BatchNormalizations_valid_acc.pdf') # Save figure to current directory in PDF format


# ## PartB 5. Construct and train networks containing one and two convolutional layers, and max-pooling layers,
# using the EMNIST Balanced data, reporting your experimental results. As a default use convolutional
# kernels of dimension 55 (stride 1) and pooling regions of 22 (stride 2, hence non-overlapping).

# In[7]:




from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer,MaxPoolingLayer,ReshapeLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,RMSPropLearningRule,AdamLearningRule
from mlp.optimisers import Optimiser
from mlp.layers import ConvolutionalLayer



#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

train_data.reset()
valid_data.reset()


input_dim1 = 28
input_dim2 = 28

kernel_dim_1 = 5
kernel_dim_2 = 5

featureMaps1 = 5
featureMaps2 = 10

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# activation_layer = ConvolutionalLayer(num_input_channels=3, num_output_channels=2, input_dim_1=4, input_dim_2=4,
#                                       kernel_dim_1=2, kernel_dim_2=2)

print('build model')
model = MultipleLayerModel([

    ReshapeLayer(output_shape=(1,28,28)),

    ConvolutionalLayer(num_input_channels=1, num_output_channels=featureMaps1, input_dim_1=input_dim1, input_dim_2=input_dim2,
                                      kernel_dim_1=kernel_dim_1, kernel_dim_2=kernel_dim_2),

    ReluLayer(),

    MaxPoolingLayer(pool_size=2),

    ReshapeLayer(),
    AffineLayer(5*12*12, output_dim, weights_init, biases_init)
])



error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule_Gradient = GradientDescentLearningRule(learning_rate=learning_rate)
learning_rule_RMS = RMSPropLearningRule(learning_rate=learning_rate)
learning_rule_Adam = AdamLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('learning_rule_Gradient\n\n')
[stats_Gradient, keys_Gradient] = train_model_and_plot_stats(
    model, error, learning_rule_Gradient, train_data, valid_data, num_epochs, stats_interval, notebook=True)

