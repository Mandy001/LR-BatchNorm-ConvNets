import matplotlib.pyplot as plt

# get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')


def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

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
# create the test date
test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)


from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer,DropoutLayer
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


model_ReluLayer_addDropout = MultipleLayerModel([
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
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('ReluLayer add DropoutLayer\n\n')
[stats_ReluLayerDropout, keys_ReluLayerDropout] = train_model_and_plot_stats(
    model_ReluLayer_addDropout, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

# Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
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
    ax_1.plot(np.arange(1, stats_ReluLayerDropout.shape[0]) * stats_interval,
              stats_ReluLayerDropout[1:, keys_ReluLayerDropout[k]], label='ReluLayerAddDropout')
ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of ReluLayer add DropoutLayer on valid dataset')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('PartA1_ReluAddDropout_valid_acc.pdf') # Save figure to current directory in PDF format