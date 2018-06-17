import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 loss='bce', learning_rate=0.01, epochs=50):
        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learning_rate
        self.epochs = epochs
        self.outputTask = output_task  # Either classification or regression
        self.outputActivation = output_activation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        self.error = np.zeros((10, 1))

        self.last_layer_activations = np.array([])
    
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'ce':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        input_activation = "sigmoid"
        n_out = layers[0].n_in if (layers is not None and len(layers) > 0) else 128
        self.layers.append(LogisticLayer(train.input.shape[1], n_out,
                           None, input_activation, False))

        # Add passed layers
        if layers is not None:
            self.layers += layers

        hidden_ac = "sigmoid"
        self.layers.append(LogisticLayer(self.layers[-1].n_out, 32,
                           None, hidden_ac, False))
                           
        # Output layer
        output_activation = "softmax"
        self.layers.append(LogisticLayer(32, 10,
                                         weights=None, activation=output_activation, is_classifier_layer=True))

        self.inputWeights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                           axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                             axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """

        activations = inp
        for curr_layer in self.layers:
            activations = curr_layer.forward(activations)
            activations = np.insert(activations, 0, 1)

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateError(target, self._get_output_layer().outp) 
    
    def _update_weights(self, learning_rate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def _backpropagate_error(self, target):
        """
        Perform the backpropagation of error through the network layers
        """
        # first deltas are the actual error gradients at the output layer
        current_deltas = self.loss.calculateDerivative(target, self._get_output_layer().outp)
        # identity matrix as required by the computeDerivative() function
        current_weights = 1.0
        # loop over reversed list since we have to start at the output layer
        for layer in reversed(self.layers):
            # get deltas of current layer
            current_deltas = layer.compute_derivative(current_deltas, current_weights)
            # remove bias weight since we don't need it here
            current_weights = layer.weights[1:,:].T

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            for img, label in zip(self.trainingSet.input, self.trainingSet.label):

                # perform backpropagation
                self._feed_forward(img)
                target = np.zeros((10,))
                target[label] = 1
                self._backpropagate_error(target)
                self._update_weights(self.learningRate)

            if verbose:

                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)
        return np.argmax(self._get_output_layer().outp)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
