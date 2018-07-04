#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression, LogisticLayer
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 100, 1000, 1000,
                      oneHot=False)

    my_mlp_classifier = MultilayerPerceptron(data.trainingSet, data.validationSet, data.testSet, loss='ce',
                                             epochs=150, learning_rate=0.03)

    print("=========================")

    # Train the classifiers
    print("\nTraining MLP classifier...")
    my_mlp_classifier.train()
    print("Finished training...")

    mlp_pred = my_mlp_classifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()
    print("\nResult of the MLP recognizer:")
    #evaluator.printComparison(data.testSet, mlp_pred)
    evaluator.printAccuracy(data.testSet, mlp_pred)
    
    # Draw
    plot = PerformancePlot("MLP validation")
    plot.draw_performance_epoch(my_mlp_classifier.performances,
                                my_mlp_classifier.epochs)
    
    
if __name__ == '__main__':
    main()
