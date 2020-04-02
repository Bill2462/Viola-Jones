"""
Routines for training.
"""

# This file is part of a Viola Jones example.
# Copyright (c) 2020 Krzysztof Adamkiewicz <kadamkiewicz835@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the “Software”), to deal in the
# Software without restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the
# following conditions: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
# OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from .haar import FeatureTypes, Haar
from .classifier import WeakClassifier
from multiprocessing import Pool
import cv2
import numpy as np

def genAllFeatures(img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
    """
    Generate all features from the given space.
    
    :param img_height: Detection frame height.
    :param img_width: Detection frame width.
    :param min_feature_width: Minimal feature width.
    :param max_feature_width: Maximal feature width.
    :param min_feature_height: Minimal feature height.
    :param max_feature_height: Maximal feature height.
    
    :return: Haar like feature objects covering the specified space.
    """
    features = []
    for feature in FeatureTypes:
        # FeatureTypes are just tuples
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
            feature_start_height = max(min_feature_height, feature[1])
            for feature_height in range(feature_start_height, max_feature_height, feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(Haar(feature, (x, y), feature_width, feature_height))
    return features

def normWeights(weights):
    """
    Normalize weights.
    
    :param weights: Weights
    :return: Normalized weights.
    """
    return weights * (1. / np.sum(weights))

def initWeights(positive_count, negative_count, trainingSamples):
    """
    Initialise weights.
    
    :param positive_count: Number of positive samples.
    :param negative_count: Number of negative samples.
    :param trainingSamples: Training samples.
    :return: Weights
    """
    positiveSampleInitialWeight = 1/(2*positive_count)
    negativeSampleInitialWeight = 1/(2*negative_count)
    weights = []
    for sample in trainingSamples:
        if sample[1] == 1:
            weights.append(positiveSampleInitialWeight)
        else:
            weights.append(negativeSampleInitialWeight)
    
    return np.array(weights)
    

def trainWeakClassifier(trainingSamples, weights, feature):
    """
    Train weak classifier based on a given feature.
    
    :param trainingSamples: Training samples.
    :param weights: Weights corresponsig to each sample.
    :param feature: Features that we want to train.
    
    :return: Training weak classifier.
    """
    #compute feature values
    featureValues = []
    positiveOrNegative = []
    for sample in trainingSamples:
        featureValues.append(feature.computeScore(sample[0], 0, 0))
        positiveOrNegative.append(sample[1])
        
    #zip with weights and sort by feature value
    featureValues = zip(featureValues, weights, positiveOrNegative)
    featureValues = sorted(featureValues, key=lambda tup: tup[0])
    
    #sum all weights of the positive and negative samples
    negativeWeightsTotal = 0
    positiveWeightsTotal = 0
    for value in featureValues:
        if value[2] == 1:
            positiveWeightsTotal += value[1]
        else:
            negativeWeightsTotal += value[1]
    
    #find the feature with the smallest error
    bestFeatureIndex = 0
    bestFeatureError = 1e10
    negativeWeightsSoFar = 0
    positiveWeightsSoFar = 0
    positiveOnTheLeft = 0
    positivesTotal = 0
    for i in range(0, len(featureValues)):
        error1 = positiveWeightsSoFar-negativeWeightsSoFar+negativeWeightsTotal
        error2 = negativeWeightsSoFar-positiveWeightsSoFar+positiveWeightsTotal
        error = min([error1, error2])
        
        if bestFeatureError > error:
            bestFeatureError = error
            bestFeatureIndex = i
            positiveOnTheLeft = positivesTotal
        
        if featureValues[i][2] == 1:
            positiveWeightsSoFar += featureValues[i][1]
            positivesTotal += 1
        else:
            negativeWeightsSoFar += featureValues[i][1]
    
    #count how much samples are there on the right
    positiveOnTheRight = positivesTotal - positiveOnTheLeft
    
    #determine the polarity and threshold
    polarity = -1
    threshold = featureValues[bestFeatureIndex][0]
    if positiveOnTheLeft > positiveOnTheRight:
        polarity = 1
    else:
        polarity = -1
    
    #build and return a weak classifier
    return WeakClassifier(feature, threshold, polarity)

def computeError(trainingSamples, weights, classifier):
    """
    Compute error.
    
    :param trainingSamples: Training samples.
    :param weights: Sample weights.
    :param classifier: Weak classifier.
    
    :return: Error
    """
    error = 0
    for i in range(0, len(trainingSamples)):
        sample = trainingSamples[i]
        error += weights[i]*(classifier.classify(sample[0], 0, 0) != sample[1])
        
    # used dataset is very small so it is possible to have error = 0, in this case set error to small value so program
    # does not fail
    if error == 0:
        error = 0.001
    return error

"""
Training worker class.
"""
class Trainer():
    """
    :param trainingSamples: Training samples.
    :param weights: Weights.
    :param features: Features.
    """
    def __init__(self, trainingSamples, weights, features):
        self.trainingSamples = trainingSamples
        self.weights = weights
        self.features = features
        
    def run(self):
        """
        Run worker.
        """
        self.classifiers = []
        for feature in self.features:
            self.classifiers.append(trainWeakClassifier(self.trainingSamples, self.weights, feature))
            
        return self

    def getResults(self):
        """
        Get the result of the training.
        
        :return: Trained classifiers.
        """
        return self.classifiers

"""
Error computation worker.
"""
class ErrorWorker():
    def __init__(self, trainingSamples, weights, classifiers):
        self.trainingSamples = trainingSamples
        self.weights = weights
        self.classifiers = classifiers
        
    def run(self):
        self.errors = []
        for classifier in self.classifiers:
            self.errors.append(computeError(self.trainingSamples, self.weights, classifier))
            
        return self

    def getResults(self):
        return self.errors

def workerRunner(worker):
    """
    Run a worker.
    
    :return: Finished worker.
    """
    return worker.run()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def getResults(workers):
    """
    Converts a list of finished workers into a result.
    
    :param workers: Finished workers.
    :return: Results
    """
    results = []
    for worker in workers:
        results += worker.getResults()
        
    return results

def adaBoost(trainingSamples, weights, featuresSet, classifierCount, threadCount=12, verbose=True):
    """
    Train a strong classifier using adaptive boosting.
    
    :param trainingSamples: Training samples.
    :param weights: Sample weights.
    :param featuresSet: List of haar like features.
    :param classifierCount: Number of classifiers that will be included in the strong classifiers.
    :param threadCount: Number of threads for which we want to.
    :param verbose: Display progress during training.
    
    :return: List of
     - trained weak classifiers,
     - weak classifier weights
     - feature set with used features removed
     - weights current weights
    """
    #result
    classifiers = []
    classifierWeights = []
    
    for i in range(0, classifierCount):
        if verbose:
            print("Training classifier " + str(i+1) + "/" + str(classifierCount) + "....")

        #normalize weights
        weights = normWeights(weights)
    
        ########## train all classifiers ##########
        #build workers
        workers = []
        for chunk in chunks(featuresSet, threadCount):
            workers.append(Trainer(trainingSamples, weights, chunk))

        #run them in a thread pool
        trainedClassifiers = []
        with Pool(processes=threadCount) as pool:
            trainedClassifiers = pool.map(workerRunner, workers)
        trainedClassifiers = getResults(trainedClassifiers)

        ########## compute error for each classifier ##########
        #build workers
        workers = []
        for chunk in chunks(trainedClassifiers, threadCount):
            workers.append(ErrorWorker(trainingSamples, weights, chunk))
        
        #run them in a thread pool
        errors = []
        with Pool(processes=threadCount) as pool:
            errors = pool.map(workerRunner, workers)
        errors = getResults(errors)

        #choose the classifier with the lowest error
        errors = np.array(errors) 
        bestError = errors.min()
        bestClassifierIndex = np.where(errors == errors.min())[0][0]
        bestClassifier = trainedClassifiers[bestClassifierIndex]
        b = bestError/(1-bestError)
        classifiers.append(bestClassifier)
        if verbose:
            print("Found feature with error: " + str(bestError))
        
        #remove used feature from the dataset
        del featuresSet[bestClassifierIndex]
        
        #compute classifier weight
        classifierWeight = np.log(1/b)
        classifierWeights.append(classifierWeight)
        if verbose:
            print("Classifier has weight: " + str(classifierWeight))
        
        #update weights
        for i in range(0, len(weights)):
            weights[i] = weights[i]*np.power(b, 1-bestError)

    return classifiers, classifierWeights
