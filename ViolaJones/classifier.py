"""
Classifier classes.
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

import numpy as np

class WeakClassifier():
    """
    :param haarFeature: Feature
    :param threshold: Threshold value
    :param polarity: Polarity
    """
    def __init__(self, haarFeature, threshold, polarity):
        self.feature = haarFeature
        self.threshold = threshold
        self.polarity = polarity
    
    def setScale(self, scale):
        """
        Set scale for the underlying haar feature.
        
        :param scale: Scale of the feature.
        """
        self.feature.setScale(scale)
    
    """
    :param int_image: Integral image
    :param x: Position of the detection frame (X)
    :param y: Position of the detection frame (Y)
    """
    def classify(self, int_image, x, y):
        return self.polarity*self.feature.computeScore(int_image, x, y) < self.polarity*self.threshold
    
class StrongClassifier():
    """
    Strong classifier.
    :param classifiers: Trained weak classifiers.
    :param weights: Weights corresponding to Trained weak classifiers.
    :param thresholdOffset: Offset added to the detection threshold.
    """
    def __init__(self, classifiers, weights, thresholdOffset=0):
        self.classifiers = classifiers
        self.weights = weights
        self.thresholdOffset = thresholdOffset
        
    def setScale(self, scale):
        """
        Set scale for all underlying classifiers.
        :param scale: scale
        """
        for i in range(0, len(self.classifiers)):
            self.classifiers[i].setScale(scale)
    
    def setThresholdOffset(self, offset):
        """
        Threshold offset is added to the detection threshold.
        :param thresholdOffset: Offset added to the detection threshold.
        """
        self.thresholdOffset = offset

    def classify(self, int_image, x, y):
        """
        :param int_image: Integral image
        :param x: Position of the detection frame (X)
        :param y: Position of the detection frame (Y)
        """
        #let each classifier vote
        score = 0
        for i in range(0, len(self.classifiers)):
            score += self.classifiers[i].classify(int_image, x, y)*self.weights[i]
        
        return score >= np.sum(self.weights) + self.thresholdOffset
