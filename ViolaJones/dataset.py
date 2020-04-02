"""
Functions associated with the dataset.
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

from os import listdir
from os.path import isfile, join
from .integralImage import IntegralImage
import numpy as np
import cv2
import random

def cropImage(img, x, y, width, height):
    """
    Crop the image.

    :param img: Image
    :type img: Numpy array.
    
    :param x: Position of the sub image, X
    :type x: int

    :param y: Position of the sub image, Y
    :type y: int
    
    :param width: Width of the sub image.
    :type width: int
    
    :param height: Height of the sub image.
    :type height: int
    """
    return np.array(img[y:y+height, x:x+width])

def random_crop(img, x, y, width, height):
    """
    Randomly crop the image. 

    :param img: Image
    :type img: Numpy array.
    
    :param x: Position of the sub image, X
    :type x: int

    :param y: Position of the sub image, Y
    :type y: int
    
    :param width: Width of the sub image.
    :type width: int
    
    :param height: Height of the sub image.
    :type height: int
    """
    x = random.randint(x[0], x[1])
    y = random.randint(y[0], y[1])
    return cropImage(img, x, y, width, height)

def load_all_images_from_dir(dir):
    """
    Load all images from directory.
    
    :param dir: Directory from which 
    :type dir: Str
    """
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    images = []
    for path in files:
        img = cv2.imread(dir+"/"+str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    return images

def scale_images(images, width, height):
    """
    Scale all images.
    
    :param images: Array of all images.
    :param width: Output images width.
    :param height: Output images height.
    
    :return: Resized images.
    """
    processedImages = []
    for img in images:
        processedImages.append(cv2.resize(img, (width, height)))
    
    return np.array(processedImages)


def convert_to_grayScale(images):
    """
    Convert an array of images to grays scale.
    
    :param images: Array of images.
    :return: Array of images converted to grayscale.
    """
    processedImages = []
    for img in images:
        processedImages.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    return np.array(processedImages)

def normalize(images):
    """
    Normalize all samples.
    
    :param images: Array of images.
    :return: Normalized images.
    """
    normalized = []
    for img in images:
        img = img.astype(np.float32) / 255
        img -= img.mean()
        if img.std() == 0:
            img /= 1
        else:
            img /= img.std()
        normalized.append(img)
    return normalized

def integrateImages(images):
    """
    Integrate the array of images.
    
    :param images: Input images.
    
    :return: Integral images.
    """
    integralImages = []
    for img in images:
        integralImages.append(IntegralImage(img))

    return integralImages

def buildDataset(rootDir):
    """
    Build and preprocess entire dataset.
    
    :param rootDir: Path to the dataset root directory.
    :type rootDir: str
    """
    dataset = HandsDataset(rootDir)
    dataset.loadFiles()
    dataset.toGrayScale()
    dataset.genNegativeSamples()
    dataset.resize()
    dataset.normalize()
    dataset.integrate()
    dataset.split()
    return dataset

"""
Image dataset class.
"""
class HandsDataset():
    """
    :param datasetDir: Dataset root directory.
    :type datasetDir: Str
    """
    def __init__(self, datasetDir):
        self.rootDir = datasetDir
        
    def loadFiles(self):
        """
        Load all file.
        """
        self.positiveSamples = load_all_images_from_dir(self.rootDir+"/positive")
        self.negativeImages = load_all_images_from_dir(self.rootDir+"/negative")
    
    def toGrayScale(self):
        """
        Convert all images to grayscale.
        """
        self.positiveSamples = convert_to_grayScale(self.positiveSamples)
        self.negativeImages = convert_to_grayScale(self.negativeImages)
    
    def genNegativeSamples(self):
        """
        Generate negative detection frames.
        """
        self.negativeSamples = []
        for img in self.negativeImages:
            self.negativeSamples.append(random_crop(img, (0, img.shape[1]-42), (0, img.shape[0]-42), 24, 24))
            self.negativeSamples.append(random_crop(img, (0, img.shape[1]-42), (0, img.shape[0]-42), 24, 24))
    
    def resize(self):
        """
        Resize all samples to the size.
        """
        self.positiveSamples = scale_images(self.positiveSamples, 24, 24)
        self.negativeSamples = scale_images(self.negativeSamples, 24, 24)
    
    def normalize(self):
        """
        Normalize all samples. 
        """
        self.positiveSamples = normalize(self.positiveSamples)
        self.negativeSamples = normalize(self.negativeSamples)
    
    def integrate(self):
        """
        Integrate all images. 
        """
        self.positiveSamples = integrateImages(self.positiveSamples)
        self.negativeSamples = integrateImages(self.negativeSamples)
    
    def split(self):
        """
        Make a split into training and testing dataset.
        """
        #split dataset into training and test dataset
        positive_train = self.positiveSamples[len(self.positiveSamples)//3:]
        negative_train = self.negativeSamples[len(self.negativeSamples)//3:]
        positive_test  = self.positiveSamples[:len(self.positiveSamples)//3]
        negative_test  = self.negativeSamples[:len(self.negativeSamples)//3]
        
        self.positive_train_count = len(positive_train)
        self.negative_train_count = len(negative_train)
        
        self.trainingDataset = list(zip(positive_train, np.ones(len(positive_train), dtype=int)))
        self.trainingDataset += list(zip(negative_train, np.zeros(len(negative_train), dtype=int)))
        
        self.testDataset = list(zip(positive_test, np.ones(len(positive_test), dtype=int)))
        self.testDataset += list(zip(negative_test, np.zeros(len(negative_test), dtype=int)))

