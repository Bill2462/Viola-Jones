"""
Functions for computing haar.
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

def enum(**enums):
    return type('Enum', (), enums)

FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]

class Haar():
    """
    :param haarType: Type of the haar feature.
    :param position: Position of the feature (x,y) inside the detection window.
    :param width: Width of the haar feature.
    :param height: Height of the haar feature.
    """
    def __init__(self, haarType, position, width, height):
        self.type = haarType
        
        #corners
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        
        #sizes
        self.width = width
        self.height = height

    def setScale(self, scale):
        """
        Set scale. All dimensions will be rescaled by this scale.
        :param scale: Scale
        """
        
        #there is some bug here but I don't have to find it
        # basically just multiply all size parameters by scale and then round
        #self.top_left = (int(self.top_left[0]*scale), int(self.top_left[1]*scale))
        #self.bottom_right = (int(self.bottom_right[0]*scale), int(self.bottom_right[1]*scale))
        #self.width = int(self.width*scale)
        #self.height = int(self.height*scale)
        pass

        """
        :param int_img: Integral image object
        :param x: Position of the detection frame (X)
        :param y: Position of the detection frame (Y)
        """
    def computeScore(self, int_img, x, y):
        score = 0
        
        #compute new corners to account for the position of the detection frame
        top_left = (self.top_left[0] + x, self.top_left[1] + y)
        bottom_right = (self.bottom_right[0] + x, self.bottom_right[1] + y)
        
        if self.type == FeatureType.TWO_VERTICAL:
            first = int_img.sumRegion(top_left, (top_left[0] + self.width, int(top_left[1] + self.height / 2)))
            second = int_img.sumRegion((top_left[0], int(top_left[1] + self.height / 2)), bottom_right)
            score = first - second
            
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = int_img.sumRegion(top_left, (int(top_left[0] + self.width / 2), top_left[1] + self.height))
            second = int_img.sumRegion((int(top_left[0] + self.width / 2), top_left[1]), bottom_right)
            score = first - second
            
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = int_img.sumRegion(top_left, (int(top_left[0] + self.width / 3), top_left[1] + self.height))
            second = int_img.sumRegion((int(top_left[0] + self.width / 3), top_left[1]), (int(top_left[0] + 2 * self.width / 3), top_left[1] + self.height))
            third = int_img.sumRegion((int(top_left[0] + 2 * self.width / 3), top_left[1]), bottom_right)
            score = first - second + third
            
        elif self.type == FeatureType.THREE_VERTICAL:
            first = int_img.sumRegion(top_left, (bottom_right[0], int(top_left[1] + self.height / 3)))
            second = int_img.sumRegion((top_left[0], int(top_left[1] + self.height / 3)), (bottom_right[0], int(top_left[1] + 2 * self.height / 3)))
            third = int_img.sumRegion((top_left[0], int(top_left[1] + 2 * self.height / 3)), bottom_right)
            score = first - second + third
            
        elif self.type == FeatureType.FOUR:
            first = int_img.sumRegion(top_left, (int(top_left[0] + self.width / 2), int(top_left[1] + self.height / 2)))
            second = int_img.sumRegion((int(top_left[0] + self.width / 2), top_left[1]), (bottom_right[0], int(top_left[1] + self.height / 2)))
            third = int_img.sumRegion((top_left[0], int(top_left[1] + self.height / 2)), (int(top_left[0] + self.width / 2), bottom_right[1]))
            fourth = int_img.sumRegion((int(top_left[0] + self.width / 2), int(top_left[1] + self.height / 2)), bottom_right)
            score = first - second - third + fourth
            
        return score
