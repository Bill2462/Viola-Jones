"""
Integral image.
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

import cv2

"""
Integral image.
"""
class IntegralImage():
    """
    :param image: Grayscale image to be integrated.
    """
    def __init__(self, image):
        self.intImage = cv2.integral(image)

    def sumRegion(self, top_left, bottom_right):
        """
        Calculates the sum in the rectangle specified by the given tuples.
        :param top_left: (x, y) of the rectangle's top left corner
        :type top_left: (int, int)
        :param bottom_right: (x, y) of the rectangle's bottom right corner
        :type bottom_right: (int, int)
        :return The sum of all pixels in the given rectangle
        :rtype int
        """
        top_left = (top_left[1], top_left[0])
        bottom_right = (bottom_right[1], bottom_right[0])
        
        if top_left == bottom_right:
            return self.intImage[top_left]
        
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])
        return self.intImage[bottom_right] - self.intImage[top_right] - self.intImage[bottom_left]+  self.intImage[top_left]
