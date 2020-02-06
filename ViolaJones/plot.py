"""
Functions for plotting stuff.
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

import matplotlib.pyplot as plt

def showImage(img, colorBar=False):
    """
    Plot image.
    
    :param img: Image
    :param colorBar: Whether to display a colorbar or not.
    """
    plt.imshow(img, cmap='gray')
    if colorBar:
        plt.colorbar()
    plt.show()

def showGrid(rows, columns, images=[], titles=[], colorBars=False, colormap=None):
    """
    Show a grid of images.
    
    :param rows: Number of rows.
    :param columns: Number of columns.
    :param images: Images
    :param titles: Image titles
    :param colorBars: Whether to display colorbars or not.
    :param colormap: Colormap.
    """
    fig, a = plt.subplots(rows, columns)
    fig.tight_layout()

    index = 0
    for i in range(0, rows):
        for k in range(0, columns):
            if colormap is None:
                plot = a[i][k].imshow(images[index])
            else:
                plot = a[i][k].imshow(images[index], cmap=colormap)

            if colorBars:
                fig.colorbar(plot, ax=a[i][k])
            index += 1
            
    plt.show()
