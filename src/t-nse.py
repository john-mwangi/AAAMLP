import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml(name="mnist_784", version=1, return_X_y=True)

pixel_values, targets = data
type(targets)
type(pixel_values)
pixel_values.shape
targets = targets.astype(int)

# Images are of size 78x78 pixels which when flattened is 784
# 70,000 observations are present.

pixel_values = pixel_values.to_numpy()
single_image = pixel_values[0,:].reshape(28,28)
plt.imshow(single_image)