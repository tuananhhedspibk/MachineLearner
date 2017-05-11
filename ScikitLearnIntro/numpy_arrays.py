import numpy as np

a = np.array([1, 2, 3]) # create a rank 1 array
#print type(a)
#print a.shape
#print a[0], a[1], a[2]

b = np.array([[1, 2, 3], [4, 5, 6]])
#print b.shape
#print b[0, 0], b[0, 1], b[1, 0]

a = np.zeros(2)
#print a

b = np.ones((2, 2, 2))
#print b

c = np.full((2,2), 7)
#print c

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = a[:2, 1:3]
#print b
# b is a subarray with 2 first rows and column (1, 2)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Two way to accessing data of middle row in array
# First: use integer indexing + slice indexing: yield an array with lower rank than original array
# Second: use slice indexing: yield and array with the same rank as original array

row_r1 = a[1, :]
row_r2 = a[1:2, :]
#print row_r1, row_r1.shape
#print row_r2, row_r2.shape
#print a.shape

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
#print col_r1, col_r1.shape
#print col_r2, col_r2.shape

#print a

a = np.array([[1, 2], [3, 4], [5, 6]])
#print a
#print a[[0, 1, 2], [0, 1, 0]]

#print np.array([a[0, 0], a[1, 1], a[2, 0]])
#print np.array([1, 3, 5])

#print a[[0, 0], [1, 1]]

# Integer array indexing

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([[[1, 2, 3], [4 ,5, 6]],[[7, 8, 9], [10, 11, 12]]])

#print a

b = np.array([0, 2, 0, 1]) # array of indicates

#print a[np.arange(4), b]

a[np.arange(4), b] += 10

#print a

# Boolean array indexing

a = np.array([[1, 2], [3, 4], [5, 6]])

bool_idx = (a > 2)	# This array has same shape with a and value of each element is boolean value represent that (a's element > 2?)

#print bool_idx

#print a[bool_idx]

#print a[a > 2]

# Datatypes

x = np.array([1, 2])
#print x.dtype

x = np.array([1.0, 2.0])
#print x.dtype

x = np.array([1, 2], dtype=np.int64)
#print x.dtype

# Array math

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

#print x + y
#print np.add(x, y)

#print x - y
#print np.subtract(x, y)

#print x * y
#print np.multiply(x, y)

#print x / y
#print np.divide(x, y)

#print np.sqrt(x)

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

#print v.dot(w)
#print np.dot(v, w)

#print x.dot(v)
#print np.dot(x, v)

#print x.dot(y)
#print np.dot(x, y)

# Sum of array

x = np.array([[1, 2], [3, 4]])

#print np.sum(x)
#print np.sum(x, axi = 0)
#print np.sum(x, axis = 1)

x = np.array([[1, 2], [3, 4]])

#print x
#print x.T

v = np.array([1, 2, 3])
#print v
#print v.T

# Broadcasting

# add vector v to each row of x and storing result at y

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)	# create an empty array with same shape as X

for i in range(4):
	y[i, :] = x[i, :] + v

#print y

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4,1))

#print vv
y = x + vv
#print y

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([1, 0, 1])

y = x + v
#print y

# Scipy

from scipy.misc import imread, imsave, imresize

# Read an image to numpy arrays
img = imread("image.jpg")
#print img.dtype, img.shape

img_tinted = img * [1, 0.95, 0.9]

img_tinted = imresize(img_tinted, (300, 300))

imsave("image_tinted.jpg", img_tinted)

from scipy.spatial.distance import pdist, squareform

x = np.array([[0, 1], [1, 0], [0, 2]])
#print x

d = squareform(pdist(x, "euclidean"))
#print d

# Mathplotlib

import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
#plt.show()

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.title("Sin and Cos")
plt.legend(["Sin", "Cos"])
#plt.show()

# subplot

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# set up a subplot grid that has height 2, width 1, set first subplot as active
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title("Sin")

# set up second subplot as active
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title("Cos")

#plt.show()

# Images

from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread("image.jpg")
img_tinted = img * [1, 0.95, 0.9]

plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))
plt.show()




