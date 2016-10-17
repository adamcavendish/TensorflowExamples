from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST/", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

# Get the next 64 images array and labels
batch_X, batch_Y = mnist.train.next_batch(100)

# Image Width and Image Height
iw, ih = 28, 28
nrows, ncols = 10, 10

# Add a figure to plot the data, like the draw panel
figure = plt.figure()
for i in range(nrows):
    for j in range(ncols):
        # Add subplot to plot images individually
        figure.add_subplot(nrows, ncols, i*ncols+j+1)
        # Reshape image to original size
        image = batch_X[i*ncols+j].reshape(iw, ih)
        # Plot the grayscaled image instead of the colored ones
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        # Do not show the axis
        plt.axis('off')
# Show the panel on the screen
plt.show()

