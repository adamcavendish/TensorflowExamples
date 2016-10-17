import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
num_training_epoch = 1000
display_step = 100

train_batch_size = 50

num_data = 1000
train_ratio = 0.7
test_ratio = 1 - train_ratio
num_train = int(num_data * train_ratio)
num_test = num_data - num_train

# Generate Training Data and Testing Data
W = -2.71828
b =  3.14159
print('Real W={}, b={}'.format(W, b))

# Generate X data in range(-10,10) in num_data size
X = np.linspace(-1, 1, num_data).reshape(-1, 1)
print('X shape:', X.shape)
mu, sigma = 0, 0.3
Y_noise = np.random.normal(mu, sigma, num_data).reshape(-1, 1)
Y = X*W + b + Y_noise
Y = Y.reshape(-1, 1)
print('Y shape:', Y.shape)

# Simple method for spliting train data and test data
combined = np.hstack((X, Y))
print('combined:', combined.shape)
np.random.shuffle(combined)
X_train = combined[:num_train, 0]
Y_train = combined[:num_train, 1]
X_test = combined[num_train:, 0]
Y_test = combined[num_train:, 1]
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
# Sophisticated Train Test Split: @see scikit-learn cross validation for reference
# URL: http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

# Tensorflow Train, Using CPU for now

with tf.device('/cpu:0'):
    # Computation Graph Input, placeholder is used to be binded at runtime
    # It is like a dummy variable that only has a type hint, but no value
    tf_X = tf.placeholder("float")
    tf_Y = tf.placeholder("float")
    
    # Set model weights
    tf_W = tf.Variable(np.random.randn(), name="weight")
    tf_b = tf.Variable(np.random.randn(), name="bias")
    
    # Construct a linear model. Use Y_pred = X*W+b for prediction
    tf_pred = tf.add(tf.mul(tf_X, tf_W), tf_b)
    
    # Mean squared error, aka MSE
    # URL: https://en.wikipedia.org/wiki/Mean_squared_error
    tf_cost = tf.reduce_mean(tf.square(tf_pred - tf_Y))
    
    # Use gradient descent to optimize the cost
    tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)
    
    # Always initializing the variables before running the graph
    init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(num_training_epoch):
        batched_X_train = X_train.reshape(train_batch_size, -1)
        batched_Y_train = Y_train.reshape(train_batch_size, -1)

        num_batches = batched_X_train.shape[-1]
        for i in range(num_batches):
            # Get current batch of training data
            cur_X, cur_Y = batched_X_train[:,i], batched_Y_train[:,i]
            # Bind the tf_X and tf_Y here, using feed_dict
            sess.run(tf_optimizer, feed_dict={tf_X: cur_X, tf_Y: cur_Y})

        if (epoch + 1) % display_step == 0:
            cur_c = sess.run(tf_cost, feed_dict={tf_X: X_train, tf_Y: Y_train})
            cur_W = sess.run(tf_W)
            cur_b = sess.run(tf_b)
            print('Epoch {:04d}: cost={:.9f}, W={:.9f}, b={:.9f}'.format(epoch, cur_c, cur_W, cur_b))

    print ("Optimization Finished!")
    # Calculate the final W, b
    fin_w = sess.run(tf_W)
    fin_b = sess.run(tf_b)
    print('final W={:.5f}, b={:.5f}'.format(fin_w, fin_b))
    print('Real  W={}, b={}'.format(W, b))

    # Test on training dataset and testing dataset
    fin_train_c = sess.run(tf_cost, feed_dict={tf_X: X_train, tf_Y: Y_train})
    fin_test_c = sess.run(tf_cost, feed_dict={tf_X: X_test, tf_Y: Y_test})
    real_train_c = np.mean(np.square(X_train*W + b - Y_train))
    real_test_c = np.mean(np.square(X_test*W + b - Y_test))
    print('Training Dataset Cost using real W,b: {:.9f}'.format(real_train_c))
    print('Testing  Dataset Cost using real W,b: {:.9f}'.format(real_test_c))
    print('Training Dataset Cost using pred W,b: {:.9f}'.format(fin_train_c))
    print('Testing  Dataset Cost using pred W,b: {:.9f}'.format(fin_test_c))

    # Graphic display
    figure = plt.figure()
    figure.add_subplot(1, 2, 1)
    plt.plot(X_train, Y_train, 'ro', label='Training data')
    plt.plot(X_train, fin_w * X_train + fin_b, label='Fitted line')
    plt.legend()

    figure.add_subplot(1, 2, 2)
    plt.plot(X_test, Y_test, 'ro', label='Testing data')
    plt.plot(X_test, fin_w * X_test + fin_b, label='Fitted line')
    plt.legend()

    plt.show()

