# This is a simple mini framework for fully connected neural nets
# tested on MNIST dataset
# created by Osama Khafagy on feb 10, 2020
import numpy as np
import matplotlib.pyplot as plt
# setting hyperparameters
Epochs = 10
batch_size = 100
display_freq = 1000
LR = 0.1
img_h = img_w = 28
img_size_flat = img_h * img_w
n_classes = 10

def weight_variable(shape):
    print(shape)
    w = np.random.randn(shape[0], shape[1]) * 0.1
    return w

def bias_variable(shape):
    b = np.zeros(shape)
    return b

def f_layer(a_l_p, w, b, g=None):
    z = np.dot(w, a_l_p) + b
    if g is not None:
        return g(z)
    else:
        return z

def b_layer(da_l, w_l, g_l_prime, a_l_p, first_layer=False):
    dz_l = da_l * g_l_prime
    da_l_p = 0
    if not first_layer:
        da_l_p = np.dot(w_l.T, dz_l)
    dw_l = (1/batch_size) * np.dot(dz_l, a_l_p.T)
    db_l = (1/batch_size) * np.sum(dz_l, axis=1, keepdims=True)
    return da_l_p, dw_l, db_l

def sigmoid_act(z):
    g = 1 / (1+np.exp(-z))
    g_prime = g * (1-g)
    return g, g_prime

def tanh_act(z):
    g = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    g_prime = 1 - g*g
    return g, g_prime

def relu_act(z):
    g = z * (z > 0)
    g_prime = 1.*(z > 0)
    return g, g_prime

def softmax_with_cross_entropy(scores, labels, test_phase=False):
    # labels and scores should have shape=(10, m)
    exp_score = np.exp(scores)          # (10, m)
    sum_exp_scores = np.sum(exp_score, axis=0, keepdims=True)       #(1, m)
    softmax_out = exp_score / sum_exp_scores        # (10, m)
    pos_index = np.argmax(labels, axis=0)        # (1, m)
    cost = np.max(softmax_out, axis=0, keepdims=True)          # (1, m)
    cost = -np.log(cost)
    if not test_phase:
        cost = (1/batch_size) * np.sum(cost, axis=1, keepdims=True)         # (1, 1)
        logits_loss = np.copy(softmax_out)
        batch_range = list(range(batch_size))
        logits_loss[pos_index, batch_range] = logits_loss[pos_index, batch_range] - 1
        return logits_loss, cost, softmax_out
    else:
        return cost, softmax_out
    # logits loss should have shape=(10,m) which is exactly what a binary cross entropy loss
    # would produce but with respect to it's only output neuron (1, m)
    # the effect of error produced by all output units is added to previous layer

def forward_prop(weights, biases, inputs, layers):
    outputs = []
    for layer in range(layers):
        outputs = f_layer(inputs[layer], weights[layer], biases[layer], g=relu_act)
    return outputs

# helper functions to load MNIST data
def load_data(mode='train'):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    if mode == 'train':
        x_train, y_train = mnist.train.images, mnist.train.labels
        x_valid, y_valid = mnist.validation.images, mnist.validation.labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
        return x_test, y_test

def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[:, start:end]
    y_batch = y[:, start:end]
    return x_batch, y_batch

# helper functions to inspect learning process
def draw_first_sample(x):
    single_sample = np.reshape(x[:, 0], (28, 28))
    plt.figure()
    plt.imshow(single_sample, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def draw_sample(sample_image):
    single_sample = np.reshape(sample_image, (28, 28))
    plt.figure()
    plt.imshow(single_sample, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()
''' Preparing dataset'''
x_train, y_train, x_valid, y_valid = load_data(mode='train')
x_train = x_train.T
y_train = y_train.T
x_valid = x_valid.T
y_valid = y_valid.T
print('size of:')
print('- Training set:\t\t{}'.format(len(y_train)))
print('- Validation set:\t\t{}'.format(len(y_valid)))
print('x_train:\t{}'.format(x_train.shape))
print('y_train:\t{}'.format(y_train.shape))
print('x_valid:\t{}'.format(x_valid.shape))
print('y_valid:\t{}'.format(y_valid.shape))


#-------------------#
# buidling the network
# 784 input (flattened image)
# layer1 has 100 hidden units, relu activation
# layer2 has 10 hidden units, relu activation
# layer3 has 10 hidden unit, softmax activation
#-----------------------#
''' creating the network '''
# layer1
W1 = weight_variable((100, 784))
b1 = bias_variable((100, 1))
print(W1.shape)
# layer 2
W2 = weight_variable((10, 100))
b2 = bias_variable((10, 1))

num_tr_iter = int(55000 / batch_size)

#------------------#
'''Running Training'''
history = []
for epoch in range(Epochs):
    for iteration in range(num_tr_iter):
        start = iteration * batch_size
        end = (iteration+1) * batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
        '''Forward pass'''
        a1, g1_ = f_layer(x_batch, W1, b1,  g=relu_act)
        a2, g2_ = f_layer(a1, W2, b2,  g=relu_act)
        da2, cost, _ = softmax_with_cross_entropy(a2, y_batch)
        history.append([epoch+1, cost])
        if iteration % display_freq == 0:
            print('Epoch {}: '.format(epoch+1))
            print('cost = ', np.squeeze(cost))
        #-----------------#
        '''Backward pass'''
        da1, dw2, db2 = b_layer(da2, W2, g2_, a1)
        #print('da[1]=\n{},\n dw[2]=\n{},\n db[3]=\n{}'.format(da2, dw3, db3))
        _, dw1, db1 = b_layer(da1, W1, g1_, x_batch, first_layer=True)
        #print('da[1]=\n{},\n dw[2]=\n{},\n db[2]=\n{}'.format(da1, dw2, db2))
        #----------------#
        '''Updating parameters'''
        W1 = W1 - LR*dw1
        W2 = W2 - LR*dw2
        b1 = b1 - LR*db1
        b2 = b2 - LR*db2
#---------------#
# plot history

history = np.array(history)
plt.plot(history[:, 0], history[:, 1], 'ro-')
plt.xlabel('Epoch')
plt.ylabel('Cost J')
plt.show()

# show and save model
np.savez('W1.npz', W1, b1)
np.savez('W2.npz', W2, b2)

# retrieving model
layer1_params = np.load('W1.npz')
layer2_params = np.load('W2.npz')
W1 = layer1_params['arr_0']
b1 = layer1_params['arr_1']
W2 = layer2_params['arr_0']
b2 = layer2_params['arr_1']
# load test data
x_test, y_test = load_data(mode='test')
x_test = x_test.T
y_test = y_test.T

print(x_test.shape)
print(y_test.shape)
# run prediction over 3 different samples
# first sample
'''Forward pass'''
X_test = x_test[:, 0]
Y_test = y_test[:, 0]
draw_sample(X_test)
a1, _ = f_layer(X_test, W1, b1,  g=relu_act)
print(a1.shape)
a2, _ = f_layer(a1, W2, b2,  g=relu_act)
print(a2.shape)
loss, a3 = softmax_with_cross_entropy(X_test, Y_test, test_phase=True)
print("Predicted handwritten digit:", np.argmax(Y_test))
# second sample
X_test = x_test[:, 17]
Y_test = y_test[:, 17]
draw_sample(X_test)
a1, _ = f_layer(X_test, W1, b1,  g=relu_act)
print(a1.shape)
a2, _ = f_layer(a1, W2, b2,  g=relu_act)
print(a2.shape)
loss, a3 = softmax_with_cross_entropy(X_test, Y_test, test_phase=True)
print("Predicted handwritten digit:", np.argmax(Y_test))
# third sample
X_test = x_test[:, 33]
Y_test = y_test[:, 33]
draw_sample(X_test)
a1, _ = f_layer(X_test, W1, b1,  g=relu_act)
print(a1.shape)
a2, _ = f_layer(a1, W2, b2,  g=relu_act)
print(a2.shape)
loss, a3 = softmax_with_cross_entropy(X_test, Y_test, test_phase=True)
print("Predicted handwritten digit:", np.argmax(Y_test))
