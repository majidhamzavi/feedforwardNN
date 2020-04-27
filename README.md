# feedforwardNN
Feed Forward Neural Network - Simply Explained!

In this repository, I am simply explaining how to build a forward neural network from scratch with 2 hidden layers while weights and biases are given. Here, I am using well-known MNIST dataset. 10,000 images have been reshaped from 28 * 28 pixels to 784 * 1, i.e. the x dataset now is 10,000 * 784. the Y dataset is then 10,000 * 1 that consists of the label of each image. Each layer has 512 neurons. By simple matrix calculations, it is easy to understand for the first layer, weight has dimensions as 784 * 512 and bias has 512 * 1; i.e. h0 = x[10,000 * 784] .dot w0[784 * 512] + b0[512 * 1]. ReLU activation function will be used here.

For the second hidden layer, weight would have dimensions as 512 * 512 and the same dimensions for biases. Again, we utilize the ReLU activation function. Therefore, h1 = h0[10,000 * 512] .dot w1[512 * 512] + b1[512 * 1].

In the last layer, based on the matrix algebra, wight is a matrix with dimensions 512 * 10 and for bias, it is a 10 * 1 matrix. So, Out = h1[10,000 * 512] .dot w2[512 * 10] + b2[10 * 1] and its dimensions would be 10,000 * 10 and getting its argmax for each row, it reduces to 10,000 * 1 should be compared to Y.
