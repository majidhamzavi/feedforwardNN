# feedforwardNN
Feed Forward Neural Network - Simply Explained!

In this repsitory, I will explain how to build a simple forward neural network with 2 hidden layes and given weights and biases. Here, I am using well-known MNIST dataset. 10,000 images have been reshaped from 28 * 28 to 784 * 1, i.e. the x dataset now is 10000 * 784. the Y datase is then 1000 * 1 that consists of the label of each image. Each layer has 512 neurons. By simple matrix calculations, it is easy to underatand for the first layer, weight has dimensions as 784 * 512 and bias has 512 * 1; i.e. h0 = x[1000 * 784] .dot w0[784 * 512] + b0[512 * 1].
