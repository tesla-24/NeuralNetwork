#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt

#loading data
filename_data = open('mnist_data.pkl','rb')
data = pickle.load(filename_data)

trainLabels = np.array(data['trainLabels'])
testLabels = np.array(data['testLabels'])

testData = np.array(data['testImages']).T
trainData =np.array(data['trainImages']).T


def sigmoid(x):
    '''
    Paramaters
    ------------------
    x : number or numpy array of numbers
    
    Output
    -------------------
    Returns sigmoid function value of a given number or array of numbers
    '''
    return 1/(1 + np.exp(-np.array(x)))

def sigmoid_derivative(x):
    '''
    Paramaters
    ------------------
    x : number or numpy array of numbers
    
    Output
    -------------------
    Returns the derivative of Sigmoid at point x   
    '''
    return sigmoid(x) * (1-sigmoid(x))

def softmax(x):
    '''
    Paramaters
    ------------------
    x : numpy array of numbers
    
    Output
    -------------------
    Returns array of probabilities obtained by applying softmax function 
    '''
    return np.exp(x)/sum(np.exp(x))

def encode_to_onehot(x):
    '''This function encodes an integer label to one hot vector
    Paramaters
    ------------------
    x : list of integers (Range : 0-9)
    
    Output
    -------------------
    Returns the one hot encoded vector of length 10 corresponding to each integer in the given array
    '''
    endoded_array = np.zeros((len(x), 10))
    endoded_array[np.arange(len(endoded_array)), x] = 1
    return endoded_array.T

def onehot_to_num(x):
    ''' This function converts onehot vector to corresponding integer label
    Paramaters
    ------------------
    x : Array of one-hot encoded vectors
    
    Output
    -------------------
    Returns the integer value corresponding to each vector in the given array
    '''
    return np.argmax(x,0)
    

def initialize_parameters(no_of_hidden_neurons):
    ''' This function initializes parameters for the MLP
    Paramaters
    ------------------
    no_of_hidden_neurons : integer denoting total no of neurons in the middle or hidden layer of 3 layer MLP
    
    Output
    -------------------
    Returns W1, B1, W2, B2
            W1 is weight matrix and B1 is bias vector connecting input layer to hidden layer
            W2 is weight matrix and B2 is bias vector connecting hidden layer to output layer
    '''
    W1 = np.random.rand(no_of_hidden_neurons,784) - 0.5
    B1 = np.zeros((no_of_hidden_neurons,1))
    W2 = np.random.rand(10,no_of_hidden_neurons) - 0.5
    B2 = np.zeros((10,1))
    return W1, B1, W2, B2

def forward(x, W1, B1, W2, B2):
    ''' This function is used for forward propogation of MLP
    
    Paramaters
    ------------------
    x : vector of length 784 representing the 28*28 image
    W1, B1 : weight matrix and bias vector connecting input layer to hidden layer
    W2, B2 : weight matrix and bias vector connecting hidden layer to output layer
    
    Output
    -------------------
    Returns Y1, layer1_out, Y2, layer2_out 
            Y1 : values of the hidden layer neurons before applying activation function
            layer1_out : values of hidden layer after applying activation function
            Y2 : values of the output layer neurons before applying activation function
            layer2_out : values of output layer after applying activation function
    
    '''
    Y1 = W1.dot(x) + B1
    layer1_out = sigmoid(Y1)
    Y2 = W2.dot(layer1_out) + B2
    layer2_out = softmax(Y2)
    return Y1, layer1_out, Y2, layer2_out

def backPropogate(data, labels, learning_rate, W1, B1, W2, B2, Y1, layer1_out, Y2, layer2_out ):
    ''' This function is used to backpropogate the error and update weights
    
    Paramaters
    ------------------
    data : 784 * n matrix representing n images
    labels : vector of length n representing the target labels
    learning_rate : learning rate to be used to update weights
    W1, B1 : weight matrix and bias vector connecting input layer to hidden layer
    W2, B2 : weight matrix and bias vector connecting hidden layer to output layer
    Y1 : values of the hidden layer neurons before applying activation function
    layer1_out : values of hidden layer after applying activation function
    Y2 : values of the output layer neurons before applying activation function
    layer2_out : values of output layer after applying activation function
    
    Output
    -------------------
    Returns the updated W1, B1, W2, B2
    '''
    l = len(data[0])
    labels_onehot = encode_to_onehot(labels)
    der_Y2 = layer2_out - labels_onehot
    der_W2 = 1/l * der_Y2.dot(layer1_out.T)
    der_B2 = 1/l * np.sum(der_Y2)
    der_Y1 = W2.T.dot(der_Y2) * sigmoid_derivative(Y1)
    der_W1 = 1/l * der_Y1.dot(data.T)
    der_B1 = 1/l * np.sum(der_Y1)
    
    W1 = W1 - learning_rate * der_W1
    B1 = B1 - learning_rate * der_B1
    W2 = W2 - learning_rate * der_W2
    B2 = B2 - learning_rate * der_B2
    
    return W1, B1, W2, B2



def calculate_accuracy(predicted, original):
    ''' This function calculates the accuracy of predicted values
    Paramaters
    ------------------
    predicted : Array of integers representing predicted classes by MLP
    original : Array of integers corresponding to true class
    
    Output
    -------------------
    Returns accuracy value
    '''
    return np.sum(predicted==original)/len(predicted)

def calculate_loss(onehot_predicted, onehot_actual):
    ''' This function calculates the  total loss of MLP network for given dataset
    Paramaters
    ------------------
    onehot_predicted : Array of onehot vectors predicted by MLP
    onehot_actual : Array of true onehot vectors
    
    Output
    -------------------
    Returns the loss value
    '''
    logs_matrix = np.log10(onehot_predicted.T+0.001)
    return np.trace(-logs_matrix.dot(onehot_actual))
    

def train(X, Y, learning_rate, no_of_steps, no_of_hidden_neurons):
    ''' This function trains the MLP network
    Paramaters
    ------------------
    X : 784 * n matrix representing tataset with n images
    Y : vector of size n representing true labels
    learning_rate : learning rate to be used to update weights
    no_of_steps : Total steps to perform in the iteration
    no_of_hidden_neurons : Total number of neurons to have in middle layer
    
    Output
    -------------------
    Returns the final W1, B1, W2, B2 representing the trained network
    '''
    W1, B1, W2, B2 = initialize_parameters(no_of_hidden_neurons)
    Iterations_array = []
    Accuracy_array = []
    Loss_array = []
    accuracy =0
    loss = 0
    for i in range(1,no_of_steps+1):
        Y1, layer1_out, Y2, layer2_out = forward(X, W1, B1, W2, B2)
        W1, B1, W2, B2 = backPropogate(X, Y, learning_rate, W1, B1, W2, B2, Y1, layer1_out, Y2, layer2_out )
        if(i%10==0):            
            predictions = onehot_to_num(layer2_out)
            accuracy = calculate_accuracy(predictions, Y)
            loss = calculate_loss(layer2_out, encode_to_onehot(Y))
            print('Iteration: '+str(i)+' | Accuracy: '+str(accuracy) + ' | Loss: '+str(loss))
            Iterations_array.append(i)
            Accuracy_array.append(accuracy)
            Loss_array.append(loss)
    print('Training Accuracy: '+str(accuracy)+ ' | Training Loss: '+str(loss))
    plt.plot(Iterations_array, Accuracy_array)
    plt.title('Accuracy vs Iterations for Training Data')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('Accuracy_plot')
    #plt.show()
    plt.close()
    plt.plot(Iterations_array, Loss_array)
    plt.title('Loss vs Iterations for Training Data')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('Loss_plot')
    #plt.show()
    plt.close()
    return W1, B1, W2, B2

def test(test_X, test_y, W1, B1, W2, B2):
    ''' This function prints the Test Accuracy and Loss
    Paramaters
    ------------------
    test_X : 784 * n matrix representing tataset with n images
    test_Y : vector of size n representing true labels
    W1, B1, W2, B2 : The trained weights and biases  

    '''
    Y1, layer1_out, Y2, layer2_out = forward(test_X, W1, B1, W2, B2)
    predictions = onehot_to_num(layer2_out)
    accuracy = calculate_accuracy(predictions, test_y)
    loss = calculate_loss(layer2_out, encode_to_onehot(test_y))
    print('Test Accuracy: '+str(accuracy)+ ' | Test Loss: '+str(loss))
    

W1, B1, W2, B2 = train(trainData, trainLabels, 1, 250, 200)
test(testData, testLabels, W1, B1, W2, B2)

print('Plots are saved in the current folder')