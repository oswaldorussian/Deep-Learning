import numpy as np
import scipy

class Perceptron():
    def __init__(self):
        """
        Initialises Perceptron classifier with initializing 
        weights, alpha(learning rate) and number of epochs.
        """
        self.w = None
        self.alpha = 0.05  # fixme: try different settings for your learning rate
        self.epochs = 200  # fixme: try different training cycles
        
    def train(self, X_train, y_train):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture on 09/01.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        # Make sure X_train has the shape (num_train, D)
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        # Randomly initial weights
        self.w = np.random.rand(10,np.shape(np.reshape(X_train[0],-1))[0])
        # Let's start training
        for num_epochs in range(self.epochs):
          print("epochs number =", num_epochs)
          for i in range(len(y_train)):
              # fixme: how do you make the prediction given the data and weights?
              pred = np.argmax(np.dot(self.w,X_train[i]))
              if(pred!=y_train[i]):  # update weights when prediction is different from the truth
                  # fixme: update the weights for the predicted y
                  self.w[pred] = self.w[pred]-self.alpha/(0.1*(num_epochs+1))*X_train[i]
                  # fixme: update the weights for the true y
                  self.w[y_train[i]] = self.w[y_train[i]]+self.alpha/(0.1*(num_epochs+1))*X_train[i]

    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        # fixme: write the code for the prediction
        pred=np.zeros(len(X_test))
        for i in range(len(X_test)):
          pred[i]=np.argmax(np.dot(self.w,X_test[i]))
        return pred 