import numpy as np

class Softmax():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.05
        self.epochs = 50
        self.reg_const = 0
    
    def calc_gradient(self, X_train, y_train):
        """
        Calculate gradient of the softmax loss
          
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns:
        - gradient with respect to weights W; an array of same shape as W
        """

        #Define Number of classes, C

        C = 10

        #Initialize the gradients matrix, grad_w

        grad_w = np.zeros((10,3072))

        #Iterate over minibatch of data

        for i in range(len(y_train)):
          
          #Calculate Log K
          Log_K=-max(np.dot(self.w,X_train[i]))

          #Loop over classes to calculate the gradient

          for j in range(C):
            if(j == y_train[i]):
              grad_w[j] = grad_w[j] - X_train[i]+np.exp(np.dot(self.w[j],X_train[i])+Log_K)*X_train[i]/(sum(np.exp(np.dot(self.w,X_train[i]))+Log_K))
            else:
              grad_w[j] = grad_w[j] + np.exp(np.dot(self.w[j],X_train[i])+Log_K)*X_train[i]/(sum(np.exp(np.dot(self.w,X_train[i]))+Log_K))
        
        return grad_w

    def train(self, X_train, y_train):
        """
        Train Softmax classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
        """

        #Initialize the weights matrix, w

        self.w = np.random.rand(10,np.shape(np.reshape(X_train[0],-1))[0])

        #Define the size of minibatches, N

        N = 100

        for num_epochs in range(self.epochs):

          print("epochs number =", num_epochs)

          #Create minibatches: first rows 0-10, then 11-20,..., 48990-49000

          for i in range((len(X_train)//N)-N):

            X_train_mb = X_train[N*i:N*i+N]
            y_train_mb = y_train[N*i:N*i+N]

            #Call calc_gradient to compute gradient for the minibatch
            grad_w = self.calc_gradient(X_train_mb,y_train_mb)

            #Update the weights considering the learning rate and dividing by N to average the gradients

            self.w = self.w - self.alpha*grad_w/N

    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        pred_softmax=np.zeros(len(X_test))
        for i in range(len(X_test)):
          pred_softmax[i]=np.argmax(np.dot(self.w,X_test[i]))
        return pred_softmax 