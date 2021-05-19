import numpy as np

class Softmax():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.5
        self.epochs = 100
        self.reg_const = 0.05
    
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

        #Number of classes

        C = 10

        grad_w = np.zeros((10,3072))

        Log_K=-max(np.dot(self.w,X_train))

        for i in range(C):
          if(i == y_train):
            grad_w[i] = -X_train+np.dot(np.exp(np.dot(self.w[i],X_train)+Log_K),X_train)/(sum(np.exp(np.dot(self.w,X_train))+Log_K))
          else: 
            grad_w[i] = np.dot(np.exp(np.dot(self.w[i],X_train)+Log_K),X_train)/(sum(np.exp(np.dot(self.w,X_train))+Log_K))
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
        self.w = np.random.rand(10,np.shape(np.reshape(X_train[0],-1))[0])

        for num_epochs in range(self.epochs):

          #Create minibatches

            for i in range(len(y_train)):
              grad_w = self.calc_gradient(X_train[i],y_train[i])
              self.w = self.w - self.alpha/(num_epochs+1)*grad_w

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