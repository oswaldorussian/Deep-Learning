import numpy as np


class NeuralNetwork:
    """
    A multi-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices.

    The network uses a nonlinearity after each fully connected layer except for the
    last. You will implement two different non-linearities and try them out: Relu
    and sigmoid.

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_sizes, output_size, num_layers, nonlinearity='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H_1)
        b1: First layer biases; has shape (H_1,)
        .
        .
        Wk: k-th layer weights; has shape (H_{k-1}, C)
        bk: k-th layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: List [H1,..., Hk] with the number of neurons Hi in the hidden layer i.
        - output_size: The number of classes C.
        - num_layers: Number of fully connected layers in the neural network.
        - nonlinearity: Either relu or sigmoid
        """
        self.num_layers = num_layers

        assert(len(hidden_sizes)==(self.num_layers-1))
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, self.num_layers + 1):
            self.params['W' + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params['b' + str(i)] = np.zeros(sizes[i])

        if nonlinearity == 'sigmoid':
            self.nonlinear = sigmoid
            self.nonlinear_grad = sigmoid_grad
        elif nonlinearity == 'relu':
            self.nonlinear = relu
            self.nonlinear_grad = relu_grad

    def forward(self, X):
        """
        Compute the scores for each class for all of the data samples.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.

        Returns:
        - scores: Matrix of shape (N, C) where scores[i, c] is the score for class
            c on input X[i] outputted from the last layer of your network.
        - layer_output: Dictionary containing output of each layer BEFORE
            nonlinear activation. You will need these outputs for the backprop
            algorithm. You should set layer_output[i] to be the output of layer i.

        """

        scores = None
        layer_output = {}
        #############################################################################
        # TODO: Write the forward pass, computing the class scores for the input.   #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C). Store the output of each layer BEFORE nonlinear activation  #
        # in the layer_output dictionary                                            #
        #############################################################################

        #Initialize the layer_output with the zeroth layer, X

        layer_output['h0']=X

        for i in range (1, self.num_layers+1):
          
          #Compute the output of the given layer prior to nonlinear activation

          if i == 1:

            layer_output['h' + str(i)] = np.dot(layer_output['h0'], self.params['W' + str(i)]) + self.params['b' + str(i)]

          else:

            layer_output['h' + str(i)] = np.dot(activated_h, self.params['W' + str(i)]) + self.params['b' + str(i)]

          #apply the nonlinearity to the layer_output

          activated_h = self.nonlinear(layer_output['h' + str(i)])

        #After we cycled through the layers, save the final activated layer_output as the class scores
        
        scores = layer_output['h' + str(self.num_layers)]

        return scores, layer_output

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """

        # Compute the forward pass

        # Store the result in the scores variable, which should be an array of shape (N, C).
        scores, layer_output = self.forward(X)

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss using the scores      #
        # output from the forward function. The loss include both the data loss and #
        # L2 regularization for weights W1,...,Wk. Store the result in the variable #
        # loss, which should be a scalar. Use the Softmax classifier loss.          #
        #############################################################################

        #
        
        #Preliminararies for loss calculation

        #Initialize data loss and regularization loss

        data_loss = 0

        reg_loss = 0

        ########################
        #       Data Loss      #
        ########################

        exp_scores = np.exp(scores)

        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[range(len(y)),y])

        data_loss = np.sum(correct_logprobs)/len(y)

        ########################
        #        Reg Loss      #
        ########################

        for i in range(1, self.num_layers+1):

          #Define weights matrix

          W = self.params['W' + str(i)]
          
          reg_loss = reg_loss + 0.5*reg*np.sum(W*W)

        ########################
        #       Total Loss     #
        ########################

        loss = data_loss + reg_loss
            
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        dscores = probs
        dscores[range(len(y)),y] = dscores[range(len(y)),y] - 1
        dscores = dscores / len(y)

        grads['W' + str(self.num_layers)] = np.dot(self.nonlinear(layer_output['h' + str(self.num_layers-1)]).T, dscores) + reg*self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = np.sum(dscores, axis = 0, keepdims = True)
        
        for i in reversed(range(1, self.num_layers)):

          if i == self.num_layers -1:

            dhidden = np.dot(dscores, self.params['W' + str(i+1)].T)

          else:

            dhidden = np.dot(dhidden, self.params['W' + str(i+1)].T)
          
          dhidden = dhidden*self.nonlinear_grad(layer_output['h' + str(i)])

          if i == 1:

            grads['W' + str(i)] = np.dot(layer_output['h'+str(i-1)].T, dhidden) + reg*self.params['W' + str(i)] 

          else:
          
            grads['W' + str(i)] = np.dot(self.nonlinear(layer_output['h'+str(i-1)]).T, dhidden) + reg*self.params['W' + str(i)] 

          grads['b' + str(i)] = np.sum(dhidden, axis = 0, keepdims = True)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False): #num_iters=100, batch_size = 200 originally
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################

            randomIndices = np.random.choice(num_train, batch_size)
            
            X_batch = X[randomIndices]
            y_batch = y[randomIndices]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            
            for i in range(1, self.num_layers+1):
              W = self.params['W' + str(i)]
              b = self.params['b' + str(i)]
              W = W - learning_rate*grads['W' + str(i)]
              b = b - learning_rate*grads['b' + str(i)]
              self.params['W' + str(i)] = W
              self.params['b' + str(i)] = b

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement classification prediction. You can use the forward      #
        # function you implemented                                                #
        ###########################################################################

        y_pred = np.zeros(len(X))
        
        scores, layer_output = self.forward(X)

        for i in range(len(X)):

          y_pred[i] = np.argmax(scores[i])

        return y_pred

def sigmoid(X):
    #############################################################################
    # TODO: Write the sigmoid function                                          #
    #############################################################################

    sig = 1/(1 + np.exp(-X))

    return sig

def sigmoid_grad(X):
    #############################################################################
    # TODO: Write the sigmoid gradient function                                 #
    #############################################################################

    sig = sigmoid(X)
    
    sig_grad = sig*(1 - sig)

    return sig_grad

def relu(X):
    #############################################################################
    #  TODO: Write the relu function                                            #
    #############################################################################

    rel = X * (X > 0)

    return rel

def relu_grad(X):
    #############################################################################
    # TODO: Write the relu gradient function                                    #
    #############################################################################
    
    rel_grad = 1. * (X > 0)

    return rel_grad
