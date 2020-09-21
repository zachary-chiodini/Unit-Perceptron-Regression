import numpy as np
from nptyping import NDArray
from functions import prettyPrint

class Network( object ) :

    def __init__( self : object ) -> None :
        self.__mse = 0.0 # mean sqaured error (mse)
        self.__rsquare = 0.0
        self.__netvars = np.array([]) # variances
        self.__network = np.array([])
        self.__results = np.array([])

    def perceptron(
        self : object,
        X : NDArray[ float ], # input matrix
        W : NDArray[ float ], # network weights,
        b : float = 0         # bias, if not included in weights
        ) -> float :
        '''
        Perceptron Unit without Activation Function
        '''
        return np.matmul( X, W ) + b

    def train(
        self : object,
        X : NDArray[ float ], # input matrix
        Y : NDArray[ float ], # target vector
        r : float = 0.01,     # learning rate
        h : float = 0.01      # convergence
        ) -> None :
        '''
        Gradient Descent Training Algorithm
        '''
        m, n = len( X ), len( X[ 0 ] )
        # creating column of ones
        X = np.hstack(
            [ np.ones( ( m, 1 ) ), X ]
            )
        # generate random weights
        W = np.random.uniform(
            -0.5, 0.5, n + 1
            ).reshape( -1, 1 )
        # training algorithm
        P = self.perceptron( X, W )
        dSdW = 2*np.matmul( X.T, P - Y )
        ddSddW = 2*np.diag(
            np.matmul( X.T, X )
            ).reshape( -1, 1 )
        t = 0
        while abs( dSdW ).sum() > h :
            t = 0.9*t + r*dSdW / ddSddW
            W = W - t
            P = self.perceptron( X, W - 0.9*t )
            dSdW = 2*np.matmul( X.T, P - Y )
        self.__network = W
        # model statistics
        SSE = np.square( P - Y ).sum()
        TSS = np.square( Y - Y.mean() ).sum()
        self.__mse = SSE / ( m - n - 1 )
        self.__rsquare = 1 - SSE / TSS
        self.__netvars = np.diag(
            self.__mse * np.linalg.inv(
                np.matmul( X.T, X )
                )
            )
        return

    def test( self : object,
              X : NDArray[ float ] ) -> None :
        '''
        Test Perceptron on Test Data After Training
        '''
        assert self.__network.size > 0, \
               'Network must be trained.'
        self.__results = self.perceptron(
            X, self.__network[ 1 : ],  b = self.__network[ 0 ]
            )
        return

    def getResults( self : object ) -> NDArray[ float ] :
        return self.__results

    def showModel( self : object ) -> None :
        '''
        Prints Model and Statistics to Screen
        '''
        assert self.__network.size > 0, \
               'Network must be trained.'
        prettyPrint(
            self.__mse,
            self.__rsquare,
            self.__network,
            self.__netvars
            )
        return
