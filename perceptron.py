import numpy as np
from nptyping import NDArray

class Network :

    def __init__( self : object ) -> None :
        self.__mse = 0.0 # mean sqaured error (mse)
        self.__bias = 0.0
        self.__rsquare = 0.0
        self.__netvars = np.array([]) # variances
        self.__network = np.array([])
        self.__results = np.array([])

    def perceptron(
        self : object,
        X : NDArray[ float ], # input matrix
        W : NDArray[ float ], # network weights
        b : float             # bias
        ) -> float :
        '''
        Perceptron Unit without Activation Function
        '''
        return np.matmul( X, W ) + b

    def train(
        self : object,
        X : NDArray[ float ], # input matrix
        Y : NDArray[ float ], # target vector
        b : float,            # bias
        r : float = 0.001,    # learning rate
        h : float = 0.001     # convergence
        ) -> None :
        '''
        Training Algorithm
        '''
        self.__bias = b
        m, n = len( X ), len( X[ 0 ] )
        # generate random weights
        W = np.random.uniform(
            -0.5, 0.5, n
            ).reshape( -1, 1 )
        # training algorithm
        P = self.perceptron( X, W, self.__bias )
        dSdW = 2*np.matmul( X.T, P - Y )
        dSdB = 2*( P - Y ).sum()
        while abs( dSdW.sum() + dSdB ) > h :
            W = W - r*dSdW
            self.__bias = self.__bias - r*dSdB
            P = self.perceptron( X, W, self.__bias )
            dSdW = 2*np.matmul( X.T, P - Y )
            dSdB = 2*( P - Y ).sum()
        self.__network = W
        # model statistics
        Xb = np.hstack(
            ( np.ones( ( m, 1 ) ), X )
            )
        SSE = np.square( P - Y ).sum()
        TSS = np.square( Y - Y.mean() ).sum()
        self.__mse = SSE / ( m - n - 1 )
        self.__rsquare = 1 - SSE / TSS
        self.__netvars = np.diag(
            self.__mse * np.linalg.inv(
                np.matmul( Xb.T, Xb )
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
            X, self.__network, self.__bias
            )
        return

    def getResults( self : object ) -> NDArray[ float ] :
        return self.__results

    def showModel( self : object ) -> None :
        '''
        Prints Model and Statistics to Screen
        '''
        print( self.__netvars**0.5 )
        model = 'f('
        for i in range( 1, len( self.__network ) + 1 ) :
            model += ' x{},'.format( i )
        model = model[ : -1 ]
        dgt = self.__firstSigFig( self.__netvars[ 0 ]**0.5 )
        model += ' ) = {}({})'.format(
            round( self.__bias, dgt ),
            round( self.__netvars[ 0 ]**0.5, dgt )
            )
        i = 1
        for slope, error in zip( self.__network, self.__netvars[ 1 : ] ) :
            error = error**0.5
            dgt = self.__firstSigFig( error )
            model += ' {} {}({})x{}'.format(
                '-' if slope[ 0 ] < 0 else '+',
                abs( round( slope[ 0 ], dgt ) ),
                round( error, dgt ),
                i
                )
            i += 1
        print()
        print( '+-------+--' + '-'*len( model ) + '+' )
        print( '| Model | {} |'.format( model ) )
        print( '+-------+--' + '-'*len( model ) + '+' )
        print( '|  MSE  | {}'.format( round( self.__mse, 2 ) ) +\
               ' '*( len( model ) -\
                     len( str( round( self.__mse, 2 ) ) ) ) +\
               ' |' )
        print( '+-------+--' + '-'*len( model ) + '+' )
        print( '|  R^2  | {}'.format( round( self.__rsquare, 2 ) ) +\
               ' '*( len( model ) -\
                     len( str( round( self.__rsquare, 2 ) ) ) ) +\
               ' |' )
        print( '+-------+--' + '-'*len( model ) + '+' )
        print()
        return

    def __firstSigFig( self: object, flt : float ) -> int :
        '''
        Returns the Decimal Place of
        the First Significant Figure
        '''
        lst = str( flt ).split( '.' )
        if len( lst ) == 1 or lst[ 0 ] != '0':
            return -len( lst[ 0 ] ) + 1
        n = 1
        for i in lst[ 1 ] :
            if i != '0' :
                break
            n += 1
        return n
