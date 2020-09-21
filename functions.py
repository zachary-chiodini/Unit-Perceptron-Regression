################
# functions.py #
################

# this code is used by the network class in perceptron.py
def prettyPrint( mse, rsquare, network, netvars ) -> None :
    '''
    Print Out Perceptron Model and Statistics Neatly
    ------------------------------------------------
    Arguments are Private Attributes of Network Class
    '''
    model = 'f('
    for i in range( 1, len( network ) ) :
        model += ' x{},'.format( i )
    model = model[ : -1 ]
    dgt = firstSigFig( netvars[ 0 ]**0.5 )
    model += ' ) = {}({})'.format(
        round( network[ 0 ][ 0 ], dgt ),
        round( netvars[ 0 ]**0.5, dgt )
        )
    for i in range( len( network ) - 1 ) :
        slope = network[ i + 1 ]
        error = netvars[ i + 1 ]**0.5
        dgt = firstSigFig( error )
        model += ' {} {}({})x{}'.format(
            '-' if slope[ 0 ] < 0 else '+',
            abs( round( slope[ 0 ], dgt ) ),
            round( error, dgt ),
            i + 1
            )
    modellst = []
    if len( model ) > 70 :
        i = 0
        slce = model[ i*70 : ( i + 1 )*70 ]
        while slce :
            i += 1
            modellst.append( slce )
            slce = model[ i*70 : ( i + 1 )*70 ]
        modellen = 70
    else :
        modellst = [ model ]
        modellen = len( model )
    print()
    print( '+-------+--' + '-'*modellen + '+' )
    print( '| Model | {} |'.format( modellst[ 0 ] ) )
    for piece in modellst[ 1 : ] :
        print( '|       | {}'.format( piece ) +\
               ' '*( modellen - len( piece ) ) + ' |' )
    print( '+-------+--' + '-'*modellen + '+' )
    print( '|   b   | {}'.format( network[ 0 ][ 0 ] ) +\
           ' '*( modellen -\
                 len( str( network[ 0 ][ 0 ] ) ) ) +\
           ' |' )
    for i in range( 1, len( network ) ) :
        print( '+-------+--' + '-'*modellen + '+' )
        print( '|  w{}   | {}'.format( i + 1, network[ i ][ 0 ] ) +\
               ' '*( modellen -\
                     len( str( network[ i ][ 0 ] ) ) ) +\
               ' |' )
    print( '+-------+--' + '-'*modellen + '+' )
    print( '| STDb  | {}'.format( netvars[ 0 ]**0.5 ) +\
           ' '*( modellen -\
                 len( str( netvars[ 0 ]**0.5 ) ) ) +\
           ' |' )
    for i in range( 1, len( netvars ) ) :
        print( '+-------+--' + '-'*modellen + '+' )
        print( '| STDw{} | {}'.format( i, netvars[ i ]**0.5 ) +\
           ' '*( modellen -\
                 len( str( netvars[ i ]**0.5 ) ) ) +\
           ' |' )
    print( '+-------+--' + '-'*modellen + '+' )
    print( '|  MSE  | {}'.format( mse ) +\
           ' '*( modellen -\
                 len( str( mse ) ) ) +\
           ' |' )
    print( '+-------+--' + '-'*modellen + '+' )
    print( '|  R^2  | {}'.format( rsquare ) +\
           ' '*( modellen -\
                 len( str( rsquare ) ) ) +\
           ' |' )
    print( '+-------+--' + '-'*modellen + '+' )
    print()
    return

def firstSigFig( flt : float ) -> int :
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
