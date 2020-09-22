import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np, pandas as pd

df = pd.read_csv( 'air_int_incap.csv' )
df[ 'CO' ] = df[ 'CO' ].apply(
    lambda x : np.log( x ) if x else np.NaN
    )
df[ 'NO2' ] = df[ 'NO2' ].apply(
    lambda x : np.log( x ) if x else np.NaN
    )
df = df[
    [ 'time-to-incapacitation', 'CO', 'NO2' ]
    ].dropna( axis = 0 )

def model( x1, x2 ) :
    return 17 - 2.8*x1 - 0.7*x2

X = np.linspace(
    df[ 'CO' ].min(),
    df[ 'CO' ].max() - 0.02,
    10
    )
Y = np.linspace(
    df[ 'NO2' ].min(),
    df[ 'NO2' ].max() - 0.02,
    10
    )
X, Y = np.meshgrid( X, Y )
Z = model( X, Y )

rcParams[ 'font.family' ] = 'cambria'
rcParams.update( { 'font.size': 18 } )

fig = plt.figure(
    figsize = ( 8 , 4.94 ) # golden ratio
    )
ax = Axes3D( fig )

# model surface
ax.plot_surface(
    X, Y, Z,
    cmap = 'rainbow_r',
    alpha = 0.5
    )
# model wireframe
ax.plot_wireframe(
    X, Y, Z,
    color = 'black',
    linewidth = 1,
    alpha = 0.8
    )
# model contour
ax.contourf(
    X, Y, Z,
    cmap = 'rainbow_r',
    offset = 0,
    alpha = 0.5
    )

X = df[ [ 'CO', 'NO2' ] ].to_numpy()
Y = df[ [ 'time-to-incapacitation' ] ].to_numpy()

# residuals
for x1, x2, z1 in zip( X[ :, 0 ], X[ :, 1 ], Y[ :, 0 ] ) :
    z2 = model( x1, x2 )
    ax.plot3D(
        [ x1, x1 ],
        [ x2, x2 ],
        [ z1, z2 ],
        c = 'red',
        solid_capstyle = 'round',
        linewidth = 5
        )
# measured data
ax.scatter3D(
    X[ :, 0 ],
    X[ :, 1 ],
    Y[ :, 0 ],
    c = Y[ :, 0 ],
    s = 200,
    cmap = 'rainbow_r',
    linewidths = 1,
    edgecolors = 'black',
    alpha = 0.8,
    )
# plot design
ax.set_zlabel( '$t(min)$', labelpad = 10 )
ax.set_xlabel( '$ln([CO])$', labelpad = 10 )
ax.set_ylabel( '$ln([NO_2])$', labelpad = 10 )
ax.set_xlim(
    [ df[ 'CO' ].min().round(),
      df[ 'CO' ].max().round() ]
    )
ax.set_ylim(
    [ df[ 'NO2' ].min().round(),
      df[ 'NO2' ].max().round() ]
    )
ax.w_xaxis.set_pane_color( ( 0, 0, 0, 0 ) )
ax.w_yaxis.set_pane_color( ( 0, 0, 0, 0 ) )
ax.w_zaxis.set_pane_color( ( 0.95, 0.95, 0.95, 1 ) )
ax.grid( b = None )
ax.view_init( elev = 20, azim = -45 )
plt.savefig( 'incapacitation.png',
             transparent = True )
plt.show()
