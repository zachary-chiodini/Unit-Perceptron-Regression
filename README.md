<h1>The Perceptron Unit for Multiple Linear Regression Analysis</h1>

<p align="jutify">
    This is a perceptron unit with training algorithm for mutiple linear regression analysis of data written in Python.
    Click <a href="https://github.com/zachary-chiodini/unit-perceptron-classification">here</a> to view how the perceptron unit 
    can be used to classify linearly separable data.
</p>

<h1>Mathematical Model</h1>
<p align="jutify">
    This is a perceptron unit with training algorithm for mutiple linear regression analysis of data written in Python
</p>

<p align="center">
    <img src="photos/equation1.png">
</p>

<p align="center">
    <img src="photos/equation2.png">
</p>

<p align="center">
    <img src="photos/equation3.png">
</p>

<p align="center">
    <img src="photos/equation4.png">
</p>

<p align="center">
    <img src="photos/equation5.png">
</p>

<p align="center">
    <img src="photos/paraboloid.png">
</p>

<p align="center">
    <img src="photos/equation6.png">
</p>

<p align="center">
    <img src="photos/equation7.png">
</p>

<p align="center">
    <img src="photos/equation8.png">
</p>

<p align="center">
    <img src="photos/equation9.png">
</p>

<p align="center">
    <img src="photos/equation10.png">
</p>

<p align="center">
    <img src="photos/equation11.png">
</p>

<p align="center">
    <img src="photos/equation12.png">
</p>

<p align="center">
    <img src="photos/equation13.png">
</p>

<p align="center">
    <img src="photos/equation14.png">
</p>

<p align="center">
    <img src="photos/equation15.png">
</p>

<p align="center">
    <img src="photos/equation16.png">
</p>

<p align="center">
    <img src="photos/equation17.png">
</p>

<p align="center">
    <img src="photos/equation18.png">
</p>

```python
import pandas as pd, numpy as np
from perceptron import Network
```

```python
df = pd.read_csv( 'air_int_incap.csv' )
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>id</th>
      <th>time-to-incapacitation</th>
      <th>1000/tti</th>
      <th>CO</th>
      <th>HCN</th>
      <th>H2S</th>
      <th>HCl</th>
      <th>HBr</th>
      <th>NO2</th>
      <th>SO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20.0</td>
      <td>2.36</td>
      <td>423.7</td>
      <td>164</td>
      <td>6.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.26</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>14.0</td>
      <td>2.38</td>
      <td>420.2</td>
      <td>174</td>
      <td>7.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.07</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1.0</td>
      <td>2.61</td>
      <td>383.1</td>
      <td>96</td>
      <td>4.7</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>5.0</td>
      <td>0.08</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2.0</td>
      <td>3.07</td>
      <td>325.7</td>
      <td>101</td>
      <td>7.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.1</td>
      <td>0.43</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>61.0</td>
      <td>3.07</td>
      <td>325.7</td>
      <td>142</td>
      <td>6.8</td>
      <td>0.0</td>
      <td>27.6</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
df[ 'CO' ] = df[ 'CO' ].apply( lambda x : np.log( x ) if x else np.NaN )
df[ 'NO2' ] = df[ 'NO2' ].apply( lambda x : np.log( x ) if x else np.NaN )
```


```python
df = df[ [ 'time-to-incapacitation', 'CO', 'NO2' ] ].dropna( axis = 0 )
```


```python
X = df[ [ 'CO', 'NO2' ] ].to_numpy()
Y = df[ [ 'time-to-incapacitation' ] ].to_numpy()
```


```python
network = Network()
network.train( X, Y, r = 0.01, h = 0.0000000001 )
```


```python
network.showModel()
```
  
    +-------+---------------------------------------------------+
    | Model | f( x1, x2 ) = 17.0(3.0) - 2.8(0.5)x1 - 0.7(0.2)x2 |
    +-------+---------------------------------------------------+
    |   b   | 17.072911943590032                                |
    +-------+---------------------------------------------------+
    |  w1   | -2.837442793250083                                |
    +-------+---------------------------------------------------+
    |  w2   | -0.7476215572004509                               |
    +-------+---------------------------------------------------+
    | STDb  | 2.5440732699408017                                |
    +-------+---------------------------------------------------+
    | STDw1 | 0.5025604849784758                                |
    +-------+---------------------------------------------------+
    | STDw2 | 0.18505563654052753                               |
    +-------+---------------------------------------------------+
    |  MSE  | 4.160156822943904                                 |
    +-------+---------------------------------------------------+
    |  R^2  | 0.6094678120459519                                |
    +-------+---------------------------------------------------+

<p align="center">
    <img src="photos/incapacitation.png">
</p>
