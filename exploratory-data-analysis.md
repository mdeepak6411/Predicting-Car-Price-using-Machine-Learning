<h1>EDA</h1>


<h2 id="import_data">1. Import Data </h2>

 Import libraries 


```python
import pandas as pd
import numpy as np
```


```python
df= pd.read_csv(r'C:\Users\DEEPAK MISHRA\Desktop\data\automobileEDA.csv')
```


```python

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>...</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>city-L/100km</th>
      <th>horsepower-binned</th>
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>...</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
      <td>11.190476</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>0.811148</td>
      <td>...</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
      <td>11.190476</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>0.822681</td>
      <td>...</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
      <td>12.368421</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>0.848630</td>
      <td>...</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
      <td>9.791667</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>0.848630</td>
      <td>...</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
      <td>13.055556</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



<h2 id="pattern_visualization">2. Analyzing Individual Feature Patterns using Visualization</h2>


```python
%%capture
! pip install seaborn
```

 Import visualization packages "Matplotlib" and "Seaborn"


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
```


```python

print(df.dtypes)
```

    symboling              int64
    normalized-losses      int64
    make                  object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-wheels          object
    engine-location       object
    wheel-base           float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinders      object
    engine-size            int64
    fuel-system           object
    bore                 float64
    stroke               float64
    compression-ratio    float64
    horsepower           float64
    peak-rpm             float64
    city-mpg               int64
    highway-mpg            int64
    price                float64
    city-L/100km         float64
    horsepower-binned     object
    diesel                 int64
    gas                    int64
    dtype: object
    


```python
df.dtypes['peak-rpm']
```




    dtype('float64')




```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>city-L/100km</th>
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>symboling</td>
      <td>1.000000</td>
      <td>0.466264</td>
      <td>-0.535987</td>
      <td>-0.365404</td>
      <td>-0.242423</td>
      <td>-0.550160</td>
      <td>-0.233118</td>
      <td>-0.110581</td>
      <td>-0.140019</td>
      <td>-0.008245</td>
      <td>-0.182196</td>
      <td>0.075819</td>
      <td>0.279740</td>
      <td>-0.035527</td>
      <td>0.036233</td>
      <td>-0.082391</td>
      <td>0.066171</td>
      <td>-0.196735</td>
      <td>0.196735</td>
    </tr>
    <tr>
      <td>normalized-losses</td>
      <td>0.466264</td>
      <td>1.000000</td>
      <td>-0.056661</td>
      <td>0.019424</td>
      <td>0.086802</td>
      <td>-0.373737</td>
      <td>0.099404</td>
      <td>0.112360</td>
      <td>-0.029862</td>
      <td>0.055563</td>
      <td>-0.114713</td>
      <td>0.217299</td>
      <td>0.239543</td>
      <td>-0.225016</td>
      <td>-0.181877</td>
      <td>0.133999</td>
      <td>0.238567</td>
      <td>-0.101546</td>
      <td>0.101546</td>
    </tr>
    <tr>
      <td>wheel-base</td>
      <td>-0.535987</td>
      <td>-0.056661</td>
      <td>1.000000</td>
      <td>0.876024</td>
      <td>0.814507</td>
      <td>0.590742</td>
      <td>0.782097</td>
      <td>0.572027</td>
      <td>0.493244</td>
      <td>0.158502</td>
      <td>0.250313</td>
      <td>0.371147</td>
      <td>-0.360305</td>
      <td>-0.470606</td>
      <td>-0.543304</td>
      <td>0.584642</td>
      <td>0.476153</td>
      <td>0.307237</td>
      <td>-0.307237</td>
    </tr>
    <tr>
      <td>length</td>
      <td>-0.365404</td>
      <td>0.019424</td>
      <td>0.876024</td>
      <td>1.000000</td>
      <td>0.857170</td>
      <td>0.492063</td>
      <td>0.880665</td>
      <td>0.685025</td>
      <td>0.608971</td>
      <td>0.124139</td>
      <td>0.159733</td>
      <td>0.579821</td>
      <td>-0.285970</td>
      <td>-0.665192</td>
      <td>-0.698142</td>
      <td>0.690628</td>
      <td>0.657373</td>
      <td>0.211187</td>
      <td>-0.211187</td>
    </tr>
    <tr>
      <td>width</td>
      <td>-0.242423</td>
      <td>0.086802</td>
      <td>0.814507</td>
      <td>0.857170</td>
      <td>1.000000</td>
      <td>0.306002</td>
      <td>0.866201</td>
      <td>0.729436</td>
      <td>0.544885</td>
      <td>0.188829</td>
      <td>0.189867</td>
      <td>0.615077</td>
      <td>-0.245800</td>
      <td>-0.633531</td>
      <td>-0.680635</td>
      <td>0.751265</td>
      <td>0.673363</td>
      <td>0.244356</td>
      <td>-0.244356</td>
    </tr>
    <tr>
      <td>height</td>
      <td>-0.550160</td>
      <td>-0.373737</td>
      <td>0.590742</td>
      <td>0.492063</td>
      <td>0.306002</td>
      <td>1.000000</td>
      <td>0.307581</td>
      <td>0.074694</td>
      <td>0.180449</td>
      <td>-0.062704</td>
      <td>0.259737</td>
      <td>-0.087027</td>
      <td>-0.309974</td>
      <td>-0.049800</td>
      <td>-0.104812</td>
      <td>0.135486</td>
      <td>0.003811</td>
      <td>0.281578</td>
      <td>-0.281578</td>
    </tr>
    <tr>
      <td>curb-weight</td>
      <td>-0.233118</td>
      <td>0.099404</td>
      <td>0.782097</td>
      <td>0.880665</td>
      <td>0.866201</td>
      <td>0.307581</td>
      <td>1.000000</td>
      <td>0.849072</td>
      <td>0.644060</td>
      <td>0.167562</td>
      <td>0.156433</td>
      <td>0.757976</td>
      <td>-0.279361</td>
      <td>-0.749543</td>
      <td>-0.794889</td>
      <td>0.834415</td>
      <td>0.785353</td>
      <td>0.221046</td>
      <td>-0.221046</td>
    </tr>
    <tr>
      <td>engine-size</td>
      <td>-0.110581</td>
      <td>0.112360</td>
      <td>0.572027</td>
      <td>0.685025</td>
      <td>0.729436</td>
      <td>0.074694</td>
      <td>0.849072</td>
      <td>1.000000</td>
      <td>0.572609</td>
      <td>0.209523</td>
      <td>0.028889</td>
      <td>0.822676</td>
      <td>-0.256733</td>
      <td>-0.650546</td>
      <td>-0.679571</td>
      <td>0.872335</td>
      <td>0.745059</td>
      <td>0.070779</td>
      <td>-0.070779</td>
    </tr>
    <tr>
      <td>bore</td>
      <td>-0.140019</td>
      <td>-0.029862</td>
      <td>0.493244</td>
      <td>0.608971</td>
      <td>0.544885</td>
      <td>0.180449</td>
      <td>0.644060</td>
      <td>0.572609</td>
      <td>1.000000</td>
      <td>-0.055390</td>
      <td>0.001263</td>
      <td>0.566936</td>
      <td>-0.267392</td>
      <td>-0.582027</td>
      <td>-0.591309</td>
      <td>0.543155</td>
      <td>0.554610</td>
      <td>0.054458</td>
      <td>-0.054458</td>
    </tr>
    <tr>
      <td>stroke</td>
      <td>-0.008245</td>
      <td>0.055563</td>
      <td>0.158502</td>
      <td>0.124139</td>
      <td>0.188829</td>
      <td>-0.062704</td>
      <td>0.167562</td>
      <td>0.209523</td>
      <td>-0.055390</td>
      <td>1.000000</td>
      <td>0.187923</td>
      <td>0.098462</td>
      <td>-0.065713</td>
      <td>-0.034696</td>
      <td>-0.035201</td>
      <td>0.082310</td>
      <td>0.037300</td>
      <td>0.241303</td>
      <td>-0.241303</td>
    </tr>
    <tr>
      <td>compression-ratio</td>
      <td>-0.182196</td>
      <td>-0.114713</td>
      <td>0.250313</td>
      <td>0.159733</td>
      <td>0.189867</td>
      <td>0.259737</td>
      <td>0.156433</td>
      <td>0.028889</td>
      <td>0.001263</td>
      <td>0.187923</td>
      <td>1.000000</td>
      <td>-0.214514</td>
      <td>-0.435780</td>
      <td>0.331425</td>
      <td>0.268465</td>
      <td>0.071107</td>
      <td>-0.299372</td>
      <td>0.985231</td>
      <td>-0.985231</td>
    </tr>
    <tr>
      <td>horsepower</td>
      <td>0.075819</td>
      <td>0.217299</td>
      <td>0.371147</td>
      <td>0.579821</td>
      <td>0.615077</td>
      <td>-0.087027</td>
      <td>0.757976</td>
      <td>0.822676</td>
      <td>0.566936</td>
      <td>0.098462</td>
      <td>-0.214514</td>
      <td>1.000000</td>
      <td>0.107885</td>
      <td>-0.822214</td>
      <td>-0.804575</td>
      <td>0.809575</td>
      <td>0.889488</td>
      <td>-0.169053</td>
      <td>0.169053</td>
    </tr>
    <tr>
      <td>peak-rpm</td>
      <td>0.279740</td>
      <td>0.239543</td>
      <td>-0.360305</td>
      <td>-0.285970</td>
      <td>-0.245800</td>
      <td>-0.309974</td>
      <td>-0.279361</td>
      <td>-0.256733</td>
      <td>-0.267392</td>
      <td>-0.065713</td>
      <td>-0.435780</td>
      <td>0.107885</td>
      <td>1.000000</td>
      <td>-0.115413</td>
      <td>-0.058598</td>
      <td>-0.101616</td>
      <td>0.115830</td>
      <td>-0.475812</td>
      <td>0.475812</td>
    </tr>
    <tr>
      <td>city-mpg</td>
      <td>-0.035527</td>
      <td>-0.225016</td>
      <td>-0.470606</td>
      <td>-0.665192</td>
      <td>-0.633531</td>
      <td>-0.049800</td>
      <td>-0.749543</td>
      <td>-0.650546</td>
      <td>-0.582027</td>
      <td>-0.034696</td>
      <td>0.331425</td>
      <td>-0.822214</td>
      <td>-0.115413</td>
      <td>1.000000</td>
      <td>0.972044</td>
      <td>-0.686571</td>
      <td>-0.949713</td>
      <td>0.265676</td>
      <td>-0.265676</td>
    </tr>
    <tr>
      <td>highway-mpg</td>
      <td>0.036233</td>
      <td>-0.181877</td>
      <td>-0.543304</td>
      <td>-0.698142</td>
      <td>-0.680635</td>
      <td>-0.104812</td>
      <td>-0.794889</td>
      <td>-0.679571</td>
      <td>-0.591309</td>
      <td>-0.035201</td>
      <td>0.268465</td>
      <td>-0.804575</td>
      <td>-0.058598</td>
      <td>0.972044</td>
      <td>1.000000</td>
      <td>-0.704692</td>
      <td>-0.930028</td>
      <td>0.198690</td>
      <td>-0.198690</td>
    </tr>
    <tr>
      <td>price</td>
      <td>-0.082391</td>
      <td>0.133999</td>
      <td>0.584642</td>
      <td>0.690628</td>
      <td>0.751265</td>
      <td>0.135486</td>
      <td>0.834415</td>
      <td>0.872335</td>
      <td>0.543155</td>
      <td>0.082310</td>
      <td>0.071107</td>
      <td>0.809575</td>
      <td>-0.101616</td>
      <td>-0.686571</td>
      <td>-0.704692</td>
      <td>1.000000</td>
      <td>0.789898</td>
      <td>0.110326</td>
      <td>-0.110326</td>
    </tr>
    <tr>
      <td>city-L/100km</td>
      <td>0.066171</td>
      <td>0.238567</td>
      <td>0.476153</td>
      <td>0.657373</td>
      <td>0.673363</td>
      <td>0.003811</td>
      <td>0.785353</td>
      <td>0.745059</td>
      <td>0.554610</td>
      <td>0.037300</td>
      <td>-0.299372</td>
      <td>0.889488</td>
      <td>0.115830</td>
      <td>-0.949713</td>
      <td>-0.930028</td>
      <td>0.789898</td>
      <td>1.000000</td>
      <td>-0.241282</td>
      <td>0.241282</td>
    </tr>
    <tr>
      <td>diesel</td>
      <td>-0.196735</td>
      <td>-0.101546</td>
      <td>0.307237</td>
      <td>0.211187</td>
      <td>0.244356</td>
      <td>0.281578</td>
      <td>0.221046</td>
      <td>0.070779</td>
      <td>0.054458</td>
      <td>0.241303</td>
      <td>0.985231</td>
      <td>-0.169053</td>
      <td>-0.475812</td>
      <td>0.265676</td>
      <td>0.198690</td>
      <td>0.110326</td>
      <td>-0.241282</td>
      <td>1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <td>gas</td>
      <td>0.196735</td>
      <td>0.101546</td>
      <td>-0.307237</td>
      <td>-0.211187</td>
      <td>-0.244356</td>
      <td>-0.281578</td>
      <td>-0.221046</td>
      <td>-0.070779</td>
      <td>-0.054458</td>
      <td>-0.241303</td>
      <td>-0.985231</td>
      <td>0.169053</td>
      <td>0.475812</td>
      <td>-0.265676</td>
      <td>-0.198690</td>
      <td>-0.110326</td>
      <td>0.241282</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

df1= df[['bore','stroke' ,'compression-ratio','horsepower']] 
df1.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>bore</td>
      <td>1.000000</td>
      <td>-0.055390</td>
      <td>0.001263</td>
      <td>0.566936</td>
    </tr>
    <tr>
      <td>stroke</td>
      <td>-0.055390</td>
      <td>1.000000</td>
      <td>0.187923</td>
      <td>0.098462</td>
    </tr>
    <tr>
      <td>compression-ratio</td>
      <td>0.001263</td>
      <td>0.187923</td>
      <td>1.000000</td>
      <td>-0.214514</td>
    </tr>
    <tr>
      <td>horsepower</td>
      <td>0.566936</td>
      <td>0.098462</td>
      <td>-0.214514</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<h2>Continuous numerical variables:</h2> 



<h4>Positive linear relationship</h4>

Let's find the scatterplot of "engine-size" and "price" 


```python
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
```




    (0, 55818.67274651027)




![png](output_17_1.png)


 examine the correlation between 'engine-size' and 'price' 


```python
df[["engine-size", "price"]].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>engine-size</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>engine-size</td>
      <td>1.000000</td>
      <td>0.872335</td>
    </tr>
    <tr>
      <td>price</td>
      <td>0.872335</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Highway mpg is a potential predictor variable of price 


```python
sns.regplot(x="highway-mpg", y="price", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x186a58e4288>




![png](output_21_1.png)



```python
df[['highway-mpg', 'price']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>highway-mpg</td>
      <td>1.000000</td>
      <td>-0.704692</td>
    </tr>
    <tr>
      <td>price</td>
      <td>-0.704692</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<h3>Weak Linear Relationship</h3>


```python
sns.regplot(x="peak-rpm", y="price", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x186a59b56c8>




![png](output_24_1.png)



```python
df[['peak-rpm','price']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>peak-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>peak-rpm</td>
      <td>1.000000</td>
      <td>-0.101616</td>
    </tr>
    <tr>
      <td>price</td>
      <td>-0.101616</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

df2 = df[["stroke","price"]]
df2.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stroke</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>stroke</td>
      <td>1.00000</td>
      <td>0.08231</td>
    </tr>
    <tr>
      <td>price</td>
      <td>0.08231</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python

sns.regplot(x="stroke", y="price", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x186a5a00548>




![png](output_27_1.png)



```python
sns.boxplot(x="body-style", y="price", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x186a5a9e648>




![png](output_28_1.png)



```python
sns.boxplot(x="engine-location", y="price", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x186a5b72e48>




![png](output_29_1.png)


examine "drive-wheels" and "price".


```python
# drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x186a5bf0508>




![png](output_31_1.png)


<h2 id="discriptive_statistics">3. Descriptive Statistical Analysis</h2>


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>city-L/100km</th>
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>201.000000</td>
      <td>201.00000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>197.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
      <td>201.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.840796</td>
      <td>122.00000</td>
      <td>98.797015</td>
      <td>0.837102</td>
      <td>0.915126</td>
      <td>53.766667</td>
      <td>2555.666667</td>
      <td>126.875622</td>
      <td>3.330692</td>
      <td>3.256904</td>
      <td>10.164279</td>
      <td>103.405534</td>
      <td>5117.665368</td>
      <td>25.179104</td>
      <td>30.686567</td>
      <td>13207.129353</td>
      <td>9.944145</td>
      <td>0.099502</td>
      <td>0.900498</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.254802</td>
      <td>31.99625</td>
      <td>6.066366</td>
      <td>0.059213</td>
      <td>0.029187</td>
      <td>2.447822</td>
      <td>517.296727</td>
      <td>41.546834</td>
      <td>0.268072</td>
      <td>0.319256</td>
      <td>4.004965</td>
      <td>37.365700</td>
      <td>478.113805</td>
      <td>6.423220</td>
      <td>6.815150</td>
      <td>7947.066342</td>
      <td>2.534599</td>
      <td>0.300083</td>
      <td>0.300083</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-2.000000</td>
      <td>65.00000</td>
      <td>86.600000</td>
      <td>0.678039</td>
      <td>0.837500</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>2.540000</td>
      <td>2.070000</td>
      <td>7.000000</td>
      <td>48.000000</td>
      <td>4150.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>5118.000000</td>
      <td>4.795918</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.000000</td>
      <td>101.00000</td>
      <td>94.500000</td>
      <td>0.801538</td>
      <td>0.890278</td>
      <td>52.000000</td>
      <td>2169.000000</td>
      <td>98.000000</td>
      <td>3.150000</td>
      <td>3.110000</td>
      <td>8.600000</td>
      <td>70.000000</td>
      <td>4800.000000</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>7775.000000</td>
      <td>7.833333</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.000000</td>
      <td>122.00000</td>
      <td>97.000000</td>
      <td>0.832292</td>
      <td>0.909722</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>3.310000</td>
      <td>3.290000</td>
      <td>9.000000</td>
      <td>95.000000</td>
      <td>5125.369458</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>10295.000000</td>
      <td>9.791667</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.000000</td>
      <td>137.00000</td>
      <td>102.400000</td>
      <td>0.881788</td>
      <td>0.925000</td>
      <td>55.500000</td>
      <td>2926.000000</td>
      <td>141.000000</td>
      <td>3.580000</td>
      <td>3.410000</td>
      <td>9.400000</td>
      <td>116.000000</td>
      <td>5500.000000</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>16500.000000</td>
      <td>12.368421</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>3.000000</td>
      <td>256.00000</td>
      <td>120.900000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>3.940000</td>
      <td>4.170000</td>
      <td>23.000000</td>
      <td>262.000000</td>
      <td>6600.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>45400.000000</td>
      <td>18.076923</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include=['object'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>make</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>engine-type</th>
      <th>num-of-cylinders</th>
      <th>fuel-system</th>
      <th>horsepower-binned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>200</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <td>top</td>
      <td>toyota</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>ohc</td>
      <td>four</td>
      <td>mpfi</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>32</td>
      <td>165</td>
      <td>115</td>
      <td>94</td>
      <td>118</td>
      <td>198</td>
      <td>145</td>
      <td>157</td>
      <td>92</td>
      <td>115</td>
    </tr>
  </tbody>
</table>
</div>



<h3>Value Counts</h3>


```python
df['drive-wheels'].value_counts()
```




    fwd    118
    rwd     75
    4wd      8
    Name: drive-wheels, dtype: int64



We can convert the series to a Dataframe as follows :


```python
df['drive-wheels'].value_counts().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>drive-wheels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fwd</td>
      <td>118</td>
    </tr>
    <tr>
      <td>rwd</td>
      <td>75</td>
    </tr>
    <tr>
      <td>4wd</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



save the results to the dataframe "drive_wheels_counts" and rename the column  'drive-wheels' to 'value_counts'.


```python
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fwd</td>
      <td>118</td>
    </tr>
    <tr>
      <td>rwd</td>
      <td>75</td>
    </tr>
    <tr>
      <td>4wd</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



rename the index to 'drive-wheels':


```python
drive_wheels_counts.index.name = 'drive_wheels'
drive_wheels_counts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value_counts</th>
    </tr>
    <tr>
      <th>drive_wheels</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fwd</td>
      <td>118</td>
    </tr>
    <tr>
      <td>rwd</td>
      <td>75</td>
    </tr>
    <tr>
      <td>4wd</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



repeat the above process for the variable 'engine-location'.


```python
# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value_counts</th>
    </tr>
    <tr>
      <th>engine-location</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>front</td>
      <td>198</td>
    </tr>
    <tr>
      <td>rear</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



<h2 id="basic_grouping">4. Basics of Grouping</h2>


```python
df['drive-wheels'].unique()
```




    array(['rwd', 'fwd', '4wd'], dtype=object)




```python
df_group_one = df[['drive-wheels','body-style','price']]
```


```python
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>drive-wheels</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4wd</td>
      <td>10241.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>fwd</td>
      <td>9244.779661</td>
    </tr>
    <tr>
      <td>2</td>
      <td>rwd</td>
      <td>19757.613333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>drive-wheels</th>
      <th>body-style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4wd</td>
      <td>hatchback</td>
      <td>7603.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4wd</td>
      <td>sedan</td>
      <td>12647.333333</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4wd</td>
      <td>wagon</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>fwd</td>
      <td>convertible</td>
      <td>11595.000000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>fwd</td>
      <td>hardtop</td>
      <td>8249.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>fwd</td>
      <td>hatchback</td>
      <td>8396.387755</td>
    </tr>
    <tr>
      <td>6</td>
      <td>fwd</td>
      <td>sedan</td>
      <td>9811.800000</td>
    </tr>
    <tr>
      <td>7</td>
      <td>fwd</td>
      <td>wagon</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <td>8</td>
      <td>rwd</td>
      <td>convertible</td>
      <td>23949.600000</td>
    </tr>
    <tr>
      <td>9</td>
      <td>rwd</td>
      <td>hardtop</td>
      <td>24202.714286</td>
    </tr>
    <tr>
      <td>10</td>
      <td>rwd</td>
      <td>hatchback</td>
      <td>14337.777778</td>
    </tr>
    <tr>
      <td>11</td>
      <td>rwd</td>
      <td>sedan</td>
      <td>21711.833333</td>
    </tr>
    <tr>
      <td>12</td>
      <td>rwd</td>
      <td>wagon</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">price</th>
    </tr>
    <tr>
      <th>body-style</th>
      <th>convertible</th>
      <th>hardtop</th>
      <th>hatchback</th>
      <th>sedan</th>
      <th>wagon</th>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4wd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7603.000000</td>
      <td>12647.333333</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <td>fwd</td>
      <td>11595.0</td>
      <td>8249.000000</td>
      <td>8396.387755</td>
      <td>9811.800000</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <td>rwd</td>
      <td>23949.6</td>
      <td>24202.714286</td>
      <td>14337.777778</td>
      <td>21711.833333</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">price</th>
    </tr>
    <tr>
      <th>body-style</th>
      <th>convertible</th>
      <th>hardtop</th>
      <th>hatchback</th>
      <th>sedan</th>
      <th>wagon</th>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4wd</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>7603.000000</td>
      <td>12647.333333</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <td>fwd</td>
      <td>11595.0</td>
      <td>8249.000000</td>
      <td>8396.387755</td>
      <td>9811.800000</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <td>rwd</td>
      <td>23949.6</td>
      <td>24202.714286</td>
      <td>14337.777778</td>
      <td>21711.833333</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python

df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
grouped_test_bodystyle
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body-style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>convertible</td>
      <td>21890.500000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>hardtop</td>
      <td>22208.500000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>hatchback</td>
      <td>9957.441176</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sedan</td>
      <td>14459.755319</td>
    </tr>
    <tr>
      <td>4</td>
      <td>wagon</td>
      <td>12371.960000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
%matplotlib inline 
```

<h4>Variables: Drive Wheels and Body Style vs Price</h4>


```python
#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
```


![png](output_55_0.png)



```python
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
```


![png](output_56_0.png)


<h2 id="correlation_causation">5. Correlation and Causation</h2>


```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>wheel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>city-L/100km</th>
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>symboling</td>
      <td>1.000000</td>
      <td>0.466264</td>
      <td>-0.535987</td>
      <td>-0.365404</td>
      <td>-0.242423</td>
      <td>-0.550160</td>
      <td>-0.233118</td>
      <td>-0.110581</td>
      <td>-0.140019</td>
      <td>-0.008245</td>
      <td>-0.182196</td>
      <td>0.075819</td>
      <td>0.279740</td>
      <td>-0.035527</td>
      <td>0.036233</td>
      <td>-0.082391</td>
      <td>0.066171</td>
      <td>-0.196735</td>
      <td>0.196735</td>
    </tr>
    <tr>
      <td>normalized-losses</td>
      <td>0.466264</td>
      <td>1.000000</td>
      <td>-0.056661</td>
      <td>0.019424</td>
      <td>0.086802</td>
      <td>-0.373737</td>
      <td>0.099404</td>
      <td>0.112360</td>
      <td>-0.029862</td>
      <td>0.055563</td>
      <td>-0.114713</td>
      <td>0.217299</td>
      <td>0.239543</td>
      <td>-0.225016</td>
      <td>-0.181877</td>
      <td>0.133999</td>
      <td>0.238567</td>
      <td>-0.101546</td>
      <td>0.101546</td>
    </tr>
    <tr>
      <td>wheel-base</td>
      <td>-0.535987</td>
      <td>-0.056661</td>
      <td>1.000000</td>
      <td>0.876024</td>
      <td>0.814507</td>
      <td>0.590742</td>
      <td>0.782097</td>
      <td>0.572027</td>
      <td>0.493244</td>
      <td>0.158502</td>
      <td>0.250313</td>
      <td>0.371147</td>
      <td>-0.360305</td>
      <td>-0.470606</td>
      <td>-0.543304</td>
      <td>0.584642</td>
      <td>0.476153</td>
      <td>0.307237</td>
      <td>-0.307237</td>
    </tr>
    <tr>
      <td>length</td>
      <td>-0.365404</td>
      <td>0.019424</td>
      <td>0.876024</td>
      <td>1.000000</td>
      <td>0.857170</td>
      <td>0.492063</td>
      <td>0.880665</td>
      <td>0.685025</td>
      <td>0.608971</td>
      <td>0.124139</td>
      <td>0.159733</td>
      <td>0.579821</td>
      <td>-0.285970</td>
      <td>-0.665192</td>
      <td>-0.698142</td>
      <td>0.690628</td>
      <td>0.657373</td>
      <td>0.211187</td>
      <td>-0.211187</td>
    </tr>
    <tr>
      <td>width</td>
      <td>-0.242423</td>
      <td>0.086802</td>
      <td>0.814507</td>
      <td>0.857170</td>
      <td>1.000000</td>
      <td>0.306002</td>
      <td>0.866201</td>
      <td>0.729436</td>
      <td>0.544885</td>
      <td>0.188829</td>
      <td>0.189867</td>
      <td>0.615077</td>
      <td>-0.245800</td>
      <td>-0.633531</td>
      <td>-0.680635</td>
      <td>0.751265</td>
      <td>0.673363</td>
      <td>0.244356</td>
      <td>-0.244356</td>
    </tr>
    <tr>
      <td>height</td>
      <td>-0.550160</td>
      <td>-0.373737</td>
      <td>0.590742</td>
      <td>0.492063</td>
      <td>0.306002</td>
      <td>1.000000</td>
      <td>0.307581</td>
      <td>0.074694</td>
      <td>0.180449</td>
      <td>-0.062704</td>
      <td>0.259737</td>
      <td>-0.087027</td>
      <td>-0.309974</td>
      <td>-0.049800</td>
      <td>-0.104812</td>
      <td>0.135486</td>
      <td>0.003811</td>
      <td>0.281578</td>
      <td>-0.281578</td>
    </tr>
    <tr>
      <td>curb-weight</td>
      <td>-0.233118</td>
      <td>0.099404</td>
      <td>0.782097</td>
      <td>0.880665</td>
      <td>0.866201</td>
      <td>0.307581</td>
      <td>1.000000</td>
      <td>0.849072</td>
      <td>0.644060</td>
      <td>0.167562</td>
      <td>0.156433</td>
      <td>0.757976</td>
      <td>-0.279361</td>
      <td>-0.749543</td>
      <td>-0.794889</td>
      <td>0.834415</td>
      <td>0.785353</td>
      <td>0.221046</td>
      <td>-0.221046</td>
    </tr>
    <tr>
      <td>engine-size</td>
      <td>-0.110581</td>
      <td>0.112360</td>
      <td>0.572027</td>
      <td>0.685025</td>
      <td>0.729436</td>
      <td>0.074694</td>
      <td>0.849072</td>
      <td>1.000000</td>
      <td>0.572609</td>
      <td>0.209523</td>
      <td>0.028889</td>
      <td>0.822676</td>
      <td>-0.256733</td>
      <td>-0.650546</td>
      <td>-0.679571</td>
      <td>0.872335</td>
      <td>0.745059</td>
      <td>0.070779</td>
      <td>-0.070779</td>
    </tr>
    <tr>
      <td>bore</td>
      <td>-0.140019</td>
      <td>-0.029862</td>
      <td>0.493244</td>
      <td>0.608971</td>
      <td>0.544885</td>
      <td>0.180449</td>
      <td>0.644060</td>
      <td>0.572609</td>
      <td>1.000000</td>
      <td>-0.055390</td>
      <td>0.001263</td>
      <td>0.566936</td>
      <td>-0.267392</td>
      <td>-0.582027</td>
      <td>-0.591309</td>
      <td>0.543155</td>
      <td>0.554610</td>
      <td>0.054458</td>
      <td>-0.054458</td>
    </tr>
    <tr>
      <td>stroke</td>
      <td>-0.008245</td>
      <td>0.055563</td>
      <td>0.158502</td>
      <td>0.124139</td>
      <td>0.188829</td>
      <td>-0.062704</td>
      <td>0.167562</td>
      <td>0.209523</td>
      <td>-0.055390</td>
      <td>1.000000</td>
      <td>0.187923</td>
      <td>0.098462</td>
      <td>-0.065713</td>
      <td>-0.034696</td>
      <td>-0.035201</td>
      <td>0.082310</td>
      <td>0.037300</td>
      <td>0.241303</td>
      <td>-0.241303</td>
    </tr>
    <tr>
      <td>compression-ratio</td>
      <td>-0.182196</td>
      <td>-0.114713</td>
      <td>0.250313</td>
      <td>0.159733</td>
      <td>0.189867</td>
      <td>0.259737</td>
      <td>0.156433</td>
      <td>0.028889</td>
      <td>0.001263</td>
      <td>0.187923</td>
      <td>1.000000</td>
      <td>-0.214514</td>
      <td>-0.435780</td>
      <td>0.331425</td>
      <td>0.268465</td>
      <td>0.071107</td>
      <td>-0.299372</td>
      <td>0.985231</td>
      <td>-0.985231</td>
    </tr>
    <tr>
      <td>horsepower</td>
      <td>0.075819</td>
      <td>0.217299</td>
      <td>0.371147</td>
      <td>0.579821</td>
      <td>0.615077</td>
      <td>-0.087027</td>
      <td>0.757976</td>
      <td>0.822676</td>
      <td>0.566936</td>
      <td>0.098462</td>
      <td>-0.214514</td>
      <td>1.000000</td>
      <td>0.107885</td>
      <td>-0.822214</td>
      <td>-0.804575</td>
      <td>0.809575</td>
      <td>0.889488</td>
      <td>-0.169053</td>
      <td>0.169053</td>
    </tr>
    <tr>
      <td>peak-rpm</td>
      <td>0.279740</td>
      <td>0.239543</td>
      <td>-0.360305</td>
      <td>-0.285970</td>
      <td>-0.245800</td>
      <td>-0.309974</td>
      <td>-0.279361</td>
      <td>-0.256733</td>
      <td>-0.267392</td>
      <td>-0.065713</td>
      <td>-0.435780</td>
      <td>0.107885</td>
      <td>1.000000</td>
      <td>-0.115413</td>
      <td>-0.058598</td>
      <td>-0.101616</td>
      <td>0.115830</td>
      <td>-0.475812</td>
      <td>0.475812</td>
    </tr>
    <tr>
      <td>city-mpg</td>
      <td>-0.035527</td>
      <td>-0.225016</td>
      <td>-0.470606</td>
      <td>-0.665192</td>
      <td>-0.633531</td>
      <td>-0.049800</td>
      <td>-0.749543</td>
      <td>-0.650546</td>
      <td>-0.582027</td>
      <td>-0.034696</td>
      <td>0.331425</td>
      <td>-0.822214</td>
      <td>-0.115413</td>
      <td>1.000000</td>
      <td>0.972044</td>
      <td>-0.686571</td>
      <td>-0.949713</td>
      <td>0.265676</td>
      <td>-0.265676</td>
    </tr>
    <tr>
      <td>highway-mpg</td>
      <td>0.036233</td>
      <td>-0.181877</td>
      <td>-0.543304</td>
      <td>-0.698142</td>
      <td>-0.680635</td>
      <td>-0.104812</td>
      <td>-0.794889</td>
      <td>-0.679571</td>
      <td>-0.591309</td>
      <td>-0.035201</td>
      <td>0.268465</td>
      <td>-0.804575</td>
      <td>-0.058598</td>
      <td>0.972044</td>
      <td>1.000000</td>
      <td>-0.704692</td>
      <td>-0.930028</td>
      <td>0.198690</td>
      <td>-0.198690</td>
    </tr>
    <tr>
      <td>price</td>
      <td>-0.082391</td>
      <td>0.133999</td>
      <td>0.584642</td>
      <td>0.690628</td>
      <td>0.751265</td>
      <td>0.135486</td>
      <td>0.834415</td>
      <td>0.872335</td>
      <td>0.543155</td>
      <td>0.082310</td>
      <td>0.071107</td>
      <td>0.809575</td>
      <td>-0.101616</td>
      <td>-0.686571</td>
      <td>-0.704692</td>
      <td>1.000000</td>
      <td>0.789898</td>
      <td>0.110326</td>
      <td>-0.110326</td>
    </tr>
    <tr>
      <td>city-L/100km</td>
      <td>0.066171</td>
      <td>0.238567</td>
      <td>0.476153</td>
      <td>0.657373</td>
      <td>0.673363</td>
      <td>0.003811</td>
      <td>0.785353</td>
      <td>0.745059</td>
      <td>0.554610</td>
      <td>0.037300</td>
      <td>-0.299372</td>
      <td>0.889488</td>
      <td>0.115830</td>
      <td>-0.949713</td>
      <td>-0.930028</td>
      <td>0.789898</td>
      <td>1.000000</td>
      <td>-0.241282</td>
      <td>0.241282</td>
    </tr>
    <tr>
      <td>diesel</td>
      <td>-0.196735</td>
      <td>-0.101546</td>
      <td>0.307237</td>
      <td>0.211187</td>
      <td>0.244356</td>
      <td>0.281578</td>
      <td>0.221046</td>
      <td>0.070779</td>
      <td>0.054458</td>
      <td>0.241303</td>
      <td>0.985231</td>
      <td>-0.169053</td>
      <td>-0.475812</td>
      <td>0.265676</td>
      <td>0.198690</td>
      <td>0.110326</td>
      <td>-0.241282</td>
      <td>1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <td>gas</td>
      <td>0.196735</td>
      <td>0.101546</td>
      <td>-0.307237</td>
      <td>-0.211187</td>
      <td>-0.244356</td>
      <td>-0.281578</td>
      <td>-0.221046</td>
      <td>-0.070779</td>
      <td>-0.054458</td>
      <td>-0.241303</td>
      <td>-0.985231</td>
      <td>0.169053</td>
      <td>0.475812</td>
      <td>-0.265676</td>
      <td>-0.198690</td>
      <td>-0.110326</td>
      <td>0.241282</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy import stats
```

<h3>Wheel-base vs Price</h3>


```python
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
```

    The Pearson Correlation Coefficient is 0.5846418222655081  with a P-value of P = 8.076488270732989e-20
    

<h3>Horsepower vs Price</h3>


```python
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
```

    The Pearson Correlation Coefficient is 0.809574567003656  with a P-value of P =  6.369057428259557e-48
    

<h3>Length vs Price</h3>




```python
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
```

    The Pearson Correlation Coefficient is 0.690628380448364  with a P-value of P =  8.016477466158986e-30
    

<h3>Width vs Price</h3>


```python
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
```

    The Pearson Correlation Coefficient is 0.7512653440522674  with a P-value of P = 9.200335510481516e-38
    

### Curb-weight vs Price


```python
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
```

    The Pearson Correlation Coefficient is 0.8344145257702846  with a P-value of P =  2.1895772388936914e-53
    

<h3>Engine-size vs Price</h3>

Let's calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price':


```python
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
```

    The Pearson Correlation Coefficient is 0.8723351674455185  with a P-value of P = 9.265491622198389e-64
    

<h3>Bore vs Price</h3>


```python
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 
```

    The Pearson Correlation Coefficient is 0.5431553832626602  with a P-value of P =   8.049189483935489e-17
    

<h3>City-mpg vs Price</h3>


```python
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
```

    The Pearson Correlation Coefficient is -0.6865710067844677  with a P-value of P =  2.321132065567674e-29
    

<h3>Highway-mpg vs Price</h3>


```python
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 
```

    The Pearson Correlation Coefficient is -0.7046922650589529  with a P-value of P =  1.7495471144477352e-31
    

<h2 id="anova">6. ANOVA</h2>

<h3>Drive Wheels</h3>


```python
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>drive-wheels</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>rwd</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>rwd</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>fwd</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4wd</td>
      <td>17450.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>fwd</td>
      <td>15250.0</td>
    </tr>
    <tr>
      <td>136</td>
      <td>4wd</td>
      <td>7603.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_gptest
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>drive-wheels</th>
      <th>body-style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>rwd</td>
      <td>convertible</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>rwd</td>
      <td>convertible</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>rwd</td>
      <td>hatchback</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>fwd</td>
      <td>sedan</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4wd</td>
      <td>sedan</td>
      <td>17450.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>196</td>
      <td>rwd</td>
      <td>sedan</td>
      <td>16845.0</td>
    </tr>
    <tr>
      <td>197</td>
      <td>rwd</td>
      <td>sedan</td>
      <td>19045.0</td>
    </tr>
    <tr>
      <td>198</td>
      <td>rwd</td>
      <td>sedan</td>
      <td>21485.0</td>
    </tr>
    <tr>
      <td>199</td>
      <td>rwd</td>
      <td>sedan</td>
      <td>22470.0</td>
    </tr>
    <tr>
      <td>200</td>
      <td>rwd</td>
      <td>sedan</td>
      <td>22625.0</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 3 columns</p>
</div>




```python
grouped_test2.get_group('4wd')['price']
```




    4      17450.0
    136     7603.0
    140     9233.0
    141    11259.0
    144     8013.0
    145    11694.0
    150     7898.0
    151     8778.0
    Name: price, dtype: float64




```python
# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   
```

    ANOVA results: F= 67.95406500780399 , P = 3.3945443577151245e-23
    

#### Separately: fwd and rwd


```python
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )
```

    ANOVA results: F= 130.5533160959111 , P = 2.2355306355677845e-23
    

examine the other groups 

#### 4wd and rwd


```python
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)   
```

    ANOVA results: F= 8.580681368924756 , P = 0.004411492211225333
    

<h4>4wd and fwd</h4>


```python
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)   
```

    ANOVA results: F= 0.665465750252303 , P = 0.41620116697845666
    
