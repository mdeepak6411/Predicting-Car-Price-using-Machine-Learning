<h1> Model Development</h1>

 Import libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
df= pd.read_csv(r'C:\Users\DEEPAK MISHRA\Desktop\data\automobileEDA.csv')
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



<h3>1. Linear Regression and Multiple Linear Regression</h3>

<h4>Linear Regression</h4>

 <b>Linear function:</b>
$$
Yhat = a + b  X
$$

<h4>load the modules for linear regression</h4>


```python
from sklearn.linear_model import LinearRegression
```

<h4>Create the linear regression object</h4>


```python
lm = LinearRegression()
lm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
X = df[['highway-mpg']]
Y = df['price']
```

Fit the linear model using highway-mpg.


```python
lm.fit(X,Y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



 We can output a prediction 


```python
Yhat=lm.predict(X)
Yhat[0:5]   
```




    array([16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085 ,
           20345.17153508])



<h4>What is the value of the intercept (a)?</h4>


```python
lm.intercept_
```




    38423.305858157386



<h4>What is the value of the Slope (b)?</h4>


```python
lm.coef_
```




    array([-821.73337832])




```python
 
lm1 = LinearRegression()
lm1 
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python

lm1.fit(df[['highway-mpg']], df[['price']])
lm1
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



<h4>Slope</h4>


```python
# Write your code below and press Shift+Enter to execute 
# Slope 
lm1.coef_

```




    array([[-821.73337832]])



<h4>Intercept</h4>


```python

# Intercept
lm1.intercept_
```




    array([38423.30585816])



<h4>Multiple Linear Regression</h4>

$$
Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
$$


```python
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
```

Fit the linear model using the four above-mentioned variables.


```python
lm.fit(Z, df['price'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



What is the value of the intercept(a)?


```python
lm.intercept_
```




    -15806.624626329198



What are the values of the coefficients (b1, b2, b3, b4)?


```python
lm.coef_
```




    array([53.49574423,  4.70770099, 81.53026382, 36.05748882])




```python

lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python

lm2.coef_
```




    array([   1.49789586, -820.45434016])



<h3>2)  Model Evaluation using Visualization</h3>


```python
# import the visualization package: seaborn
import seaborn as sns
%matplotlib inline 
```

<h3>Regression Plot</h3>


```python
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
```




    (0, 48269.88316351671)




![png](output_40_1.png)



```python
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
```




    (0, 47422.919330307624)




![png](output_41_1.png)



```python
 
df[["peak-rpm","highway-mpg","price"]].corr()
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
      <th>highway-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>peak-rpm</td>
      <td>1.000000</td>
      <td>-0.058598</td>
      <td>-0.101616</td>
    </tr>
    <tr>
      <td>highway-mpg</td>
      <td>-0.058598</td>
      <td>1.000000</td>
      <td>-0.704692</td>
    </tr>
    <tr>
      <td>price</td>
      <td>-0.101616</td>
      <td>-0.704692</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<h3>Residual Plot</h3>



```python
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
```


![png](output_44_0.png)


<h3>Multiple Linear Regression</h3>

 make a prediction 


```python
Y_hat = lm.predict(Z)
```


```python
plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
```


![png](output_48_0.png)


<p>We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.</p>

<h2>Polynomial Regression and Pipelines</h2>

<p>We will use the following function to plot the data:</p>


```python
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
```

get the variables


```python
x = df['highway-mpg']
y = df['price']
```

 fit the polynomial using the function <b>polyfit</b>, then use the function <b>poly1d</b> to display the polynomial function.


```python
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
```

            3         2
    -1.557 x + 204.8 x - 8965 x + 1.379e+05
    

 plot the function 


```python
PlotPolly(p, x, y, 'highway-mpg')
```


![png](output_58_0.png)



```python
np.polyfit(x, y, 3)
```




    array([-1.55663829e+00,  2.04754306e+02, -8.96543312e+03,  1.37923594e+05])




```python

# calculate polynomial
# Here we use a polynomial of the 11rd order (cubic) 
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p)
PlotPolly(p1,x,y, 'Highway MPG')
```

            3         2
    -1.557 x + 204.8 x - 8965 x + 1.379e+05
    


![png](output_60_1.png)


We can perform a polynomial transform on multiple features. First, we import the module:


```python
from sklearn.preprocessing import PolynomialFeatures
```

We create a <b>PolynomialFeatures</b> object of degree 2: 


```python
pr=PolynomialFeatures(degree=2)
pr
```




    PolynomialFeatures(degree=2, include_bias=True, interaction_only=False,
                       order='C')




```python
Z_pr=pr.fit_transform(Z)
```

The original data is of 201 samples and 4 features 


```python
Z.shape
```




    (201, 4)



after the transformation, there 201 samples and 15 features


```python
Z_pr.shape
```




    (201, 15)



<h2>Pipeline</h2>


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```


```python
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
```


```python
pipe=Pipeline(Input)
pipe
```




    Pipeline(memory=None,
             steps=[('scale',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('polynomial',
                     PolynomialFeatures(degree=2, include_bias=False,
                                        interaction_only=False, order='C')),
                    ('model',
                     LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                                      normalize=False))],
             verbose=False)




```python
pipe.fit(Z,y)
```




    Pipeline(memory=None,
             steps=[('scale',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('polynomial',
                     PolynomialFeatures(degree=2, include_bias=False,
                                        interaction_only=False, order='C')),
                    ('model',
                     LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                                      normalize=False))],
             verbose=False)




```python
ypipe=pipe.predict(Z)
ypipe[0:4]
```




    array([13102.74784201, 13102.74784201, 18225.54572197, 10390.29636555])




```python
# Write your code below and press Shift+Enter to execute 
Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]
```




    array([13699.11161184, 13699.11161184, 19051.65470233, 10620.36193015,
           15521.31420211, 13869.66673213, 15456.16196732, 15974.00907672,
           17612.35917161, 10722.32509097])



<h2> Measures for In-Sample Evaluation</h2>

<h3>Model 1: Simple Linear Regression</h3>

Let's calculate the R^2


```python
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
```

    The R-square is:  0.4965911884339175
    

We can predict the output i.e., "yhat" using the predict method, where X is the input variable:


```python
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
```

    The output of the first four predicted value is:  [16236.50464347 16236.50464347 17058.23802179 13771.3045085 ]
    

 import the function <b>mean_squared_error</b> from the module <b>metrics</b>


```python
from sklearn.metrics import mean_squared_error
```

we compare the predicted results with the actual results 


```python
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
```

    The mean square error of price and predicted value is:  31635042.944639895
    

<h3>Model 2: Multiple Linear Regression</h3>

Let's calculate the R^2


```python
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
```

    The R-square is:  0.8093562806577457
    


```python
Y_predict_multifit = lm.predict(Z)
```


```python
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))
```

    The mean square error of price and predicted value using multifit is:  11980366.87072649
    

<h3>Model 3: Polynomial Fit</h3>


```python
from sklearn.metrics import r2_score
```


```python
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
```

    The R-square value is:  0.6741946663906513
    

<h3>MSE</h3>


```python
mean_squared_error(df['price'], p(x))
```




    20474146.42636125



<h2> Prediction and Decision Making</h2>



```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline 
```

Create a new input 


```python
new_input=np.arange(1, 100, 1).reshape(-1, 1)
```

 Fit the model 


```python
lm.fit(X, Y)
lm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



Produce a prediction


```python
yhat=lm.predict(new_input)
yhat[0:5]
```




    array([37601.57247984, 36779.83910151, 35958.10572319, 35136.37234487,
           34314.63896655])



we can plot the data 


```python
plt.plot(new_input, yhat)
plt.show()
```


![png](output_106_0.png)

