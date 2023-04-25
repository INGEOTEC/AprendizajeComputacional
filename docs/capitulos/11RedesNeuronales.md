---
layout: default
title: Redes Neuronales
nav_order: 12
---

# Redes Neuronales
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer, diseñar y aplicar redes neuronales artificiales para problemas de regresión y clasificación.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
```
---

# Introducción

Las redes neuronales son sin duda uno de los algoritmos de aprendizaje supervisado que mas han tomado auge en los últimos tiempos. Para iniciar la descripción de redes neuronales se toma de base el algoritmo de [Regresión Logística](/AprendizajeComputacional/capitulos/10Optimizacion/#sec:regresion-logistica-optimizacion) pero en el caso de múltiples clases, es decir, Regresión Logística Multinomial.

# Regresión Logística Multinomial
{: #sec:regresion-logistica-multinomial }

La idea de regresión logística es modelar $$\mathbb P(\mathcal Y=y \mid \mathcal X=\mathbf x)=\textsf{Ber}(y \mid \textsf{sigmoid}(\mathbf w \cdot \mathbf x + w_0)),$$ es decir, que la 
clase $$y$$ esta modelada como una distribución Bernoulli con parámetro $$\textsf{sigmoid}(\mathbf w \cdot \mathbf x + w_0).$$ Siguiendo está definición que es equivalente a la mostrada [anteriormente](/AprendizajeComputacional/capitulos/10Optimizacion/#sec:regresion-logistica-optimizacion), se puede modelar a un problema de multiples clases como $$\mathbb P(\mathcal Y=y \mid \mathcal X=\mathbf x)=\textsf{Cat}(y \mid \textsf{softmax}(W \mathbf x + \mathbf w_0)),$$ es decir, que la clase proviene de una distribución Categórica con parámetros $$\textsf{softmax}(W \cdot \mathbf x + \mathbf w_0),$$ donde $$W \in \mathbb R^{K \times d},$$ $$\mathbf x \in \mathbb R^d$$ y $$\mathbf w_0 \in \mathbb R^d.$$ 

La función $$\textsf{softmax}(\mathbf v)$$, donde $$\mathbf v = W \mathbf x + \mathbf w_0$$ está definida como:

$$\mathbf v_i = \frac{\exp \mathbf v_i}{\sum_{j=1}^K \exp \mathbf v_j}.$$

La función `jax.nn.softmax` implementa $$\textsf{softmax}$$; en el siguiente ejemplo se calcula para el vector `[2, 1, 3]` 

```python
jax.nn.softmax(np.array([2, 1, 3]))
```

dando como resultado `Array([0.24472848, 0.09003057, 0.66524094], dtype=float32)`. 
Se puede observar que $$\textsf{softmax}$$ transforma los valores del vector $$\mathbf v$$ en probabilidades. 

Para seguir explicando este tipo de regresión logística se utilizará el problema del Iris, el cual se obtiene de la siguiente manera, es importante notar que las entradas están normalizadas para tener media cero y desviación estándar uno. 

```python
D, y = load_iris(return_X_y=True)
normalize = StandardScaler().fit(D)
D = normalize.transform(D)
```

El siguiente paso es generar el modelo de la Regresión Logística Multinomial, el cual depende de una matriz de coeficientes $$W \in \mathbb R^{K \times d}$$ y $$\mathbf w_0 \in \mathbb R^d.$$ Los parámetros iniciales se puede generar con la función `parametros_iniciales` tal y como se muestra a continuación.

```python
n_labels = np.unique(y).shape[0]
def parametros_iniciales(key=0):
    key = jax.random.PRNGKey(key)
    d = D.shape[1]
    params = []
    for _ in range(n_labels):
        key, subkey = jax.random.split(key)
        _ = dict(w=jax.random.normal(subkey, (d, )) * jnp.sqrt(2 / d),
                 w0=jnp.ones(1))
        params.append(_)
    return params
```

Utilizando los parámetros en el formato anterior, hace que el modelo se pueda implementar con las siguientes instrucciones. Donde el ciclo es por cada uno de los parámetros de las $$K$$ clases y la última línea calcula el $$\textsf{softmax}.$$

```python
@jax.jit
def modelo(params, X):
    o = []
    for p in params:
        o.append(X @ p['w'] + p['w0'])
    return jax.nn.softmax(jnp.array(o), axis=0).T
```

Una característica importante es que la función de perdida, en este caso, a la [Entropía Cruzada](/AprendizajeComputacional/capitulos/04Rendimiento/#sec:entropia-cruzada), requiere codificada la probabilidad de cada clase en un vector, donde el índice con probabilidad $$1$$ corresponde a la clase, esto se puede realizar con el siguiente código. 

```python
y_oh = jax.nn.one_hot(y, n_labels)
```

Ahora se cuenta con todos los elementos para implementar la función de Entropía Cruzada para múltiples clases, la cual se muestra en el siguiente fragmento. 

```python
@jax.jit
def media_entropia_cruzada(params, X, y_oh):
    hy = modelo(params, X)
    return - (y_oh * jnp.log(hy)).sum(axis=1).mean()
```


## Optimización

El siguiente paso es encontrar los parámetros del modelo, para esto se utiliza el método de optimización visto en [Regresión Logística](/AprendizajeComputacional/capitulos/10Optimizacion/#sec:regresion-logistica-optimizacion) con algunos ajustes. Lo primero es que se desarrolla todo en una función `fit` que recibe el parámetro $$\eta$$, los parámetros a identificar y el número de épocas, es decir, el número de iteraciones que se va a realizar el procedimiento. 

Dentro de la función `fit`se observa la función `update` que calcula los nuevos parámetros, también regresa el valor del error, esto para poder visualizar como va aprendiendo el modelo. La primera linea después de la función `update` genera la función que calculará el valor y el gradiente de la función `media_entropia_cruzada`. Finalmente viene el ciclo donde se realizan las actualizaciones de los parámetros y se guarda el error calculado en cada época. 

```python
def fit(eta, params, epocas=500):
    @jax.jit
    def update(params, eta, X, y_oh):
        _ , gs = error_grad(params, X, y_oh)
        return _, jax.tree_map(lambda p, g: p - eta * g, params, gs)

    error_grad  = jax.value_and_grad(media_entropia_cruzada)
    error = []
    for i in range(epocas):
        _, params = update(params, eta, D, y_oh)
        error.append(_)
    return params, error
```

## Optimización Método Adam
{: #sec:adam }

Como se había visto en [previamente](/AprendizajeComputacional/capitulos/10Optimizacion/#sec:actualizacion-parametros) existen diferentes métodos para encontrar los parámetros, en particular en esta sección se utilizará el método Adam (implementado en la librería [optax](https://optax.readthedocs.io)) para encontrar los parámetros de la Regresión Logística Multinomial. Se decide utilizar este método dado que su uso es frecuente en la identificación de parámetros de redes neuronales. 

Siguiendo la misma estructura que la función `fit`, la función `adam` recibe tres parámetros el primer es la instancia de optimizador, la segunda son los parámetros y finalmente el número de épocas que se va a ejecutar. La primera línea de la función `update` (que se encuentra en `adam`) calcula el valor de la función de error y su gradiente, estos son utilizados por `optimizer.update` para calcular la actualización de parámetros así como el nuevo estado del optimizador, los nuevos parámetros son calculados en la tercera línea y la función regresa los nuevos parámetros, el estado del optimizador y el error en esa iteración. La primera línea después de `update` inicializa el optimizador, después se general la función que calculará el valor y gradiente de la función `media_entropia_cruzada`. El ciclo llama la función `update` y guarda el error encontrado en cada época. 

```python
def adam(optimizer, params, epocas=500):
    @jax.jit
    def update(params, opt_state, X, y_oh):
        loss_value, grads = error_grad(params, X, y_oh)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    opt_state = optimizer.init(params)
    error_grad  = jax.value_and_grad(media_entropia_cruzada)    
    error = []
    for i in range(epocas):
        params, opt_state, loss_value = update(params, opt_state, D, y_oh)
        error.append(loss_value)
    return params, error
```

## Comparación entre Optimizadores

Los optimizadores descritos anteriormente se pueden utilizar con el siguiente código, donde la primera línea calcula los parámetros iniciales, después se llama a la función `fit` para encontrar los parámetros con el primer método. La tercera línea genera una instancia del optimizador Adam; el cual se pasa a la función `adam` para encontrar los parámetros con este método. 

```python
params = parametros_iniciales()
p1, error1 = fit(1e-2, params)
optimizer = optax.adam(learning_rate=1e-2)
p2, error2 = adam(optimizer, params)
```

La siguiente figura muestra cómo la media de la Entropía Cruzada se minimiza con respecto a las épocas para los dos métodos. Se puede observar como el método `adam` converge más rápido y llega a un valor menor de Entropía Cruzada. 

![Comparación Métodos](/AprendizajeComputacional/assets/images/comp-gd-adam.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
df = pd.DataFrame(dict(entropia=np.array(error1),
                       optimizador='fit',
                       epoca=np.arange(1, 501)))

df = pd.concat((df, pd.DataFrame(dict(entropia=np.array(error2),
                                      optimizador='adam',
                                      epoca=np.arange(1, 501)))))

sns.relplot(data=df, x='epoca', y='entropia', hue='optimizador', kind='line')
```  
</details>
<!--
plt.tight_layout()
plt.savefig('comp-gd-adam.png', dpi=300)
-->

Finalmente el accuracy en el conjunto de entrenamiento del modelo estimado con `fit` es $$0.8867$$ (`(y == modelo(p1, D).argmax(axis=1)).mean()`) y del estimado con `adam` es $$0.9667.$$

# Perceptrón

La unidad básica de 
procesamiento es el perceptrón, el cual está definido como 
$$\hat y = \sum_{i=1}^d w_i x_i + w_0$$. 
Utilizando un vector $$x = (1,x_1, x_2, \ldots, x_d)$$, entonces
el perceptrón se puede definir como $$\hat y = w \cdot x$$, donde
$$w=(w_0, w_1, \ldots, w_d)$$. En caso de clasificación se puede
definir una función umbral para seleccionar la clase, una posibilidad
sería la función $$\textsf{sigmoid}$$ y la ecuación sería equivalente
a regresión logística. 

Para el caso de $$K$$ clases, el perceptrón se puede definir como $$\hat y = W \cdot x$$ donde $$W \in \mathbb R^{K \times (d + 1)}$$. Si se requiere conocer
la probabilidad de ocurrencia de cada clase, se puede utilizar la función
softmax, que está definida como:

$$
\begin{eqnarray*}
    f_i &=& w_i \cdot x \\ 
    \hat y_i &=& \frac{\exp f_i}{\sum_{j=1}^K \exp f_k}
\end{eqnarray*}
$$

En el siguiente video se describe con mas detalle la función
softmax y el procedimiento para obtener las derivadas
con respecto a $$f_i$$ y $$f_k$$, lo cual ayudará mas adelante
para estimar $$w_i$$ utilizando la regla de la cadena.  

{%include softmax.html %}

Continuando con la descripción de softmax en el siguiente
video se describe el procedimiento para  
calcular $$\frac{\partial}{\partial f_i} E$$, siendo $$E$$ la función de 
Entropía cruzada:

$$E = -\sum_{(x, y) \in \mathcal X} \sum_{i=1}^K y_i \log s_i(x),$$

donde la función de softmax es $$s_i(x) = \frac{\exp f_i(x)}{\sum_{k=1}^K \exp f_k(x)}$$.

{%include softmax_entropia.html %}

Utilizando la regla de la cadena, se calcula la derivada del error con 
respecto al parámetro $$w_i$$ como:

$$
\begin{eqnarray*}
\frac{\partial}{\partial w_i} E &=& \frac{\partial}{\partial f_i} E \frac{\partial}{\partial w_i} f_i \\
               &=& -\sum_{(x, y) \in \mathcal X} (y_i - s_i(x)) \frac{\partial}{\partial w_i} f_i(x) \\
               &=& -\sum_{(x, y) \in \mathcal X} (y_i - s_i(x)) x
\end{eqnarray*}
$$

quedando que el(los) parámetro(s) se actualiza(n) como: $$w_i = w_i + \eta \sum_{(x, y) \in \mathcal X} (y_i - s_i(x)) x$$

Finalizando la introducción es importante resaltar la relación que existe entre
un algoritmo de Regresión Logística de $$K$$ clases con un perceptrón de
$$K$$ salidas, lo cual se describe a continuación. 

{%include perceptron_reg_log.html %}

# Perceptrón Multicapa

En el caso de combinar dos ($$m$$) perceptrones 
utilizando el siguiente procedimiento: 
$$o = W_o x$$ y la salida sea la entrada de $$\hat y = W_y o$$;
se puede observar que es equivalente al uso de un solo perceptrón, es decir,
$$\hat y = W_y o = W_y W_o x = W x$$, es en este caso donde
cobra importancia el incluir una función no lineal que haga 
la diferencia. Por ejemplo, se podrían apilar las siguientes dos
capas:

$$
\begin{eqnarray*}
    h &=& \textsf{sigmoid}(W_hx) \\
    \hat y &=& W_y h
\end{eqnarray*}
$$

de esta manera se tendrían que estimar las matrices $$W_h$$ y $$W_y$$. En el 
siguiente video se describe esta red neuronal de manera gráfica
incluyendo sus componentes. 

{%include RNN.html %}

# Estimación de parámetros

Para obtener los parámetros de la red neuronal se utiliza propagación hacia
atrás y descenso de gradiente. Hasta el momento se ha descrito descenso de 
gradiente utilizando toda la información del conjunto de entrenamiento. Es
decir, $$E = -\sum_{(x,y) \in \mathcal X} L(y, f(x))$$, donde $$L$$ es una
función de perdida. Esto tiene la desventaja de que se requiere
evaluar todo el conjunto entrenamiento para hacer solamente una actualización
de los parámetros, esto se le conoce como _full-batch learning_. Otro camino
es seleccionar una subconjunto de 
$$\mathcal X$$, i.e., $$\mathcal M \subset \mathcal X$$, 
y calcular el error en ese conjunto, a esto se le conoce como 
_min-batch learning_ y en el caso que la cardinalidad de $$\mathcal M$$
sea 1, entonces se le conoce como _on-line learning_. Completar una 
actualización de los parámetros se denomina una iteración, y pasar
por todas las instancias de $$\mathcal X$$ en alguna iteración se le
denomina 1 época (_epoch_). Finalmente, $$\mathcal M$$ puede ser seleccionado
de manera determinista de tal manera que se pueda saber cuando ocurrió
una época o se puede seleccionar de manera aleatoria, si se hace de manera
aleatoria se le conoce como _Stochastic Gradient Descent_.

Hasta el momento solo se ha mencionado la función $$\textsf{sigmoid}$$ como
función de activación, es importante mencionar que aunque
está función tuvo mucho auge, actualmente no está siendo
tan utilizada por el problema que se ilustra en el siguiente video. 

{%include gradiente.html %}

Una función de activación que no representa el problema de 
desvanecimiento del gradiente y que además tiene una implementación
muy eficiente es: $$\textsf{ReLU}(x) = \max(0, x)$$. Donde la derivada es:

$$\textsf{ReLU'}(x) = 
\begin{cases}
    0 \text{ } x \leq 0 \\
    1 \text{ } x > 0
\end{cases}
$$

# Ejemplo

Finalmente veremos un ejemplo utilizando la implementación de
redes neuronales de la librería _sklearn_. En la siguiente líneas
se carga la librería y los datos del problema del Iris. 

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
``` 

Una vez que se tienen los datos cargados se puede usar entrenar una
red neuronal con los parámetros por defecto y predecir en el mismo conjunto
entrenado. 

```python
ann = MLPClassifier().fit(X, y)
ann.predict(X)
```
