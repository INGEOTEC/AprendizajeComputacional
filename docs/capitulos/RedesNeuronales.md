---
layout: default
title: Redes Neuronales
nav_order: 12
---

# Redes Neuronales
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer, diseñar y aplicar redes neuronales artificiales para problemas de regresión y clasificación

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Las redes neuronales son sin duda uno de los algoritmos de aprendizaje 
supervisado que mas han tomado auge en los últimos tiempos. La unidad básica de 
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
