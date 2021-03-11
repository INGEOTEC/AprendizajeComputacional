---
layout: default
title: Métodos No Paramétricos
nav_order: 8
---

# Métodos No Paramétricos
{: .fs-10 }

El **objetivo** de la unidad es conocer las características de diferentes métodos no paramétricos y aplicarlos para 
resolver problemas de regresión y clasificación.

---

# Introducción

Los métodos paramétricos asumen que los datos provienen de un modelo común, esto da la ventaja de que el problema 
de estimar el modelo se limita a encontrar los parámetros del mismo, por ejemplo los parametros de una distribución 
gausiana. Por otro lado en los métodos no paramétricos asumen que datos similares se comportan de manera similar, 
estos algoritmos también se les conoces como algoitmos de memoría o basados en instancias.

## Histogramas

El primer problema que estudiaremos será la estimación no paramétrica de una función de densidad, $$f$$, recordando que se cuenta con una conjunto $$\mathcal X = \{x_i\}$$ que es tomado de $$f$$ y el objetivo es usar $$\mathcal X$$ para encontrar el estimado de la función de densidad $$\hat f$$. 

El **histograma** es una manera para estimar la función de densidad. Para formar un histograma se divide la linea en $$h$$ segmentos disjuntos, los cuales se denominan _bins_. El histograma corresponde a una función constante por partes, donde la altura es la proporción de elementos de $$\mathcal X$$ que caen en el bin analizado. 

Suponiendo que todos los valores en $$\mathcal X$$ están en el rango $$[0, 1]$$, los bins se pueden definir 
como 

$$B_1 = [0, \frac{1}{m}), B_2=[\frac{1}{m}, \frac{2}{m}), \ldots, B_m=[\frac{m-1}{m}, 1],$$ 

donde hay $$m$$ bins y $$h=\frac{1}{m}$$. Se puede definir a $$\hat p_j = \frac{1}{N} \sum_{x \in \mathcal X} I( x \in B_j )$$ y $$p_j = \int_{B_j} f(u) du$$, usando está definición se puede observar que la estimación de $$f$$ es: 

$$ \hat f(x) = \sum_{j=1}^N \frac{\hat p_j}{h} I(x \in B_j). $$

Con esta formulación se puede ver la motivación de usar histogramas como estimador de $$f$$ vease:

$$\mathbb E(\hat f(x)) = \frac{\mathbb E(\hat p_j)}{h} = \frac{p_j}{h} = \frac{\int_{B_j} f(u) du}{h} \approx \frac{hf(x)}{h} = f(x).$$

## Selección de $$h$$

Una parte crítica para usar un histograma es la selección de $$h$$ o equivalente el número de bins del estimador. Utilizando el método descrito en [^Wasserman], el cual se basa en minimizar el riesgo haciendo una validación cruzada, obteniendo la siguiente ecuación:

$$ \hat J(h) = \frac{2}{(N-1) h} - \frac{N+1}{(N-1)h} \sum_{j=1}^N \hat p_j.$$

### Ejemplo

Para illustrar el uso de la ecuación de minimización del riesgo se utilizará en el ejemplo utilizado en [^Wasserman]. Los datos se pueden descargar de [^astronomia].

El primer paso es importar las librerías necesarias para realizar el ejercicio. 

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from matplotlib import pylab as plt
```

Con las librerías en el entorno, se continua leyendo el conjunto de datos y normalizandolos
para que se encuentren en el intervalo de $$[0, 1]$$. 

```python  
D = [list(map(float, x.strip().split())) for x in open("a1882_25.dat").readlines()]
D = np.array(D)
D = D[:, 2]
D = D[D <= 0.2]
D = MinMaxScaler().fit_transform(np.atleast_2d(D).T)[:, 0]
```

Haciendo un paréntesis en el ejemplo, para poder calcular $$\hat p_j$$ es necesario 
calcular el histograma utilizando, dado que los valores están normalizados podemos 
realizar el histograma utilizando algunas funciones de _numpy_ y librerías tradicionales. 

Lo primero es encontrar los límites de los bins y después podemos utilizar una función que
indica el índice donde se debe insertar el valor para que el arreglo quede ordenado. 
El único detalle que hay que considerar es que regresa $$0$$ si el valor es menor que el 
límite inferior y el tamaño del arreglo si es mayor. Entonces las líneas $$3$$ 
y $$4$$ se encargan de arreglar estas dos características. 
Finalmente se cuenta el número de elementos que pertenencen a cada bin.

```python
limits = np.linspace(0, 1, m + 1)
_ = np.searchsorted(limits, D, side='right')
_[_ == 0] = 1
_[_ == m + 1] = m
p_j = Counter(_)
```

Realizando el procedimiento anterior se obtiene el siguiente histograma. 

![Histograma](/AprendizajeComputacional/assets/images/histogram.png)

Uniendo estos elementos se puede definir una función de riesgo de la siguiente 
manera

```python
def riesgo(D, m=10):
    """Riesgo de validación cruzada de histograma"""
    N = D.shape[0]
    h = 1 / m
    limits = np.linspace(0, 1, m + 1)
    _ = np.searchsorted(limits, D, side='right')
    _[_ == 0] = 1
    _[_ == m + 1] = m
    p_j = Counter(_)
    cuadrado = sum([(x / N)**2 for x in p_j.values()])
    return (2 / ((N - 1) * h)) - ((N + 1) * cuadrado / ((N - 1) * h))
```

donde las partes que no han sido descritas solamente implementan las ecuación $$\hat J(h)$$.

![Histograma](/AprendizajeComputacional/assets/images/hist-riesgo.png)

# Clasificador de vecinos cercanos

El clasificador de vecinos cercanos es un clasificador simple de entender, la idea es utilizar el conjunto de entrenamiento y una función de distancia para asignar la clase de acuerdo a los k-vecinos más cercanos al objeto deseado.

{%include vecinos_cercanos.html %}

## KDtree

La manera más sencilla de crear el clasificador de vecinos cercanos es utilizando un método exhaustivo en el cálculo de distancia. Otra forma de realizar esto es mediante el uso de alguna estructura de datos que te permita el no realizar todas las operaciones. Uno de estos métodos puede ser KDTree. 

{%include kdtree.html %}

# Regresión

La idea de utilizar vecinos cercanos no es solamente para problemas de clasificación, en problemas de regresión se puede seguir un razonamiento equivalente como se muestra en el siguiente video. 

{%include vecinos_regresion.html %}

## Referencias

[^Wasserman]: All of Statistics. A Concise Course in Statistical Inference. Larry Wasserman. MIT Press.

[^astronomia]: [http://www.stat.cmu.edu/~larry/all-of-statistics/=Rprograms/a1882_25.dat](http://www.stat.cmu.edu/~larry/all-of-statistics/=Rprograms/a1882_25.dat)