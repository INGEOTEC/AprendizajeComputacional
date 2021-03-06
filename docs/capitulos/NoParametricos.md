---
layout: default
title: Métodos No Paramétricos
nav_order: 8
---

# Métodos No Paramétricos
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer las características de diferentes métodos no paramétricos y aplicarlos para 
resolver problemas de regresión y clasificación.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Los métodos paramétricos asumen que los datos provienen de un modelo común, esto da la ventaja de que el problema 
de estimar el modelo se limita a encontrar los parámetros del mismo, por ejemplo los parametros de una distribución 
gausiana. Por otro lado en los métodos no paramétricos asumen que datos similares se comportan de manera similar, 
estos algoritmos también se les conoces como algoitmos de memoría o basados en instancias.

# Histogramas

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
from collections import Counter
from matplotlib import pylab as plt
```

Con las librerías en el entorno, se continua leyendo el conjunto de datos, dentro del ejemplo usado en [^Wasserman] se eliminaron todos los datos menores a $$0.2$$, esto se 
refleja en la última línea. 

```python  
D = [list(map(float, x.strip().split())) for x in open("a1882_25.dat").readlines()]
D = np.array(D)
D = D[:, 2]
D = D[D <= 0.2]
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
limits = np.linspace(D.min(), D.max(), m + 1)
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
    limits = np.linspace(D.min(), D.max(), m + 1)
    h = limits[1] - limits[0]
    _ = np.searchsorted(limits, D, side='right')
    _[_ == 0] = 1
    _[_ == m + 1] = m
    p_j = Counter(_)
    cuadrado = sum([(x / N)**2 for x in p_j.values()])
    return (2 / ((N - 1) * h)) - ((N + 1) * cuadrado / ((N - 1) * h))
```

donde las partes que no han sido descritas solamente implementan las ecuación $$\hat J(h)$$.

Finalemente se busca minimar el valor $$h$$ que minimiza la ecuación, iterando
por diferentes valores de $$m$$ se obtiene la siguiente gráfica del riesgo con
diferentes niveles de bin. 

![Histograma Riesgo](/AprendizajeComputacional/assets/images/hist-riesgo.png)

# Estimador de Densidad por Kernel

Como se puede observar el histograma es un estimador discreto, otro estimador muy utilizado
que cuenta con la característica de ser suave es el estimador de densidad por kernel, $$K$$,
el cual está definido de la siguiente manera.

$$ \hat f(x) = \frac{1}{hN} \sum_{w \in \mathcal X} K(\frac{x - w}{h}), $$

donde el kernel $$K$$ podría ser $$K(x) = \frac{1}{\sqrt{2\pi}} \exp [-\frac{x^2}{2}]$$, con parámetros $$\mu = 0$$ y $$\sigma=1$$. 

La siguiente figura muestra la estimación obtenida, con $$h=0.01$$, en los datos 
utilizados en el ejemplo del histograma. 

![Histograma Riesgo](/AprendizajeComputacional/assets/images/estimador_kerner.png)

# Estimador de Densidad por Vecinos Cercanos

Dado un conjunto $$\mathcal X$$ y una medida de distancia $$d$$, los $$k$$ vecinos cercanos
a $$x \notin \mathcal X$$ se puede calcular ordenando $$\mathcal X$$ de la siguiente manera. Sea 
$$(\pi_1, \pi_2, \ldots, \pi_N)$$ la permutación tal que 
$$d(w_{\pi_1}, x)=\min_{w \in \mathcal X} d(w, x)$$, donde $$w_{\pi_j} \in \mathcal X$$, 
$$\pi_2$$ corresponde al indice menor quitando de $$\mathcal X$$ el elemento $$x_{\pi_1}$$ y así 
sucesivamente. Usando esta notación los $$k$$ vecinos corresponden a $$(\pi_1, \pi_2, \ldots, \pi_k)$$. 

Una maneara intuitiva de definir $$h$$ sería en lugar de pensar en un valor constante para 
toda la función, utilizar la distancia que existe con el $$k$$ vecino mas cercano, es decir, 
$$h=d(x_{\pi_k}, x)=d_k(x)$$. Remplazando esto en el estimado de densidad por kernel se obtiene:

$$ \hat f(x) = \frac{1}{d_k(x) N} \sum_{w \in \mathcal X} K(\frac{x - w}{d_k(x)}).$$

Utilizando los datos anteriores el estimador por vecinos cercanos,
con $$k=100$$, quedaría como:

![Histograma Riesgo](/AprendizajeComputacional/assets/images/estimador_knn.png)

# Caso multidimensional

Para el caso multidimensional el estimador quedaría como 

$$ \hat f(x) = \frac{1}{h^dN} \sum_{w \in \mathcal X} K(\frac{x - w}{h}), $$

donde $$d$$ corresponde al número de dimensiones. Un kernel utilizado es:

$$ K(x) = (\frac{1}{\sqrt{2\pi}})^d \exp [- \frac{\mid\mid x \mid\mid ^2}{2}].$$

# Clasificador de vecinos cercanos

El clasificador de vecinos cercanos es un clasificador simple de entender, la idea es utilizar el conjunto de entrenamiento y una función de distancia para asignar la clase de acuerdo a los k-vecinos más cercanos al objeto deseado.

{%include vecinos_cercanos.html %}

## KDtree

La manera más sencilla de crear el clasificador de vecinos cercanos es utilizando un método exhaustivo en el cálculo de distancia. Otra forma de realizar esto es mediante el uso de alguna estructura de datos que te permita el no realizar todas las operaciones. Uno de estos métodos puede ser KDTree. 

{%include kdtree.html %}

# Regresión

La idea de utilizar vecinos cercanos no es solamente para problemas de clasificación, en problemas de regresión se puede seguir un razonamiento equivalente como se muestra en el siguiente video. 

{%include vecinos_regresion.html %}

# Referencias

[^Wasserman]: All of Statistics. A Concise Course in Statistical Inference. Larry Wasserman. MIT Press.

[^astronomia]: [http://www.stat.cmu.edu/~larry/all-of-statistics/=Rprograms/a1882_25.dat](http://www.stat.cmu.edu/~larry/all-of-statistics/=Rprograms/a1882_25.dat)