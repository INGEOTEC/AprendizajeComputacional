---
layout: default
title: Árboles de Decisión
nav_order: 9
---

# Árboles de Decisión
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar árboles de decisión a problemas de clasificación y 
regresión.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, load_diabetes
from scipy.stats import norm
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns
sns.set_theme()
```
---

# Introducción

Los árboles de decisión son una estructura de datos jerárquica, la cual se construye utilizando una 
estrategia de divide y vencerás. Los árboles están conformados por nodos internos, donde se realizan 
operaciones, y hojas las cuales indican la clase o la respuesta.

```python
X_1 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]]).rvs(1000)
X_2 = multivariate_normal(mean=[-5, -10], cov=[[2, 1], [1, 3]]).rvs(1000)
X_3 = multivariate_normal(mean=[15, -6], cov=[[2, 3], [3, 7]]).rvs(1000)

df = pd.DataFrame([dict(x=x, y=y, clase=1) for x, y in X_1] + \
                  [dict(x=x, y=y, clase=2) for x, y in X_2] + \
                  [dict(x=x, y=y, clase=3) for x, y in X_3])

sns.relplot(data=df, kind='scatter',
            x='x', y='y', hue='clase')
```

<!--
plt.savefig()
-->

En la siguiente figura se muestra un ejemplo de un árbol de decisión.

![Árbol](/AprendizajeComputacional/assets/images/tree.png)

La descripción de la figura anterior se puede observar en el siguiente video.

{%include arboles.html %}

# Construcción de un Árbol de Decisión

La construcción un árbol se realiza mediante un procedimiento recursivo en donde se aplica una función 
$$f_m(x)$$ que etiqueta cada hijo del nodo $$m$$ con respecto a las etiquetas del proceso de 
clasificación. Por ejemplo, en el árbol de la figura anterior, la función $$f_m$$ tiene la forma $$f_m(x) = x_i \leq a$$, donde el parámetro $$i$$ y $$a$$ son aquellos que optimizan una función de 
aptitud. En el siguiente video se illustra como funciona este proceso recursivo en un problema de 
clasificación.

{%include arboles_construccion.html %}

Para poder seleccionar la variable independiente y el valor que se utilizará para hacer el corte
se requiere medir que tan bueno sería seleccionar la variable y un valor particular. Para medirlo
se utiliza la entropía, $$H(\mathcal X)$$, la cual es una medida de "caos" o sorpresa. En este caso, mide la "uniformidad" de la distribución de probabilidad. Se define como:

$$H(\mathcal X) = \sum_{x \in \mathcal X} - p(x) \log_2 p(x).$$

En la siguiente imagen, se puede ver como el valor de la entropía es máximo cuando se tiene el mismo número de muestras para cada clase, y conforme la uniformidad se pierde, el valor de la entropía se disminuye. El valor de la entropía es igual a 0 cuando todas las muestras pertenecen a la misma clase o categoría. 

![Entropía](/AprendizajeComputacional/assets/images/entropia.png)

Una manera de encontrar los parámetros $$ i $$ y $$ a $$ de la función de corte $$ f_m(x) = x_i \leq a $$ es utilizando la entropía $$ H(\mathcal X) $$. La idea es que en cada corte, se minimice la entropía.

{%include arboles2.html %}

Al evaluar cada posible corte, se calcula la ganancia de entropía en base a la siguiente ecuación:

$$ \textsf{Ganancia} = H( \mathcal X ) - \sum_m \frac{\mid \mathcal X_m \mid}{\mid \mathcal X \mid}  H(\mathcal X_m) $$

donde $$ \mathcal X $$ representa todas las muestras, $$ \mid \mathcal X \mid $$ el número total de muestras, $$ \mathcal X_m $$ las muestras en el nodo $$ m $$ y finalmente, $$ \mid \mathcal X_m \mid $$ el número de muestras en el nodo $$ m $$.

Finalmente, se selecciona el corte cuya ganancia sea máxima.

## Regresión

Hasta este momento se ha visto como se optimizan los parámetros de la función de corte $$f_m$$ para 
problemas de clasificación. 

La única diferencia con problemas de regresión es la función que se utiliza para optimizar los 
parámetros, en el caso de clasificación es entropía y en el caso de regresión podría ser el error 
cuadrático medio o la suma de los errores absolutos. 

# Clasificación

En el siguiente video se describe los efectos que tiene uno de los parámetros de los árboles de decisión,
esto en particular para generar árboles que se pueda entender facilmente. 

{%include arboles_clasificacion.html %}

# Regresión

En el siguiente video se realiza un análisis equivalente al hecho en problemas de clasificación. 

{%include arboles_regresion.html %}


