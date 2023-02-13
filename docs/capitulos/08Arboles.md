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
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, load_diabetes
from sklearn.inspection import DecisionBoundaryDisplay
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns
sns.set_theme()
```
---

# Introducción

Los árboles de decisión son una estructura de datos jerárquica, la cual se 
construye utilizando una estrategia de divide y vencerás. Los árboles son un 
método no paramétrico diseñado para problemas
de regresión y clasificación. 

El árbol se camina desde la raíz hacia las hojas; en cada nodo se tiene una 
regla que muestra el camino de acuerdo a la entrada y la hoja indica la clase 
o respuesta que corresponde a la entrada.

# Clasificación
{: #sec:clasificacion }

Utilizando el procedimiento para generar [tres Distribuciones Gausianas](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
se generan las siguientes poblaciones con 
medias $$\mu_1=[5, 5]^T$$, $$\mu_2=[-5, -10]^T$$ y $$\mu_3=[15, -6]^T$$; 
utilizando las matrices de covarianza originales.

![Tres Distribuciones Gausianas](/AprendizajeComputacional/assets/images/clases3-arboles.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

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
</details>
<!--
plt.savefig('clases3-arboles.png', dpi=300)
-->

Con estas tres poblaciones, donde cada distribución genera una clase
se crea un árbol de decisión. El árbol se muestra en la siguiente
figura, donde se observa la siguiente información en cada nodo. La 
primera linea muestra la función de corte, la función de corte
se define en el proceso de construcción del árbol y se usa para 
guiar a cualquier elemento que se quiera clasificar. 

La segunda y tercera linea de información en el nodo, corresponde
a datos obtenidos durante el proceso de entrenamiento, esto son 
el número de elementos que llegaron al nodo y la frecuencia de cada 
clase en ese nodo. Por ejemplo, la raíz (nodo superior)
tiene la función de corte $$x \leq 10.294$$, recibió $$3000$$ elementos
y cada clase tiene $$1000$$ elementos. Utilizando la frecuencia de 
clase se define la función de costo la cual se usa para dividir 
los elementos que llegan al nodo, aquellos para los cuales la función
es verdadera se envían al nodo izquierdo y el resto al nodo derecho. 
De esta manera se observa que $$1999$$ están en el nodo izquierdo 
de la raíz y $$1001$$ en el nodo derecho de la raíz. Los nodos 
que no tiene hijos, se les conoce como hojas; estos nodos son los 
que indican la case, por ejemplo la hoja derecha tiene $$1000$$ 
elementos en la clase $$2$$. 

![Árbol](/AprendizajeComputacional/assets/images/tree.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
X = np.concatenate((X_1, X_2, X_3), axis=0)
y = np.array([1] * 1000 + [2] * 1000 + [3] * 1000)
arbol = tree.DecisionTreeClassifier().fit(X, y)
_ = tree.plot_tree(arbol, impurity=False,
                   feature_names=['x', 'y'], label='none')
```
</details>
<!--
plt.savefig('tree.png', dpi=300)
-->

![Árbol de Decisión y su Función de Decisión](/AprendizajeComputacional/assets/images/tree-funcion-decision.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>
```python
ax = plt.subplot(2, 1, 1)
arbol = tree.DecisionTreeClassifier(min_samples_split=1002).fit(X, y)
_ = tree.plot_tree(arbol, impurity=False, ax=ax,
                   feature_names=['x', 'y'], label='none')
ax = plt.subplot(2, 1, 2)
DecisionBoundaryDisplay.from_estimator(arbol, X, cmap=plt.cm.RdYlBu,
                                       response_method='predict',
                                       ax=ax, xlabel='x', ylabel='y')
for i, color in enumerate('ryb'):
    mask = y == (i + 1)
    plt.scatter(X[mask, 0], X[mask, 1], c=color,
        label=f'{i+1}', cmap=plt.cm.RdYlBu, edgecolor='black')
```
</details>
<!--
plt.savefig('tree-funcion-decision.png', dpi=300)
-->

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


