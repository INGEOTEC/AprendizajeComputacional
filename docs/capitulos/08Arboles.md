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
figura, donde se observa, en cada nodo interno, la siguiente información. La 
primera linea muestra el identificador del nodo, la segunda corresponde
a la función de corte, la tercera línea es la entropía 
($$H(\mathcal Y) = -\sum_{y \in \mathcal Y} \mathbb P(\mathcal Y=y) \log_2 \mathbb P(\mathcal Y=y)$$), 
la cuarta es el número de elementos que llegaron al nodo y 
la última la frecuencia de cada clase en ese nodo. Por ejemplo, 
la raíz (#0) tiene la función de corte $$x \leq 10.294$$, tiene una entropía 
de $$1.585$$, recibió $$3000$$ elementos y cada clase tiene $$1000$$ ejemplos.

Los hojas (nodos #2, #3, #5, y #6) no cuentan con una función de corte, 
dado que son la parte final del árbol. En el árbol mostrado se observa
que la entropía en todos los casos es $$0$$, lo cual indica que todos los 
elementos que llegaron a ese nodo son de la misma clase. No en todos los 
casos las hojas tienen entropía cero y existen parámetros en la creación
del árbol que permiten crear árboles más simples. Por ejemplo, la hoja #6
tiene solamente un ejemplo, uno se podría preguntar ¿qué pasaría si esa
hoja se elimina? El resultado es tener un árbol más simple. 

![Árbol](/AprendizajeComputacional/assets/images/tree.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
X = np.concatenate((X_1, X_2, X_3), axis=0)
y = np.array([1] * 1000 + [2] * 1000 + [3] * 1000)
arbol = tree.DecisionTreeClassifier(criterion='entropy').fit(X, y)
_ = tree.plot_tree(arbol, node_ids=True,
                   feature_names=['x', 'y'], label='root')
```
</details>
<!--
plt.savefig('tree.png', dpi=300)
-->

La siguiente figura muestra el árbol generado cuando el nodo #6 se quita. Se observa
un árbol con menos nodos, aunque la entropía es diferente de cero en la hoja #4. 
La segunda parte de la figura muestra la función de decisión que genera el árbol 
de decisión. Se observa que cada regla divide el espacio en dos. La raíz (#0)
divide los datos utilizando $$x \leq 10.294$$, donde todos los elementos para los cuales
la función es verdadera se envían al nodo izquierdo (#1), de lo contrario se envían a la 
hoja derecha (#4). Al nodo #1 llegan $$1999$$ los cuales se divide en
utilizando $$y \leq -1.693$$, dando como resultado $$1000$$ en su hoja izquierda (#2)
y $$999$$ en su hoja derecha (#3). 



![Árbol de Decisión y su Función de Decisión](/AprendizajeComputacional/assets/images/tree-funcion-decision.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
ax = plt.subplot(2, 1, 1)
arbol = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=1002).fit(X, y)
_ = tree.plot_tree(arbol, node_ids=True,
                   feature_names=['x', 'y'], label='root')
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

Utilizando el árbol mostrado en la figura anterior, se puede explicar el proceso
de clasificar un nuevo elemento. Por ejemplo, el 
elemento $$\mathbf u=(x=-3, y=0.5)$$ (que se muestra en la siguiente 
figura como un punto negro) se clasifica de la siguiente manera. La función de 
corte ($$x \leq 10.294$$) de la raíz (#0) indica $$\mathbf u$$ pasa al nodo izquierdo #1.
La función de corte ($$y \leq -1.693$$) del nodo #1 indica que $$\mathbf u$$ se
envía a la hoja #3. La clase mayoritaria ($$999$$ elementos) en la hoja #3 
es la clase $$1$$, entonces $$\mathbf u$$ pertenece a la clase $$1$$ de acuerdo 
al árbol de decisión generado. 

![Predicción](/AprendizajeComputacional/assets/images/tree-prediccion.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
DecisionBoundaryDisplay.from_estimator(arbol, X, cmap=plt.cm.RdYlBu,
                                       response_method='predict',
                                       xlabel='x', ylabel='y')
for i, color in enumerate('ryb'):
    mask = y == (i + 1)
    plt.scatter(X[mask, 0], X[mask, 1], c=color,
                label=f'{i+1}', cmap=plt.cm.RdYlBu,
                edgecolor='black')

plt.scatter([-3], [0.5], c='k',
            label=f'{i+1}', cmap=plt.cm.RdYlBu, 
            edgecolor='black')
```
</details>
<!--
plt.savefig('tree-prediccion.png', dpi=300)
-->

## Entrenamiento
{: #sec:clasificacion-entrenamiento }

La construcción un árbol se realiza mediante un procedimiento recursivo en donde 
se aplica una función  de corte $$f_m(\mathbf x)$$ que divide los datos utilizando
el corte que maximice una función de costo. 


etiqueta cada hijo del 
nodo $$m$$ con respecto a las etiquetas del proceso de clasificación. Por ejemplo, en el árbol de la figura anterior, la función $$f_m$$ tiene la forma $$f_m(x) = x_i \leq a$$, donde el parámetro $$i$$ y $$a$$ son aquellos que optimizan una función de 
aptitud. En el siguiente video se ilustra como funciona este proceso recursivo en un problema de 
clasificación.


Para poder seleccionar la variable independiente y el valor que se utilizará para hacer el corte
se requiere medir que tan bueno sería seleccionar la variable y un valor particular. Para medirlo
se utiliza la entropía, $$H(\mathcal X)$$, la cual es una medida de "caos" o sorpresa. En este caso, mide la "uniformidad" de la distribución de probabilidad. Se define como:

$$H(\mathcal X) = \sum_{x \in \mathcal X} - p(x) \log_2 p(x).$$

En la siguiente imagen, se puede ver como el valor de la entropía es máximo cuando se tiene el mismo número de muestras para cada clase, y conforme la uniformidad se pierde, el valor de la entropía se disminuye. El valor de la entropía es igual a 0 cuando todas las muestras pertenecen a la misma clase o categoría. 


Una manera de encontrar los parámetros $$ i $$ y $$ a $$ de la función de corte $$ f_m(x) = x_i \leq a $$ es utilizando la entropía $$ H(\mathcal X) $$. La idea es que en cada corte, se minimice la entropía.

Al evaluar cada posible corte, se calcula la ganancia de entropía en base a la siguiente ecuación:

$$ \textsf{Ganancia} = H( \mathcal X ) - \sum_m \frac{\mid \mathcal X_m \mid}{\mid \mathcal X \mid}  H(\mathcal X_m) $$

donde $$ \mathcal X $$ representa todas las muestras, $$ \mid \mathcal X \mid $$ el número total de muestras, $$ \mathcal X_m $$ las muestras en el nodo $$ m $$ y finalmente, $$ \mid \mathcal X_m \mid $$ el número de muestras en el nodo $$ m $$.

Finalmente, se selecciona el corte cuya ganancia sea máxima.

# Regresión

Hasta este momento se ha visto como se optimizan los parámetros de la función de corte $$f_m$$ para 
problemas de clasificación. 

La única diferencia con problemas de regresión es la función que se utiliza para optimizar los 
parámetros, en el caso de clasificación es entropía y en el caso de regresión podría ser el error 
cuadrático medio o la suma de los errores absolutos. 
