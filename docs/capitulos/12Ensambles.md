---
layout: default
title: Ensambles
nav_order: 13
---

# Ensambles
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar diferentes técnicas para realizar un ensamble de clasificadores o regresores.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from scipy.stats import binom
from sklearn.datasets import load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import recall_score, mean_absolute_percentage_error
from collections import Counter
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
```

---

# Introducción

Como se ha visto hasta el momento, cada algoritmo de clasificación y regresión tiene un sesgo, este puede provenir de los supuestos que se asumieron cuando se entrenó o diseño; por ejemplo, asumir que los datos provienen de una distribución gausiana multivariada o que se pueden separar los ejemplos mediante un hiperplano, entre otros. Dado un problema se desea seleccionar aquel algoritmo que tiene el mejor rendimiento, visto de otra manera, se selecciona el algoritmo cuyo sesga mejor alineado al problema. Una manera complementaria sería utilizar varios algoritmos y tratar de predecir basados en las predicciones individuales de cada algoritmo. En esta unidad se explicarán diferentes metodologías que permiten combinar predicciones de algoritmos de clasificación y regresión. 

# Fundamentos

La descripción de ensambles se empieza observando el siguiente comportamiento. Suponiendo que se cuenta con $$M$$ algoritmos de clasificación binaria cada uno tiene un accuracy de $$p=0.51$$ y estos son completamente independientes. El proceso de clasificar un elemento corresponde a preguntar la clase a los $$M$$ clasificadores y la clase que se recibe mayor votos es la clase seleccionada, esta votación se comporta como una variable aleatoria que tiene una distribución Binomial. Suponiendo con la clase del elemento es $$1$$, en esta condición la función cumulativa de distribución ($$\textsf{cdf}$$) con 
parámetros $$k=\lfloor \frac{M}{2}\rfloor,$$ $$n=M$$ y $$p=0.51$$ 
indica seleccionar la clase $$0$$ y $$1 - \textsf{cdf}$$ corresponde a la probabilidad
de seleccionar la clase $$1$$. 

La siguiente figura muestra como cambia el accuracy, cuando el número de clasificadores se incrementa, cada uno de esos clasificadores son independientes y tiene un accuracy de $$p=0.51,$$ se puede observar que 
cuando $$M=501$$ el accuracy es $$0.673$$ y con $$9,999$$ clasificadores se tiene un accuracy de $$0.977.$$

![1 - CDF Binomial](/AprendizajeComputacional/assets/images/ensamble-binomial.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
N = range(3, 10002, 2)
cdf_c = [1 - binom.cdf(np.floor(n / 2), n, 0.51) for n in N]
df = pd.DataFrame(dict(accuracy=cdf_c, ensamble=N))
sns.relplot(data=df, x='ensamble', y='accuracy', kind='line')
```  
</details>
<!--
plt.tight_layout()
plt.savefig('ensamble-binomial.png', dpi=300)
-->

En el caso de regresión, en particular cuando se usa como función de error el cuadrado del error, i.e., $$(\hat y - y)^2$$ se tiene el intercambio entre varianza y sesgo, el cual se deriva de la siguiente manera. 

$$\begin{eqnarray}
\mathbb E[(\hat y - y)^2] &=&\\
&=&\mathbb E[(\hat y - \mathbb E[\hat y] + \mathbb E[\hat y] - y)^2]\\
&=&\underbrace{\mathbb E[(\hat y - \mathbb E[\hat y])^2]}_{\mathbb V(\hat y)} + \mathbb E[(\mathbb E[\hat y] - y)^2] + 2 \mathbb E[(\hat y - \mathbb E[\hat y])(\mathbb E[\hat y] - y)]\\
&=&\mathbb V(\hat y) + (\underbrace{\mathbb E[\hat y] - y}_{\text{sesgo}})^2 + 2 \underbrace{\mathbb E[(\hat y - \mathbb E[\hat y])]}_{\mathbf E[\hat y] - \mathbb E[\hat y] = 0}(\mathbb E[\hat y] - y)\\
&=&\mathbb V(\hat y) + (\mathbb E[\hat y] - y)^2
\end{eqnarray}$$


Se observa que el cuadrado del error está definido por la varianza de $$\hat y$$ (i.e., $$\mathbb V(\hat y)$$), la cual es independiente de la 
salida $$y$$ y el sesgo al cuadrado del algoritmo (i.e., $$(\mathbb E[\hat y] - y)^2$$).

En el contexto de ensamble, asumiendo que se tienen $$M$$ regresores independientes donde la predicción está dada por $$\bar y = \frac{1}{M}\sum_{i=1}^M \hat y^i$$, se tiene que el sesgo de cada predictor individual es igual al sesgo de su promedio (i.e., $$(\mathbb E[\bar y] - y) = (\mathbb E[\hat y^i] - y)$$) como se puede observar a continuación. 

$$\begin{eqnarray}
\mathbb E[\bar y] &=& \mathbb E[\frac{1}{M} \sum_{i=1}^M \hat y^i]\\
&=&\frac{1}{M} \sum_{i=1}^M \underbrace{\mathbb E[\hat y^i]}_{\mathbb E[\hat y]} =\frac{1}{M} M \mathbb E[\hat y] =\mathbb E[\hat y]
\end{eqnarray}$$

Por otro lado la varianza del promedio (i.e., $$\mathbb V(\bar y)$$) está dada por $$\mathbb V(\bar y)=\frac{1}{M} \mathbb V(\hat y)$$, que se deriva siguiendo los pasos del [error estandar de la media.](/AprendizajeComputacional/capitulos/14Estadistica/#sec:error-estandar-media)

Esto quiere decir que si se tienen $$M$$ regresores independientes, entonces el error cuadrado de su promedio es menor que el error de cada regresor individual, esto es porque su la varianza se reduce tal y como se mostró. 

Tanto en el caso de clasificación como en el caso del error cuadrado, es poco probable contar con clasificadores y regresores que sean completamente independientes, entonces sus predicciones van a estar relacionadas en algún grado y no se podrá llegar a las reducciones obtenidas en el procedimiento presentado. 

# Bagging

Siguiendo con la idea de combinar $$M$$ instancias independientes de un tipo de algoritmo, en esta sección se presenta el algoritmo Bagging (Bootstrap Aggregation) el cual como su nombre lo indica se basa la técnica de [Bootstrap](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap) para generar $$M$$ instancias del algoritmo y la combinación es mediante votación o el promedio en caso de regresión o que se cuente con la probabilidad de cada clase.

## Ejemplo: Dígitos

Para ejemplificar el uso del algoritmo de Bagging se utilizará el conjunto de datos de Dígitos. Estos datos se pueden obtener y generar el conjunto de entrenamiento ($$\mathbb T$$) y prueba ($$\mathbb G $$) con las siguientes instrucciones.

```python
X, y = load_digits(return_X_y=True)
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
```

Los algoritmos que se utilizarán de base son [Máquinas de Soporte Vectorial Lineal](/AprendizajeComputacional/capitulos/09Lineal/#sec:svm) y [Árboles de Decisión](/AprendizajeComputacional/capitulos/08Arboles). Lo primero que se realizará es entrenar una instancia de estos algoritmos para poder comparar su rendimiento en el conjunto de prueba contra Bagging. 

La siguientes instrucciones entrenan una máquina de soporte vectorial, calculando en la segunda línea el [macro-recall](/AprendizajeComputacional/capitulos/04Rendimiento/#sec:macro). El rendimiento se presenta en una tabla para facilitar la comparación. 

```python
svc = LinearSVC(dual=False).fit(T, y_t)
recall_score(y_g, svc.predict(G), average="macro")
```

Complementando las instrucciones anteriores, en el siguiente código se entrena un Árbol de Decisión. 

```python
tree = DecisionTreeClassifier(criterion='entropy',
                              min_samples_split=9).fit(T, y_t)
recall_score(y_g, tree.predict(G), average="macro")
```

El algoritmo de Bootstrap inicia generando las muestras tal y como se realizó en el ejemplo del [error estándar de la media](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap-ejemplo); el siguiente código genera las muestras, en particular el ensamble sería de $$M=11$$ elementos. 

```python
B = np.random.randint(T.shape[0], size=(11, T.shape[0]))
```

Empezando con Bagging usando como clasificador base la Máquina de Soporte Vectorial Lineal. La primera línea de las siguientes instrucciones, entra los máquinas de soporte, después se realizan las predicciones. En la tercera línea se calcula la clase que tuvo la mayor cantidad de votos y finalmente se calcula el error en términos de macro-recall.

```python
svc_ins = [LinearSVC(dual=False).fit(T[b], y_t[b]) for b in B]
hys = np.array([m.predict(G) for m in svc_ins])
hy = np.array([Counter(x).most_common(n=1)[0][0] for x in hys.T])
recall_score(y_g, hy, average="macro")
```

El siguiente algoritmo son los Árboles de Decisión; la única diferencia con respecto a las instrucciones anteriores es la primera línea donde se entrenan los árboles. 

```python
tree_ins = [DecisionTreeClassifier(criterion='entropy',
                                   min_samples_split=9).fit(T[b], y_t[b])
            for b in B]
hys = np.array([m.predict(G) for m in tree_ins])
hy = np.array([Counter(x).most_common(n=1)[0][0] for x in hys.T])
recall_score(y_g, hy, average="macro")
```

Como se mencionó, la predicción final se puede realizar de dos manera en clasificación una es usando votación, como se vió en los códigos anteriores y la segunda es utilizando el promedio de las probabilidades. En el caso de las Máquinas de Soporte Vectorial, estas no calculas las probabilidad de cada clase, pero se cuenta con el valor de la función de decisión, en el siguiente código se usa está información, la segunda y tercera línea normaliza los valores para que ningún valor sea mayor que $$1$$ y menor que $$-1$$ y finalmente se calcula la suma para después seleccionar la clase que corresponde al argumento máximo. 

```python
hys = np.array([m.decision_function(G) for m in svc_ins])
hys = np.where(hys > 1, 1, hys)
hys = np.where(hys < -1, -1, hys)
hys = hys.sum(axis=0)
recall_score(y_g, hys.argmax(axis=1), average="macro")
```

El procedimiento anterior se puede adaptar a los Árboles de Decisión utilizando el siguiente código. 

```python
hys = np.array([m.predict_proba(G) for m in tree_ins])
recall_score(y_g, hys.sum(axis=0).argmax(axis=1), average="macro")
```

Finalmente, la siguiente tabla muestra el rendimiento de las diferentes combinaciones, se puede observar el valor tanto de las Máquinas de Soporte Vectorial (M.S.V) Lineal y de los Árboles de decisión cuando se utilizaron fuera del ensamble; en el segundo renglón se muestra el rendimiento cuando la predicción del ensamble se hizo mediante votación y el último renglón presenta el rendimiento cuando se hace la suma. 

Comparando los diferentes rendimientos, se puede observar que no existe mucha diferencia en rendimiento en las M.S.V Lineal y que la mayor mejora se presentó en los Árboles de Decisión. Este comportamiento es esperado dado que para que Bagging funciones adecuadamente requiere algoritmos inestables, es decir, algoritmos cuyo comportamiento cambia considerablemente con un cambio pequeño en el conjunto de entrenamiento, este es el caso de los Árboles. Por otro lado las M.S.V son algoritmos estables y un cambio pequeño en su conjunto de entrenamiento no tendrá una repercusión considerable en el comportamiento del algoritmo. 

|                   |M.S.V. Lineal|Árboles de Decisión|
|-------------------|-------------|-------------------|
|Único              |$$0.9435$$   |$$0.8527$$         |
|Votación ($$M=11$$)|$$0.9423$$   |$$0.9553$$         |
|Suma ($$M=11$$)    |$$0.9519$$   |$$0.9574$$         |


## Ejemplo: Diabetes

Ahora toca el turno de atacar un problema de regresión mediante Bagging, el problema que se utilizará es el de Diabetes. Las instrucciones para obtener el problema y generar los conjuntos de entrenamiento ($$\mathbb T$$) y prueba ($$\mathbb G $$) se muestra a continuación. 

```python
X, y = load_diabetes(return_X_y=True)
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
```

Tomando el caso de Dígitos como base, el primer algoritmo a entrenar es la M.S.V. Lineal y se usa como mediada de rendimiento el porcentaje del error absoluto; tal y como se muestran en las siguientes instrucciones. 

```python
svr = LinearSVR().fit(T, y_t)
mean_absolute_percentage_error(y_g, svr.predict(G))
```

El árbol de decisión y su rendimiento se implementa con el siguiente código. 

```python
tree = DecisionTreeRegressor(min_samples_split=9).fit(T, y_t)
mean_absolute_percentage_error(y_g, tree.predict(G)
```

Al igual que en el caso de clasificación, la siguiente instrucción genera los índices para generar las muestras. Se hace un ensamble de $$M=11$$ elementos. 

```python
B = np.random.randint(T.shape[0], size=(11, T.shape[0]))
```

En el caso de regresión, la predicción final corresponde al promedio de las predicciones individuales, la primera línea de las siguientes instrucciones se entrena las M.S.V Lineal, en la segunda instrucción se hacen las predicciones y se en la tercera se realiza el promedio y se mide el rendimiento. 

```python
svr_ins = [LinearSVR().fit(T[b], y_t[b]) for b in B]
hys = np.array([m.predict(G) for m in svr_ins])
mean_absolute_percentage_error(y_g, hys.mean(axis=0))
```

De manera equivalente se entrenan los Árboles de Decisión, como se muestra a continuación. 

```python
tree_ins = [DecisionTreeRegressor(min_samples_split=9).fit(T[b], y_t[b]) for b in B]
hys = np.array([m.predict(G) for m in tree_ins])
mean_absolute_percentage_error(y_g, hys.mean(axis=0))
```

La siguiente tabla muestra el rendimiento de los algoritmos de regresión utilizados, al igual que en el caso de clasificación, la M.S.V. no se ve beneficiada con el uso de Bagging. Por otro lado los Árboles de Decisión tienen un incremento en rendimiento considerable al usar Bagging. 

|                       |M.S.V. Lineal|Árboles de Decisión|
|-----------------------|-------------|-------------------|
|Único                  |$$0.4228$$   |$$0.4910$$         |
|Promedio ($$M=11$$)    |$$0.4211$$   |$$0.3918$$         |

Hasta este momento los ensambles han sido de $$M=11$$ elementos, queda la duda como varía el rendimiento con respecto al tamaño del ensamble. La siguiente figura muestra el rendimiento de Bagging utilizando Árboles de Decisión, cuando el ensamble cambia $$M=2,\ldots,500.$$ Se observa que alrededor 
de $$M=200$$ y $$M=300$$ el error está estable y después hay una pequeña disminución en $$M=500.$$ Por otro lado 
de $$M=50$$ a $$M=100$$ hay un incremento en el error. 
 
![Ensamble Diabetes](/AprendizajeComputacional/assets/images/ensamble-diabetes.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
B = np.random.randint(T.shape[0], size=(500, T.shape[0]))
tree_ins = [DecisionTreeRegressor(min_samples_split=9).fit(T[b], y_t[b]) for b in B]
hys = np.array([m.predict(G) for m in tree_ins])

M = range(2, len(tree_ins) + 1)
p = [mean_absolute_percentage_error(y_g, 
                                    hys[:i].mean(axis=0))
     for i in M]
df = pd.DataFrame(dict(error=p, ensamble=M))
sns.relplot(data=df, x='ensamble', y='error', kind='line')

```  
</details>
<!--
plt.tight_layout()
plt.savefig('ensamble-diabetes.png', dpi=300)
-->

# Stack Generalization

Este tipo de ensamble es una generalización a todos los ensambles, la idea es 
utilizar las predicciones de varios clasificadores para generar la predicción 
final. 

En bagging la función que se utilizó fue simplemente utilizar la media de las 
predicciones hecha por los clasificadores base, pero la media podría no ser 
la mejor función que una esta información. 

Por otro lado en Stack Generalization se entrena otro clasificador sobre las 
predicciones para tomar la decisión final. 

En el siguiente video se muestra este procedimiento. 

{%include stack_generalization.html %}