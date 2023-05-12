---
layout: default
title: Comparación de Algoritmos
nav_order: 14
---

# Comparación de Algoritmos
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar diferentes procedimientos estadísticos para comparar y analizar el rendimiento de algoritmos. 

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from scipy.stats import norm, wilcoxon
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score
import numpy as np
```

---

{%include 13Comparacion.html %}

---

# Introducción

Hasta el momento se han descrito diferentes algoritmos de clasificación y regresión; se han presentado diferentes medidas para conocer su rendimiento, pero se ha dejado de lado el conocer la distribución de estas medidas para poder tener mayor información sobre el rendimiento del algoritmo y también poder comparar y seleccionar el algoritmo que tenga las mejores prestaciones ya sea en rendimiento o en complejidad. 

# Intervalos de confianza
{: #sec:intervalos }

El análisis del rendimiento se inicia partiendo de que el rendimiento se puede estimar a partir del conjunto de prueba, $$\mathcal G$$; el valor obtenido estima el rendimiento real, $$\theta$$, el cual se considera una constante. Una manera de conocer el rango de valores donde se puede encontrar $$\theta$$ es generando su intervalo de confianza. El intervalo de confianza de $$\theta$$ está dado por $$C = (a(\mathcal G), b(\mathcal G)),$$ de tal manera 
que $$P_{\theta}(\theta \in C) \geq 1 - \alpha$$. Es importante
mencionar que el intervalo no mide la probabilidad de $$\theta$$ dado que $$\theta$$ es una constante, en su lugar mide de que el valor estimado esté dentro de esos límites con esa probabilidad. Por otro lado se utiliza la notación $$a(\mathcal G)$$ 
y $$b(\mathcal G)$$ para hacer explicito que en este caso los límites del intervalo son obtenidos utilizando el conjunto de prueba. Una manera de entender el intervalo de confianza de cualquier parámetro es suponer que si el parámetro se 
estima $$100$$ veces con el mismo procedimiento, en diferentes muestras, un intervalo del 95% de confianza dice que 95 de las veces la estimación del parámetro estará en el intervalo calculado.

## Método: Distribución Normal

Existen diferentes procedimientos para generar intervalos de confianza, uno de ellos es asumir que la estimación de $$\theta$$, i.e., $$\hat \theta$$ se distribuye como una normal, i.e., $$\hat \theta \sim \mathcal N(\mu, \sigma^2),$$ donde $$\sigma=\textsf{se}=\sqrt{\mathbb V(\hat \theta)}$$ corresponde al [error estándar](/AprendizajeComputacional/capitulos/14Estadistica/#sec:error-estandar) de la estimación $$\hat \theta.$$ En estas condiciones el intervalo está dado por:

$$C = (\hat \theta - z_{\frac{\alpha}{2}}\textsf{se}, \hat \theta + z_{\frac{\alpha}{2}}\textsf{se}),$$

donde $$z_{\frac{\alpha}{2}} = \Phi^{-1}(1 - \frac{\alpha}{2})$$ y $$\Phi$$ es la función de distribución acumulada de una normal. 

## Ejemplo: Accuracy

Recordado que dado una entrada el clasificador puede acertar la clase a la que pertenece esa entrada, entonces el resultado se puede representar como $$1$$ si la respuesta es correcta y $$0$$ de lo contrario. En este caso la respuesta es una variable aleatoria con una distribución de Bernoulli. Recordando que la distribución Bernoulli está definida por un parámetro $$p$$, estimado como $$\hat p = \frac{1}{N} \sum_{i=1}^N \mathcal X_i$$ donde $$\mathcal X_i$$ corresponde al resultado del algoritmo en el $$i$$-ésimo ejemplo. La varianza de una distribución Bernoulli es $$p(1-p)$$ por lo que el error estándar 
es: $$se=\sqrt{\frac{p(1-p)}{N}}$$ dando como resultado el siguiente intervalo:

$$C = (\hat p_N - z_{\frac{\alpha}{2}}\sqrt{\frac{p(1-p)}{N}}, \hat p_N + z_{\frac{\alpha}{2}}\sqrt{\frac{p(1-p)}{N}}).$$

Suponiendo $$N=100$$ y $$p=0.85$$ el siguiente código calcula el intervalo usando $$\alpha=0.05$$

```python
alpha = 0.05
z = norm().ppf(1 - alpha / 2)
p = 0.85
N = 100
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
```

$$C = (0.78, 0.92)$$.

En el caso anterior se supuso que se contaba con los resultados de un algoritmo de clasificación, con el objetivo de completar este ejemplo a continuación se presenta el análisis con un Naive Bayes en el problema del Iris. 

Lo primero que se realiza es cargar los datos y dividir en el conjunto de entrenamiento ($$\mathcal T$$) y prueba ($$\mathcal G$$) como se muestra a continuación. 

```python
X, y = load_iris(return_X_y=True)
T, G, y_t, y_g = train_test_split(X, y, test_size=0.3)
```

El siguiente paso es entrenar el algoritmo y realizar las predicciones en el conjunto de prueba ($$\mathcal G$$) tal y como se muestra en las siguientes instrucciones. 

```python
model = GaussianNB().fit(T, y_t)
hy = model.predict(G)
``` 

Con las predicciones se estima el accuracy y se siguen los pasos para calcular el intervalo de confianza como se ilustra en el siguiente código.
 
```python
X = np.where(y_g == hy, 1, 0)
p = X.mean()
N = X.shape[0]
C = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
```

El intervalo de confianza obtenido es $$C = (0.8953, 1.0158),$$ se puede observar que el límite superior es mayor 
que $$1$$ lo cual no es posible dado que el máximo valor del accuracy es $$1,$$ esto es resultado de generar el intervalo de confianza asumiendo una distribución normal. 

Cuando se cuenta con conjuntos de datos pequeños y además no se ha definido un conjunto de prueba, se puede obtener las predicciones del algoritmo de clasificación mediante el uso de validación cruzada usando K-fold. En el siguiente código se muestra su uso, el cambio solamente es en el procedimiento para obtener las predicciones.

```python
kf = StratifiedKFold(n_splits=10, shuffle=True)
hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])
```

El resto del código es equivalente al usado previamente obteniendo el siguiente intervalo de confianza $$C = (0.9196, 0.9871).$$

```python
X = np.where(y == hy, 1, 0)
p = X.mean()
N = X.shape[0]
C = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
```

## Método: Bootstrap del error estándar

Existen ocasiones donde no es sencillo identificar el error estándar ($$\textsf{se}$$) y por lo mismo no se puede calcular el intervalo de confianza. En estos casos se emplea la técnica de [Bootstrap](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap) para estimar $$\mathbb V(\hat \theta).$$ Un ejemplo donde no es sencillo encontrar analíticamente el error estándar es en el $$recall.$$

Es más sencillo entender este método mediante un ejemplo. Usando el ejercicio de $$N=100$$ y $$p=0.85$$ y $$\alpha=0.05$$ descrito previamente, el siguiente código primero construye las variables aleatorias de tal manera que den $$p=0.85$$

```python
alpha = 0.05
N = 100
z = norm().ppf(1 - alpha / 2)
X = np.zeros(N)
X[:85] = 1
```

`X` es una arreglo que podrían provenir de la evaluación de un clasificador usando alguna medida de similitud entre predicción y valor medido. El siguiente paso es generar seleccionar con remplazo y obtener $$\hat \theta$$ para cada muestra, en este caso $$\hat \theta$$ corresponde a la media. El resultado se guarda en una lista $$B$$ y se repite el experimento $$500$$ veces.

```python
S = np.random.randint(X.shape[0],
                      size=(500, X.shape[0]))
B = [X[s].mean() for s in S]
``` 

El error estándar es y el intervalo de confianza se calcula con las siguientes instrucciones 

```python
se = np.sqrt(np.var(B))
C = (p - z * se, p + z * se)
``` 

El intervalo de confianza corresponde a $$C = (0.7746, 0.9254)$$. Se puede observar que previamente se había obtenido un intervalo de confianza de: $$(0.78, 0.92)$$.

Continuando con el mismo ejemplo pero ahora analizando Naive Bayes en el problema del Iris. El primer paso es obtener evaluar las predicciones que se puede observar en el siguiente código (previamente descrito.)

```python
X, y = load_iris(return_X_y=True)
kf = StratifiedKFold(n_splits=10, shuffle=True)

hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])

X = np.where(y == hy, 1, 0)
```

Realizando la selección con remplazo y calculando el intervalo se obtiene un intervalo de $$C=(0.9215, 0.9852)$$. Se puede observar que previamente se había obtenido un intervalo de $$(0.9196, 0.9871)$$. 

```python
S = np.random.randint(X.shape[0],
                      size=(500, X.shape[0]))
B = [X[s].mean() for s in S]
se = np.sqrt(np.var(B))
C = (p - z * se, p + z * se)
```

## Método: Percentil

Existe otra manera de calcular los intervalos de confianza y es mediante el uso del percentil, utilizando directamente las estimaciones realizadas a $$\hat \theta$$ en la selección. El siguiente código muestra este método usando el ejemplo anterior, obteniendo un intervalo de $$C=(0.9263, 0.9800).$$

```python
alpha = 0.05 / 2
C = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
```

## Ejemplo: macro-recall

Hasta el momento se ha usado una medida de rendimiento para la cual se puede conocer su varianza de manera analítica. Existen problemas donde esta medida no es recomendada, en el siguiente ejemplo utilizaremos macro-recall para medir el rendimiento de Naive Bayes en el problema del Iris. El primer paso es realizar las predicciones del algoritmo usando validación cruzada y hacer la muestra con reemplazo $$B$$. 

```python
alpha = 0.05
z = norm().ppf(1 - alpha / 2)

X, y = load_iris(return_X_y=True)
kf = StratifiedKFold(n_splits=10, shuffle=True)

hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])

S = np.random.randint(hy.shape[0],
                      size=(500, hy.shape[0]))
B = [recall_score(y[s], hy[s], average="macro")
     for s in S]
```

El siguiente paso es calcular el intervalo asumiendo que este se comporta como una normal tal y como se muestra en las siguientes instrucciones; obteniendo un intervalo de $$C=(0.9098, 0.9855).$$

```python
p = np.mean(B)
se = np.sqrt(np.var(B))
C = (p - z * se, p + z * se)
``` 

Completando el ejercicio, el intervalo se puede calcular directamente usando el percentil, como se muestra a continuación, estimando un intervalo de $$C=(0.9167, 0.9765).$$

```python
alpha = 0.05 / 2
C = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
```

# Comparación de Algoritmos

Se han descrito varios procedimientos para conocer los intervalos de
confianza de un algoritmos de aprendizaje. Es momento para describir
la metodología para conocer si dos algoritmos se comportan similar
en un problema dado. 

## Método: Distribución t de Student

Suponiendo que se tienen las medidas de rendimiento de dos algoritmos mediante validación cruzada de K-fold, es decir, se tiene el rendimiento del primer algoritmo como $$p_i^1$$ y del segundo como $$p_i^2$$ en la $$i$$-ésima instancia. Suponiendo que el rendimiento es una normal, entonces la resta, i.e., $$p_i = p_i^1 - p_i^2$$ también sería normal. Dado que se está comparando los algoritmos en los mismos datos, se puede utilizar la prueba $$t$$ de Student de muestras dependientes. La estadística de la prueba está dada por $$\frac{\sqrt{K} m}{S} \sim t_{K-1}$$, donde $$m$$ 
y $$S^2$$ es la media varianza estimada.

En el siguiente ejemplo se compara el rendimiento de Árboles Aleatorios y Naive Bayes en el problema de Breast Cancer. El primer paso es cargar las librerías así como obtener las predicciones de los algoritmos. 

```python
K = 30
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
X, y = datasets.load_breast_cancer(return_X_y=True)

P = []
for tr, ts in kf.split(X, y):
    forest = RandomForestClassifier().fit(X[tr], y[tr]).predict(X[ts])
    naive = GaussianNB().fit(X[tr], y[tr]).predict(X[ts])
    P.append([recall_score(y[ts], hy, average="macro") for hy in [forest, naive]])
P = np.array(P)
```

Como se puede observar la medida de rendimiento es macro-recall. Continuando con el procedimiento para obtener la estadística $$t_{K-1}$$

```python
p = P[:, 0] - P[:, 1]
t = np.sqrt(K) * np.mean(p) / np.std(p)
```

donde el valor de la estadística es $$t = 2.926$$, si el valor está fuera del siguiente intervalo $$(-2.045, 2.045)$$ se rechaza la hipótesis nula de que los dos algoritmos se comportan similar. 

En caso de que la medida de rendimiento no esté normalmente distribuido, la prueba no-parametrica equivalente corresponde a Wilcoxon. En el siguiente ejemplo se muestra como se calcularía obteniendo un $$p_{value}=0.0138$$. En ambos casos podemos concluir que los algoritmos Árboles Aleatorios y Naive Bayes son estadisticamente diferentes con una confianza del 95% en el problema de Breast Cancer.

```python
wilcoxon(P[:, 0], P[:, 1])
```

## Método: Bootstrap en diferencias

Un método para comparar el rendimiento de dos algoritmo que no asume ningún tipo de distribución se puede realizar mediante la técnica de Bootstrap. La idea es calcular las predicciones de los algoritmos y realizar la muestra calculando en cada una la diferencia del rendimiento. Este se procedimiento se explicará mediante un ejemplo. 

El primer paso es calcular las predicciones de los algoritmos, en este caso se realizar una validación cruzada, tal y como se muestra a continuación. 


```python
forest = np.empty_like(y)
naive = np.empty_like(y)
for tr, ts in kf.split(X, y):
    forest[ts] = RandomForestClassifier().fit(X[tr], y[tr]).predict(X[ts])
    naive[ts] = GaussianNB().fit(X[tr], y[tr]).predict(X[ts])
```

El macro-recall para los Bosques Aleatorios es $$0.96$$ (`recall_score(y, forest, average="macro")`) y para el Naive Bayes es $$0.93$$ (`recall_score(y, naive, average="macro")`). Lo que se observa es que los bosques tienen un mejor rendimiento, entonces la distribución de la diferencia del rendimiento entre bosques y Naive Bayes no debería de incluir al cero, si lo incluye la masa que está al lado izquierdo del cero debe de ser menor, esa mas corresponde al valor $$p.$$

Las muestras de la diferencia de rendimiento se pueden calcular de las siguientes instrucciones. 

```python
S = np.random.randint(y.shape[0],
                      size=(500, y.shape[0]))
B = [recall_score(y[s], forest[s], average="macro") - recall_score(y[s], naive[s], average="macro")
     for s in S]
```

Finalmente, el $$p_{value}=0.002$$ (`(np.array(B) < 0).mean()`) corresponde a la proporción de muestras que son menores de $$0$$, es decir, aquellas muestras donde Naive Bayes tiene un mejor desempeño que los bosques. En este caso el $$p_{value}$$ es menor 
que $$0.05$$ por lo que se puede rechazar la hipótesis nula con una confianza superior al 95% y concluir que existe una diferencia estadísticamente significativa en el rendimiento entre los dos algoritmos. 
