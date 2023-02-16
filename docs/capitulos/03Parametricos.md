---
layout: default
title: Métodos Paramétricos
nav_order: 4
---

# Métodos Paramétricos
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer las características de los modelos paramétricos y aplicar 
máxima verosimilitud para estimar los parámetros del modelo paramétrico en problemas de regresión y 
clasificación.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from EvoMSA.model import GaussianBayes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_diabetes
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
```
---

{%include 03Parametricos.html %}

---

# Introducción

Existen diferentes tipos de algoritmos que se puede utilizar para resolver problemas de
aprendizaje supervisado y no supervisado. En particular, esta unidad se enfoca en presentar 
las técnicas que se pueden caracterizar como métodos paramétricos. 

Los métodos paramétricos se identifican por asumir que los datos provienen de una distribución 
de la cual se desconocen los parámetros y el procedimiento es encontrar los parámetros 
de la distribución que mejor modelen los datos. Una vez obtenidos los parámetros 
se cuenta con todos los elementos para utilizar el modelo y predecir la característica para
la cual fue entrenada. 

# Metodología
{: #sec:metodologia-met-parametricos }

Hasta el momento se han presentado ejemplos de los pasos 4 y 5 
de la [metodología general](/AprendizajeComputacional/capitulos/01Tipos/#sec:metodologia-general);
esto fue en la sección de 
[Predicción](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:prediccion-normal) 
y la sección de 
[Error de Clasificación](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:error-clasificacion). 
Esta sección complementa los ejemplos anteriores al utilizar todos pasos de la 
[metodología general de aprendizaje supervisado.](/AprendizajeComputacional/capitulos/01Tipos/#sec:metodologia-general)
En particular se enfoca al paso 3 que corresponde al diseño del algoritmo $$f$$ 
que modela el fenómeno de interés utilizando los datos $$\mathcal T \subset \mathcal D.$$

El algoritmo $$f$$ corresponde a asumir que los datos $$\mathcal D$$ provienen de 
una distribución $$F$$ la cual tiene una serie de parámetros $$\theta$$ que 
son identificados con $$\mathcal T.$$

# Estimación de Parámetros
{: #sec:estimacion-parametros }

Se inicia la descripción de métodos paramétricos presentando el procedimiento general
para estimar los parámetros de una distribución. Se cuenta con un conjunto $$\mathcal D$$
donde los elementos $$x \in \mathcal D$$ son $$x \in \mathbb R^d$$. 
Los elementos $$x \in \mathcal D$$ tienen un distribución $$F$$, 
i.e., $$x \sim F$$, son independientes y $$F$$ está 
definida por la función de densidad de probabilidad $$f_{\theta}$$, que a su vez está definida
por $$\theta$$ parámetros. Utilizando $$\mathcal D$$ el objetivo es identificar
los parámetros $$\theta$$ que hacen observar a $$\mathcal D$$ lo más probable. 

## Verosimilitud

Una solución para maximizar el observar $$\mathcal D$$ es maximizando la verosimilitud. 
La verosimilitud es la función distribución conjunta de los elementos en $$\mathcal D$$,
i.e., $$f_\theta(x_1, x_2, \ldots, x_N).$$ Considerando que la muestras
son independientes 
entonces $$f_\theta(x_1, x_2, \ldots, x_N) = \prod_{x \in \mathcal D} f_\theta (x).$$
La función de verosimilitud considera la ecuación anterior como una función 
de los parámetros $$\theta,$$ es decir,

$$\mathcal L(\theta) = \prod_{x \in \mathcal D} f_\theta (x),$$

siendo el logaritmo de la verosimilitud 

$$\ell(\theta) = \log \mathcal L(\theta) = \sum_{x \in \mathcal D} \log f_\theta (x).$$

## Distribución de Bernoulli

La verosimilitud se ejemplifica con la identificación del parámetro $$p$$ de una distribución 
Bernoulli. Una distribución Bernoulli modela dos estados, por un lado se tiene la clase
negativa identificada por $$0$$; identificando la clase positiva como $$1$$. Entonces, la 
probabilidad de observar $$1$$ es $$\mathbb P(X=1) = p$$ y $$\mathbb P(X=0) = 1 - p$$.
Estas ecuaciones se pueden combinar para definir $$f_\theta(x) = p^x (1 - p)^{1-x}.$$

Utilizando el logaritmo de la verosimilitud se tiene:

$$\ell(p) = \sum_{i=1}^N \log p^{x_i} (1 - p)^{1-x_i} = \sum_{i=1}^N x_i \log p + (1-x_i) \log (1 - p).$$

Recordando que el máximo de $$\ell(\mathcal p)$$ se obtiene 
cuando $$\frac{d}{dp} \ell(\mathcal p) = 0$$, entonces estimar $$p$$ corresponde a resolver 
lo siguiente:

$$ \begin{eqnarray} \frac{d}{dp} \ell(\mathcal p) &=& 0 \\ \frac{d}{dp} [ \sum_{i=1}^N x_i \log p + (1-x_i) \log (1 - p)] &=& 0 \\ \frac{d}{d p} [ \sum_{i=1}^N x_i \log p + \log (1 - p) (N - \sum_{i=1}^N x_i) ] &=& 0\\ \sum_{i=1}^N x_i \frac{d}{d p} \log \mathcal p + (N - \sum_{i=1}^N x_i) \frac{d}{d p} \log (1 - \mathcal p) &=& 0\\ \sum_{i=1}^N x_i \frac{1}{p} + (N - \sum_{i=1}^N x_i) \frac{-1}{(1 - p)} &=& 0\\ \end{eqnarray}$$

Realizando algunas operaciones algebraicas se obtiene:

$$\hat p = \frac{1}{N}\sum_{i=1}^N x_i.$$

## Ejemplo: Distribución Gausiana
{: #sec:estimacion-distribucion-gausiana }

Esta sección sigue un camino práctico presentando el código para estimar
los parámetros de una distribución Gausiana donde se conocen todos los parámetros. 
La distribución se usa para generar 1000 muestras y después de esta población
se estiman los parámetros; de estas manera se tienen todos los elementos para
comparar los parámetros reales $$\theta$$ de los parámetros estimados $$\hat \theta.$$

La distribución que se usará se utilizó para generar un 
[problema sintético](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
de tres clases. Los parámetros de la distribución son: $$\mathbf \mu = [5, 5]^T$$
y  $$\Sigma = \begin{pmatrix} 4 & 0 \\ 0 & 2 \\ \end{pmatrix}.$$ 
La siguiente instrucción se puede utilizar para generar 1000 muestras de esa distribución. 

```python
D = multivariate_normal(mean=[5, 5], 
                        cov=[[4, 0], 
                             [0, 2]]).rvs(size=1000)
```

La media estimada de los datos en `D` se calcula usando la función `np.mean` de
la siguiente manera

```python
mu = np.mean(D, axis=0)
```

donde el eje de operación es el primero que corresponde 
al índice $$0.$$ La media estimada es: $$\hat \mu = [4.9334, 5.0413]^T$$ con un 
[error estándar](/AprendizajeComputacional/capitulos/14Estadistica/#sec:error-estandar-media) (`se`)
de $$[0.0648, 0.0436]^T$$ que se calcula con el siguiente código. 

```python
se = np.std(D, axis=0) / np.sqrt(1000)
```

Hasta el momento se ha estimado $$\mu$$, falta por estimar $$\Sigma$$, que se
puede realizar con la siguiente instrucción

```python
cov = np.cov(D, rowvar=False)
```

donde el parámetro `rowvar` indica la forma en que están proporcionados los datos. 
La estimación da los siguientes 
valores $$\hat \Sigma = \begin{pmatrix}  4.2076 & -0.0694 \\ -0.0694 & 1.9044 \\ \end{pmatrix};$$
se puede observar que son similares al parámetro con que se simularon los datos. 

Siguiendo con la inercia de presentar el error estándar de cada estimación, en las siguientes
instrucciones se presenta el error estándar de $$\hat \Sigma$$, el cual se calcula utilizando
la técnica de [bootstrap](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap)
implementada en el siguiente código. Se puede observar que la función `np.cov` se ejecuta
utilizando la muestra indicada en la variable `s`. El error estándar (`se`) de $$\hat \Sigma$$ 
corresponde a $$\begin{pmatrix} 0.1845 & 0.0869 \\ 0.0869 & 0.0875 \\ \end{pmatrix}.$$ Se
observa que los elementos fuera de la diagonal tienen un error estándar tal
que el cero se encuentra en el intervalo $$\hat \Sigma \pm se;$$ lo cual 
indica que el cero es un valor factible. Lo anterior se puede verificar  
tomando en cuenta que se conoce $$\Sigma$$ y que 
el parámetro real es $$0$$ para aquellos elementos fuera de la diagonal. 


```python
S = np.random.randint(D.shape[0],
                      size=(500, D.shape[0]))
B = [np.cov(D[s], rowvar=False) for s in S]
se = np.std(B, axis=0)
```

# Metodología de Clasificación

Habiendo descrito el proceso para estimar los parámetros de una distribución,
por un lado se presentó de manera teórica con la distribución
[Bernoulli](/AprendizajeComputacional/capitulos/03Parametricos/#distribucción-de-bernoulli)
y de manera práctica con una distribución
[Gausiana](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-distribucion-gausiana),
se está en la posición de usar todos estos elementos para presentar
el proceso completo de clasificación. La [metodología general de aprendizaje supervisado.](/AprendizajeComputacional/capitulos/01Tipos/#sec:metodologia-general) está definida
 por cinco pasos, estos pasos se especializan para el problema 
 de clasificación y regresión, utilizando modelos paramétricos, de la siguiente manera. 

 1. Todo empieza con un conjunto de datos $$\mathcal D$$ que tiene la información del fenómeno de interés.
2. Se selecciona el conjunto de ([entrenamiento](/AprendizajeComputacional/capitulos/03Parametricos/#sec:conjunto-entre-prueba)) $$\mathcal T \subset \mathcal D.$$ 
3. Se diseña un algoritmo, $$f$$, el cual se basa en un [modelo](/AprendizajeComputacional/capitulos/03Parametricos/#sec:model-clasificacion) y la [estimación de sus parámetros](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-parametros) utilizando $$\mathcal T.$$
4. Se utiliza $$f$$ para [predecir.](/AprendizajeComputacional/capitulos/03Parametricos/#sec:prediccion)
5. Se mide el [rendimiento](/AprendizajeComputacional/capitulos/03Parametricos/#sec:rendimiento) utilizando un conjunto de [prueba.](/AprendizajeComputacional/capitulos/03Parametricos/#sec:conjunto-entre-prueba)


La metodología de clasificación se ilustra utilizando el 
[problema sintético](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
de tres clases que se presentó en la unidad de 
[pasada.](/AprendizajeComputacional/capitulos/02Teoria_Decision)
Específicamente las entradas que definían a cada clase estaban en la variables
`X_1`, `X_2` y `X_3`. Entonces las clases se pueden colocar 
en la variable `y` tal como se indica a continuación. 

```python
X = np.concatenate((X_1, X_2, X_3))
y = np.array([1] * 1000 + [2] * 1000 + [3] * 1000)
```

Las variables `X` y `y` contiene la información del conjunto
$$\mathcal D = (\mathcal X, \mathcal Y)$$ donde cada renglón de `X`
es una realización de la variable aleatoria $$\mathcal X$$ y equivalentemente
cada elemento en `y` es una realización de $$\mathcal Y.$$

## Conjunto de Entrenamiento y Prueba
{: #sec:conjunto-entre-prueba }

[Anteriormente](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-distribucion-gausiana) se había utilizado a $$\mathcal D$$ en el procedimiento
de maximizar la verosimilitud, esto porque el objetivo en ese procedimiento
era estimar los parámetros de la distribución. Pero el objetivo en 
aprendizaje supervisado es diseñar un algoritmo (función en este caso)
que modele la relación entre $$\mathcal X$$ y $$\mathcal Y$$. 
Para conocer esto es necesario medir el rendimiento del algoritmo
en instancias que no han sido vistas en el entrenamiento.

En consecuencia, se requieren contar con datos para medir el rendimiento, 
a este conjunto de datos se le 
conoce como el conjunto de prueba, $$\mathcal G$$. $$\mathcal G$$ se crea
a partir de $$\mathcal D$$ de tal manera que $$\mathcal G \cap \mathcal T = \emptyset$$
y $$\mathcal D =  \mathcal G \cup \mathcal T.$$ La siguiente instrucción
se puede utilizar para dividir la generación de estos conjuntos a partir de $$\mathcal D.$$

```python
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
```

El parámetro `test_size` indica la proporción del tamaño de conjunto $$\mathcal G$$ 
en relación con el conjunto $$\mathcal D.$$

## Modelo
{: #sec:model-clasificacion }

El inicio de métodos paramétricos es el 
[Teorema de Bayes](/AprendizajeComputacional/capitulos/02Teoria_Decision/#teorema-de-bayes) $$\mathbb P(\mathcal Y \mid \mathcal X) = \frac{ \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)}{\mathbb P(\mathcal X)}$$ donde se usa la verosimilitud $$\mathbb P(\mathcal X \mid \mathcal Y)$$ y
el prior $$\mathbb P(\mathcal Y)$$ para definir la probabilidad a posteriori $$\mathbb P(\mathcal Y \mid \mathcal X)$$. En métodos paramétricos se asume que se puede modelar la verosimilitud
con una distribución particular, que por lo generar es una distribución Gausiana multivariada. 
Es decir, la variable aleatoria $$\mathcal X$$ dado $$\mathcal Y$$ ($$\mathcal X_{\mid \mathcal Y}$$) es $$\mathcal X_{\mid \mathcal Y} \sim \mathcal N(\mu_{\mathcal Y}, \Sigma_{\mathcal Y}),$$ donde
se observa que los parámetros de la distribución Gausiana dependen de la variable 
aleatoria $$\mathcal Y$$ y estos pueden ser identificados 
cuando $$\mathcal Y$$ tiene un valor específico. 

## Estimación de Parámetros
{: #sec:estimacion-parametros }

Dado que por [definición del problema](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales) se conoce que la verosimilitud para cada clase proviene de una 
Gausiana, i.e., $$\mathcal X_{\mid \mathcal Y} \sim \mathcal N(\mu_{\mathcal Y}, \Sigma_{\mathcal Y}),$$ en esta sección se estimarán los parámetros utilizando este conocimiento. 

El primer paso en la estimación de parámetros es calcular el prior $$\mathbb P(\mathcal Y)$$,
el cual corresponde a clasificar el evento sin observar el valor de $$\mathcal X.$$ 
Esto se puede modelar mediante una distribución Categórica 
con parámetros $$p_i$$ donde $$\sum_i^K p_i = 1$$. 
Estos parámetros se pueden estimar utilizando la función `np.unique` de la siguiente manera

```python
labels, counts = np.unique(y_t, return_counts=True)
prior = counts / counts.sum()
```

La variable `prior` contiene en el primer elemento $$\mathbb P(\mathcal Y=1) = 0.3292,$$
en el segundo $$\mathbb P(\mathcal Y=2) = 0.3412$$ y 
en el tercero $$\mathbb P(\mathcal Y=3) = 0.3296$$ que es
aproximadamente $$\frac{1}{3}$$ el cual es el valor real del prior. 

Siguiendo los pasos en [estimación de parámetros de una Gausiana](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-distribucion-gausiana) se pueden estimar los parámetros
para cada Gausiana dada la clase. Es decir, se tiene que estimar los 
parámetros $$\mu$$ y $$\Sigma$$ para la clase $$1$$, $$2$$ y $$3$$. Esto se puede realizar
iterando por las etiquetas contenidas en la variable `labels` y seleccionando los datos
en `T` que corresponden a la clase analizada, ver el uso de la variable `mask` en el slice
de la linea 4 y 5. Después se inicializa una instancia de la clase `multivariate_normal`
para ser utilizada en el cómputo de la función de densidad de probabilidad. El paso final
es guardar las instancias de las distribuciones en la lista `likelihood`.


```python
likelihood = []
for k in labels:
    mask = y_t == k
    mu = np.mean(T[mask], axis=0)
    cov = np.cov(T[mask], rowvar=False)
    likelihood_k = multivariate_normal(mean=mu, cov=cov)
    likelihood.append(likelihood_k)
```

Los valores estimados para la media, en cada clase 
son: $$\hat \mu_1 = [5.1234, 5.0177]^T,$$ $$\hat \mu_2 = [1.5229, -1.5553]^T$$ 
y $$\hat \mu_3 = [12.5064, -3.4501]^T$$. Para las  matrices de covarianza, 
los valores estimados corresponden 
a $$\hat \Sigma_1 = \begin{pmatrix} 4.0896 & 0.0409 \\ 0.0409 & 1.9562 \\ \end{pmatrix},$$ $$\hat \Sigma_2 = \begin{pmatrix} 1.9533 & 1.0034 \\ 1.0034 & 2.8304 \\ \end{pmatrix}$$ 
y $$\hat \Sigma_3 = \begin{pmatrix} 2.0451 & 3.0328 \\ 3.0328 & 6.8235 \\ \end{pmatrix}.$$
Estas estimaciones se pueden comparar con los 
[parámetros reales.](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
También se puede calcular su error estándar para identificar si el parámetro real, $$\theta$$, 
se encuentra en el intervalo definido 
por $$\hat \theta - 2\hat{se} \leq \hat \theta \leq \hat \theta + 2 \hat{se}$$ 
que corresponde aproximadamente al 95% de confianza asumiendo que la distribución
de la estimación del parámetro es Gausiana. 

## Predicción
{: #sec:prediccion }

Una vez que se tiene la función que modela los datos, se está en condiciones de utilizarla
para [predecir](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:prediccion-normal)
nuevos datos. 

En esta ocasión se organiza el procedimiento de predicción en diferentes funciones, 
la primera función recibe los datos a predecir `X` y los componentes del modelo, que 
son la verosimilitud (`likelihood`) y el `prior`. La función 
calcula $$\mathbb P(\mathcal Y=y \mid \mathcal X=x)$$
que es la probabilidad de cada clase dada la entrada $$x$$. Se puede observar en 
la primera linea que se usa la función de densidad de probabilidad (`pdf`) para
cada clase y esta se multiplica por el `prior` y en la tercera linea se 
calcula la evidencia. Finalmente, se regresa el a posteriori.  

```python
def predict_prob(X, likelihood, prior):
    likelihood = [m.pdf(X) for m in likelihood]
    posterior = np.vstack(likelihood).T * prior
    evidence = posterior.sum(axis=1)
    return posterior / np.atleast_2d(evidence).T
```

La función `predict_proba` se utiliza como base para predecir la clase, 
para la cual se requiere el mapa entre índices y clases que se encuentra
en la variable `labels`. Se observa que se llama a la función `predict_proba`
y después se calcula el argumento que tiene la máxima probabilidad
regresando la etiqueta asociada. 

```python
def predict(X, likelihood, prior, labels):
    _ = predict_prob(X, likelihood, prior)
    return labels[np.argmax(_, axis=1)]
```
## Rendimiento
{: #sec:rendimiento }

El rendimiento del algoritmo se mide en el conjunto de prueba `G`, 
utilizando como medida el [error de clasificación.](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:error-clasificacion) El primer paso es predecir
las clases de los elementos en `G`, utilizando la función `predict` que fue 
diseñada anteriormente. Después se mide el error, con la instrucción 
de la segunda línea. El error que tiene el algoritmo 
en el conjunto de prueba es $$0.01,$$ el cual es ligeramente superior al 
encontrado con el modelo ideal.  

```python
hy = predict(G, likelihood, prior, labels)
error = (y_g != hy).mean()
```

El error estándar se calcula con la siguiente instrucción 
el cual tiene un valor de $$0.0041.$$

```python
se_formula = np.sqrt(error * (1 - error) / y_g.shape[0])
```

# Clasificador Bayesiano Ingenuo

Uno de los clasificadores mas utilizados, sencillo de implementar y competitivo, es
el clasificador Bayesiano Ingenuo. 
[Anteriormente](/AprendizajeComputacional/capitulos/03Parametricos/#sec:model-clasificacion)
se asumió que la variable aleatoria $$\mathcal X = (\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_d)$$
dado $$\mathcal Y$$ ($$\mathcal X_{\mid \mathcal Y}$$) 
es $$\mathcal X_{\mid \mathcal Y} \sim \mathcal N(\mu_{\mathcal Y}, \Sigma_{\mathcal Y}),$$ 
donde $$\mu_{\mathcal Y} \in \mathbb R^d$$, $$\Sigma_{\mathcal Y} \in \mathbb R^{d \times d}$$ 
y $$f(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_d)$$ 
es la función de densidad de probabilidad conjunta.

En el clasificador Bayesiano Ingenuo se asume que las variables $$\mathcal X_i$$ 
y $$\mathcal X_j$$ para $$i \neq j$$ son independientes, esto trae como consecuencia
que $$f(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_d) = \prod_i^d f(\mathcal X_i).$$
Esto quiere decir que cada variable está definida como una Gausina donde se tiene que 
identificar $$\mu$$ y $$\sigma^2.$$

La estimación de los parámetros de estas distribuciones se puede realizar 
utilizando un código similar siendo la única diferencia que en se calcula $$\sigma^2$$ 
de cada variable en lugar de la covarianza $$\Sigma$$, esto se puede observar 
en la quinta línea donde se usa la función `np.var` en el primer eje. El resto del 
código es equivalente al usado en la
[estimación de parámetros.](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-parametros)

```python
likelihood = []
for k in labels:
    mask = y_t == k
    mu = np.mean(T[mask], axis=0)
    var = np.var(T[mask], axis=0, ddof=1)
    likelihood_k = multivariate_normal(mean=mu, cov=var)
    likelihood.append(likelihood_k)
```

Los parámetros estimados en la versión ingenua son equivalentes 
con respecto a las medias,
i.e., $$\hat \mu_1 = [5.1234, 5.0177]^T$$, $$\hat \mu_2 = [1.5229, -1.5553]^T$$ 
y $$\hat \mu_3 = [12.5064, -3.4501]^T$$. La diferencia se puede observar
en las varianzas, que a continuación se muestran como matrices de covarianza para
resaltar la diferencia, 
i.e., $$\hat \Sigma_1 = \begin{pmatrix} 4.0896 & 0.0 \\ 0.0 & 1.9562 \\ \end{pmatrix}$$
, $$\hat \Sigma_2 = \begin{pmatrix} 1.9533 & 0.0 \\ 0.0 & 2.8304 \\ \end{pmatrix}$$ 
y $$\hat \Sigma_3 = \begin{pmatrix} 2.0451 & 0.0 \\ 0.0 & 6.8235 \\ \end{pmatrix}$$
se observa como los elementos fuera de la diagonal son ceros, lo cual indica
la independencia entra las variables de entrada. 

Finalmente, el código para 
[predecir](/AprendizajeComputacional/capitulos/03Parametricos/#sec:prediccion) no se tiene 
que modificar dado que el modelo está dado en las variables `likelihood` y `prior`.

El `error` del clasificador Bayesiano Ingenuo, en el conjunto de prueba, es 
de $$0.0133$$ y su error estándar (`se_formula`) es $$0.0047.$$ Lo cual se calculó
utilizando las siguientes dos instrucciones. 

```python
error = (y_g != hy_ingenuo).mean()
se_formula = np.sqrt(error * (1 - error) / y_g.shape[0])
```

# Ejemplo: Breast Cancer Wisconsin

Esta sección ilustra el uso del clasificador Bayesiano al 
generar dos modelos (Clasificador Bayesiano y Bayesiano Ingenuo)
del conjunto de datos de *Breast Cancer Wisconsin.* Estos
datos se pueden obtener utilizando la función `load_breast_cancer`
tal y como se muestra a continuación.

```python
X, y = load_breast_cancer(return_X_y=True)
```

El primer paso es contar con los conjuntos de 
[entrenamiento y prueba](/AprendizajeComputacional/capitulos/03Parametricos/#conjunto-de-entrenamiento-y-prueba) para poder realizar de manera
completa la evaluación del proceso de clasificación. Esto se realiza
ejecutando la siguiente instrucción.

```python
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
```

## Entrenamiento

Los dos modelos que se utilizarán será el clasificador Bayesiano Gausiano
y Bayesiano Ingenuo, utilizando la clase `GaussianBayes`
que se explica en el  
[apéndice.](/AprendizajeComputacional/capitulos/15Codigo/#clasificador-bayesiano-gausiano)
Las siguientes dos instrucciones inicializan estos dos clasificadores, 
la única diferencia es el parámetro `naive` que indica si el clasificador
es ingenuo. 

```python
gaussian = GaussianBayes().fit(T, y_t)
naive = GaussianBayes(naive=True).fit(T, y_t)
```

## Predicción

Habiendo definido los dos clasificadores, las predicciones del conjunto de prueba 
se realiza de la siguiente manera. 

```python
hy_gaussian = gaussian.predict(G)
hy_naive = naive.predict(G)
```

## Rendimiento
{: #sec:gaussina-perf-breast_cancer }

El rendimiento de ambos clasificadores se calcula de la siguiente manera 

```python
error_gaussian = (y_g != hy_gaussian).mean()
error_naive = (y_g != hy_naive).mean()
```

El clasificador Bayesiano Gausiano tiene un error de $$0.0614$$
y el error de Bayesiano Ingenuo es $$0.0877.$$ Se ha visto que el error
es una variable aleatoria, entonces la pregunta es saber si esta diferencia
en rendimiento es significativa o es
una diferencia que proviene de la aleatoriedad de los datos. 

# Diferencias en Rendimiento 

Una manera de ver si existe una diferencia en rendimiento es calcular 
la diferencia entre los dos errores de clasificación, esto es 

```python
diff = (y_g != hy_naive).mean() -  (y_g != hy_gaussian).mean()
```

que tiene un valor de $$0.0263$$. De la misma manera que se ha utilizado la 
técnica de 
[bootstrap](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap)
para calcular el error estándar de la media, se puede usar para estimar
el error estándar de la diferencia en rendimiento. El siguiente código
muestra el procedimiento para estimar este error estándar. 

```python
S = np.random.randint(y_g.shape[0],
                      size=(500, y_g.shape[0]))
B = [(y_g[s] != hy_naive[s]).mean() -  (y_g[s] != hy_gaussian[s]).mean()
     for s in S]
se = np.std(B, axis=0)
```

El error estándar de la diferencia de rendimiento es de $$0.0240,$$
una procedimiento simple para saber si la diferencia observada es 
significativa, es dividir la diferencia entre su 
error estándar dando un valor de $$1.0978.$$ En el caso que el valor
fuera igual o superior a 2 se sabría que la diferencia es significativa
con una confianza de al menos 95%, esto asumiendo que la diferencia se
comporta como una distribución Gausiana. 

El histograma de los datos que se tienen en la variable `B`
se observa en la siguiente figura. Se puede ver que la forma del
histograma asemeja una distribución Gausiana y que el cero esta en el cuerpo de
la Gausiana, tal y como lo confirmó el cociente que se calculado.

```python
sns.displot(B, kde=True)
```

<!--
plt.savefig('comp_bayes_breast_cancer.png', dpi=300)
-->

![Diferencia entre Clasificadores Bayesianos](/AprendizajeComputacional/assets/images/comp_bayes_breast_cancer.png)

Se puede conocer la probabilidad de manera exacta calculando
el área bajo la curva a la izquierda del cero, este sería el valor $$p$$, 
si este es menor a 0.05 quiere decir que se tiene una confianza mayor 
del 95% de que los rendimientos son diferentes. Para este ejemplo,
el área se calcula con el siguiente código

```python
dist = norm(loc=diff, scale=se)
dist.cdf(0)
```

teniendo el valor de $$0.1361$$, lo que significa que 
se tiene una confianza del 86% de que los dos algoritmos 
son diferentes considerando el error de clasificación como medida de
rendimiento. 

# Regresión
{: #sec:regresion-ols }

Hasta este momento se han revisado métodos paramétricos en 
clasificación, ahora es el turno de abordar 
el problema de regresión. La diferencia entre clasificación
y regresión como se describió en la sección 
de [aprendizaje supervisado](/AprendizajeComputacional/capitulos/01Tipos/#sec:aprendizaje-supervisado) 
es que en regresión $$\mathcal Y \in \mathbb R.$$

El procedimiento de regresión que se describe en esta sección
es regresión de **Mínimos Cuadrados Ordinaria** (OLS -*Ordinary Least Squares*-), en el 
cual se asume  
que $$\mathcal Y \sim \mathcal N(\mathbf w^T \mathbf x + \epsilon, \sigma^2)$$,
de tal manera que $$y = \mathbb E[\mathcal N(\mathbf w^T \mathbf x + \epsilon, \sigma^2)].$$

<!--
Donde $$\mathbb E[\epsilon] = 0$$ y $$\mathbb V[\epsilon] = \sigma.$$
-->

Trabajando con $$y = \mathbb E[\mathcal N(\mathbf w^T \mathbf x + \epsilon, \sigma^2)],$$
se considera lo siguiente $$y = \mathbb E[\mathcal N(\mathbf w^T \mathbf x, 0) + \mathcal N(0, \sigma^2)]$$ 
que implica que el error $$\epsilon$$ es independiente de $$\mathbf x$$, 
lo cual se transforma en $$y = \mathbf w^T \mathbf x + \mathbb E[\epsilon],$$ donde $$\mathbb E[\epsilon]=0.$$
Por lo tanto $$y = \mathbf w^T \mathbf x.$$

La función de densidad de probabilidad de una Gausiana corresponde a

$$f(\alpha) = \frac{1}{\sigma \sqrt{2 \pi}} \exp{-\frac{1}{2} (\frac{\alpha -  \mu}{\sigma})^2},$$

donde $$\alpha$$, en el caso de regresión, corresponde 
a $$\mathbf w^T \mathbf x$$ (i.e., $$\alpha = \mathbf w^T \mathbf x$$).

Utilizando el método de verosimilitud el cual corresponde a maximizar 

$$\begin{eqnarray}
\mathcal L(\mathbf w, \sigma) &=& \prod_{(\mathbf x, y) \in \mathcal D} f(\mathbf w^T \mathbf x) \\
&=& \prod_{(\mathbf x, y) \in \mathcal D} \frac{1}{\sigma \sqrt{2\pi}} \exp{(-\frac{1}{2} (\frac{\mathbf w^T \mathbf x -  y}{\sigma})^2)} \\
\ell(\mathbf w, \sigma) &=& \sum_{(\mathbf x, y) \in \mathcal D}\log \frac{1}{\sigma \sqrt{2\pi}}  -\frac{1}{2} (\frac{\mathbf w^T \mathbf x -  y}{\sigma})^2 \\
&=& - \frac{1}{2\sigma^2}  \sum_{(\mathbf x, y) \in \mathcal D} (\mathbf w^T \mathbf x -  y)^2 - N \log \frac{1}{\sigma \sqrt{2\pi}}.
\end{eqnarray}
$$

El valor de cada parámetro se obtiene al calcular la derivada parcial con respecto al parámetro de 
interés, entonces se resuelven $$d$$ derivadas parciales para cada uno de los 
coeficientes $$\mathbf w$$. En este proceso se observar que el 
término $$N \log \frac{1}{\sigma \sqrt{2\pi}}$$ no depende de $$\mathbf w$$ entonces no afecta el 
máximo siendo una constante en el proceso de derivación y por lo tanto se desprecia. 
Lo mismo pasa para la constante $$\frac{1}{2\sigma^2}$$. Una vez obtenidos los 
parámetros $$\mathcal w$$ se obtiene el valor $$\sigma.$$ 

Una manera equivalente de plantear este problema es como un problema de algebra lineal, 
donde se tiene una matriz de observaciones $$X$$ que se construyen con las 
variables $$\mathbf x$$ de $$\mathcal X,$$ donde cada renglón de $$X$$ es una observación,
y el vector dependiente $$\mathbf y$$ donde cada elemento es la respuesta correspondiente
a la observación.

Viéndolo como un problema de algebra lineal lo que se tiene es 

$$ X \mathbf w = \mathbf y,$$

donde para identificar $$\mathbf w$$ se pueden realizar lo siguiente

$$ X^T X \mathbf w = X^T \mathbf y.$$

Despejando $$\mathbf w$$ se tiene

$$\mathbf w = (X^T X)^{-1} X^T \mathbf y.$$

Previamente se ha presentado el error estándar de cada parámetro que se ha estimado, 
en caso de la regresión el 
[error estándar](/AprendizajeComputacional/capitulos/14Estadistica/#sec:error-estandar-ols)
de $$\mathcal w_j$$ es $$\sigma \sqrt{(X^T X)^{-1}_{jj}}.$$

## Ejemplo: Diabetes
{: #sec:diabetes }

Esta sección ilustra el proceso de resolver un problema de regresión utilizando OLS.
El problema a resolver se obtiene mediante la función `load_diabetes` de la siguiente manera

```python
X, y = load_diabetes(return_X_y=True)
```

El siguiente paso es generar los conjuntos de 
[entrenamiento y prueba](AprendizajeComputacional/capitulos/03Parametricos/#sec:conjunto-entre-prueba)

```python
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
```

Con el conjunto de entrenamiento `T` y `y_t` se estiman los 
parámetros de la regresión lineal tal y como se muestra a continuación

```python
m = LinearRegression().fit(T, y_t)
```

Los coeficientes de la regresión lineal 
son $$\mathbf w=[11.4506, -270.97, 529.6703, 325.5005, -664.1954, 304.8051, 27.2395, 204.9071, 657.794, 98.7585 ]$$ 
y $$w_0=152.0305$$ lo cual se encuentran en las siguientes variables

```python
m.coef_
m.intercept_
```

La pregunta es si estos coeficientes son estadísticamente diferentes de cero, esto
se puede conocer calculando el error estándar de cada coeficiente. Para lo cual 
se requiere estimar $$\sigma$$ que corresponde a la desviación estándar del error
tal y como se muestra en las siguientes instrucciones.

```python
error = y_t - m.predict(T)
std_error = np.std(error)
```

El error estándar de $$\mathbf w$$ es 

```python
diag = np.arange(T.shape[1])
_ = np.sqrt((np.dot(T.T, T)**(-1))[diag, diag])
se = std_error * _
```

y para saber si los coeficientes son significativamente diferente de cero
se calcula el cociente `m.coef_` entre `se`; teniendo los siguientes
valores $$[0.1996, -4.7733, 9.4381, 5.8703, -11.9881, 5.5425, 0.4751, 3.6897, 11.4076, 1.8092].$$
Se observa que hay varios coeficientes con valor absoluto menor que 2, lo cual significa
que esas variables tiene un coeficiente que estadísticamente no es diferente de cero. 

La predicción del conjunto de prueba se puede realizar con la siguiente instrucción

```python
hy = m.predict(G)
```

Finalmente, la siguiente figura muestra las predicciones contra las mediciones reales.
También se incluye la linea que ilustra el modelo ideal. 


```python
sns.scatterplot(x=hy, y=y_g)
sns.lineplot(x=[hy.min(), hy.max()], y=[y_g.min(), y_g.max()])
```

<!--
plt.savefig('scatter_lineal_regresion.png', dpi=300)
-->

![Regresión Lineal](/AprendizajeComputacional/assets/images/scatter_lineal_regresion.png)


Complementando el ejemplo anterior, se realiza un modelo que primero elimina 
las variables que no son estadísticamente diferentes de cero (primera linea) y 
después crea nuevas variables al incluir el cuadrado, ver las líneas dos y tres
del siguiente código. 

```python
mask = np.fabs(m.coef_ / se) >= 2
T = np.concatenate((T[:, mask], T[:, mask]**2), axis=1)
G = np.concatenate((G[:, mask], G[:, mask]**2), axis=1)
```

Se observa que la identificación de los coeficientes $$\mathbf w$$ sigue siendo
lineal aun y cuando la representación ya no es lineal por incluir el cuadrado. 
Siguiendo los pasos descritos previamente, se inicializa el modelo y después se
realiza la predicción.

```python
m2 = LinearRegression().fit(T, y_t)
hy2 = m2.predict(G)
```

En este momento se compara si la diferencia entre el error cuadrático medio, del primer
y segundo modelo, la diferencia es $$37.5319$$ indicando que el primer modelo es mejor. 

```python
diff = ((y_g - hy2)**2).mean() -  ((y_g - hy)**2).mean()
```

Para comprobar si esta diferencia es significativa se calcula el error estándar, 
utilizando 
[bootstrap](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap)
tal y como se muestra a continuación. 

```python
S = np.random.randint(y_g.shape[0],
                      size=(500, y_g.shape[0]))
B = [((y_g[s] - hy2[s])**2).mean() -  ((y_g[s] - hy[s])**2).mean()
     for s in S]
se = np.std(B, axis=0)
```

Finalmente, se calcula el área bajo la curva a la izquierda del cero, teniendo
un valor de $$0.3666$$ lo cual indica que los dos modelos son similares. En este caso
se prefiere el modelo más simple porque se observar que incluir el cuadrado de las variables
no contribuye a generar un mejor model. El área bajo la curva se calcula con el siguiente 
código. 

```python
dist = norm(loc=diff, scale=se)
dist.cdf(0)
```
