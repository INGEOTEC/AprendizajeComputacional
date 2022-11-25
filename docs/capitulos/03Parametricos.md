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
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
```

---

# Introducción

Existen diferentes tipos de algoritmos que se puede utilizar para resolver problemas de
aprendizaje supervisado y no supervisado. En particular, esta unidad se enfoca en presentar 
las técnicas que se pueden caracterizar como métodos paramétricos. 

Los métodos paramétricos se identifican por asumir que los datos provienen de una distribución 
de la cual se desconocen los parámetros y el procedimiento es encontrar aquellos parámetros 
de la distribución que mejor modelen los datos. Una vez obtenidos los parámetros 
se cuenta con todos los elementos para utilizar el modelo y predecir la característica para
la cual fue entrenada. 

# Metodología

Hasta el momento se han presentado ejemplos de los pasos 4 y 5 
de la [metodología general](/AprendizajeComputacional/capitulos/01Tipos/#sec:metodologia-general);
esto fue en la sección de 
[Predicción](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:prediccion-normal) 
y la sección de 
[Error de Clasificación](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:error-clasificacion). 
Esta sección complementa los ejemplos anteriores al utilizar todos pasos 
de la 
[metodología general de aprendizaje supervisado.](/AprendizajeComputacional/capitulos/01Tipos/#sec:metodologia-general)
En partícular se enfoca al paso 3 que corresponde al diseño del algoritmo $$f$$ 
que modela el fenómeno de interés utilizando los datos $$\mathcal T \subset \mathcal D.$$

El algoritmo $$f$$ corresponde a asumir que los datos $$\mathcal D$$ provienen de 
una distribución $$F$$ la cual tiene una serie de parámetros $$\theta$$ que 
son identificados con $$\mathcal T.$$


# Estimación de Parámetros

Se inicia la descripción de métodos parámetricos presentando el procedimiento general
para estimar los parámetros de una distribución. Se cuenta con un conjunto $$\mathcal D$$
donde los elementos $$x \in \mathcal D$$ son $$x \in \mathbb R^d$$. Los elementos $$x \in \mathcal D$$ tienen un distribución $$F$$, i.e., $$x \sim F$$, son independientes y $$F$$ está 
definida por la función de densidad de probabilidad $$f_{\theta}$$ que está definida
por $$\theta$$ parámetros. Utilizando $$\mathcal D$$ el objetivo es identificar
los parámetros $$\theta$$ que harían observar a $$\mathcal D$$ lo más probable. 

## Verosimilitud

Una manera de plantear lo anterior es maximizando la verosimilitud. La versosimilitud es 
la función distribución conjunta de los elementos en $$\mathcal D$$,
i.e., $$f_\theta(x_1, x_2, \ldots, x_N).$$ Considerando que la muestras
son independientes entonces 
$$f_\theta(x_1, x_2, \ldots, x_N) = \prod_{x \in \mathcal D} f_\theta (x).$$
La función de verosimilitud considera la ecuación anterior como una función 
de los parámetros $$\theta,$$ es decir,

$$\mathcal L(\theta) = \prod_{x \in \mathcal D} f_\theta (x),$$

siendo el logaritmo de la verosimilitud 

$$\ell(\theta) = \log \mathcal L(\theta) = \sum_{x \in \mathcal D} \log f_\theta (x).$$

## Distribucción de Bernoulli

La verosimilitud se ejemplifica con la identificación del parámetro $$p$$ de una distribución 
Bernoulli. Una distribución Bernoulli modela dos estados, por un lado se tiene la clase
negativa identificada por $$0$$; identificando la clase positiva como $$1$$. Entonces, la 
probabilidad de ver $$1$$ es $$\mathbb P(X=1) = p$$ y $$\mathbb P(X=0) = 1 - p$$.
Estas ecuaciones se pueden combinar para definir $$f_\theta(x) = p^x (1 - p)^{1-x}.$$

Utilizando el logaritmo de la verosimilitud se tiene:

$$\ell(p) = \sum_{i=1}^N \log p^{x_i} (1 - p)^{1-x_i} = \sum_{i=1}^N x_i \log p + (1-x_i) \log (1 - p)$$.

Recordando que el máximo de $$\ell(\mathcal p) $$ se obtiene cuando $$\frac{d}{dp} \ell(\mathcal p) = 0$$. En estas condiciones estimar $$p$$ quedaría como:

$$ \begin{eqnarray} \frac{d}{dp} \ell(\mathcal p) &=& 0 \\ \frac{d}{dp} [ \sum_{i=1}^N x_i \log p + (1-x_i) \log (1 - p)] &=& 0 \\ \frac{d}{d p} [ \sum_{i=1}^N x_i \log p + \log (1 - p) (N - \sum_{i=1}^N x_i) ] &=& 0\\ \sum_{i=1}^N x_i \frac{d}{d p} \log \mathcal p + (N - \sum_{i=1}^N x_i) \frac{d}{d p} \log (1 - \mathcal p) &=& 0\\ \sum_{i=1}^N x_i \frac{1}{p} + (N - \sum_{i=1}^N x_i) \frac{-1}{(1 - p)} &=& 0\\ \end{eqnarray}$$

Realizando algunas operaciones algebraicas se obtiene:

$$\hat p = \frac{1}{N}\sum_{i=1}^N x_i $$.

## Ejemplo: Distribución Gausiana
{: #sec:estimacion-distribucion-gausiana }

Esta sección sigue un camino práctico, donde se presenta el código para estimar
los parámetros de una distribución Gausiana donde se conocen todos los parámetros, 
la distribución se usa para generar 1000 muestras y después de esas muestras 
se estiman los parámetros; de estas manera se tienen todos los elementos para
comparar los parametros reales $$\theta$$ de los parámetros estimados $$\hat \theta.$$

La distribución que se usará se utlizo en para generar un 
[problema sintético](/AprendizajeComputacional/capitulos/03Parametrics/#sec:tres-normales)
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

donde el eje donde se realiza la operación es el primero que corresponde 
al índice $$0.$$ La media estimada es: $$\hat \mu = [4.9334, 5.0413]^T$$ con una 
[error estándar](/AprendizajeComputacional/capitulos/14Estadistica/#sec:error-estandar-media) (`se`)
de $$[0.0648, 0.0436]^T$$ que se calcula con el siguiente código. 

```python
se = np.std(D, axis=0) / np.sqrt(1000)
```

Hasta el momento se ha estimado $$\mu$$, falta por estimar $$\Sigma$$, lo cual es 
puede realizar con la siguiente instrucción

```python
cov = np.cov(D, rowvar=False)
```

donde el parámetro `rowvar` indica la forma en que están proporcionados los datos. La estimación
da los siguientes 
valores $$\hat \Sigma = \begin{pmatrix}  4.2076 & -0.0694 \\ -0.0694 & 1.9044 \\ \end{pmatrix}$$
se puede observar que son similares al parámetro con que se simularon los datos. 

Siguiendo con la inercia de presentar el error estándar de cada estimación, en las siguientes
instrucciones se presenta el error estándar de $$\hat \Sigma$$, el cual se calcula utilizando
la técnica de [Bootstrap](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap)
utilizando el siguiente código. Se puede observar que la función `np.cov` se ejecuta
utilizando la muestra indicada en la variable `s`. El error estándar (`se`) de $$\hat \Sigma$$ 
corresponde a $$\begin{pmatrix} 0.1845 & 0.0869 \\ 0.0869 & 0.0875 \\ \end{pmatrix}.$$ Se
puede observar que los elementos fuera de la diagonal tienen un error estándar que 
superior al número y que el cero se encuentra en el intervalo $$\hat \Sigma \pm se$$ lo cual 
indica que el cero es un valor factible; dado que se conoce $$\Sigma$$ se puede verificar
que el parámetro real es $$0$$ para aquellos elementos fuera de la diagonal. 


```python
S = np.random.randint(D.shape[0],
                      size=(500, D.shape[0]))
B = [np.cov(D[s], rowvar=False) for s in S]
se = np.std(B, axis=0)
```

# Metodología de Clasificación

Habiendo descrito el proceso para estimar los parámetros de una distribución,
por un lado se presento de manera teórica con la distribución
[Bernoulli](/AprendizajeComputacional/capitulos/03Parametricos/#distribucción-de-bernoulli)
y de manera práctica con una distribución
[Gausiana](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-distribucion-gausiana).
Se está en la posición de usar todos estos elementos para presentar
el proceso completo de clasificación. La [metodología general de aprendizaje supervisado.](/AprendizajeComputacional/capitulos/01Tipos/#sec:metodologia-general) está definida
 por cinco pasos, estos pasos se especializan para el problema 
 de clasificación y regresión, utilizando modelos paramétricos, de la siguiente manera. 

 1. Todo empieza con un conjunto de datos $$\mathcal D$$ que tiene la información del fenómeno de interés.
2. Se selecciona el conjunto de ([entrenamiento](/AprendizajeComputacional/capitulos/03Parametricos/#sec:conjunto-entre-prueaba)) $$\mathcal T \subset \mathcal D.$$ 
3. Se diseña un algoritmo, $$f$$, el cual se basa en un [modelo](/AprendizajeComputacional/capitulos/03Parametricos/#sec:model-clasificacion) y la [estimación de sus parámetros](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-parametros) utilizando $$\mathcal T.$$
4. Se utiliza $$f$$ para [predecir.](/AprendizajeComputacional/capitulos/03Parametricos/#sec:prediccion)
5. Se mide el [rendimiento](/AprendizajeComputacional/capitulos/03Parametricos/#sec:rendimiento) utilizando un conjunto de [prueba.](/AprendizajeComputacional/capitulos/03Parametricos/#sec:conjunto-entre-prueaba)


La metodología de clasificación se ilustra utilizando el 
[problema sintético](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
de tres clases que se presentó en la unidad de 
[pasada.](/AprendizajeComputacional/capitulos/02Teoria_Decision)
Especificamente las entradas que definian a cada clase estaban en la variables
`X_1`, `X_2` y `X_3`. Entonces las clases se pueden colocar 
en la variable `y` tal como se indica a continuación. 

```python
X = np.concatenate((X_1, X_2, X_3))
y = np.concatenate([np.ones(1000),
                    np.ones(1000) + 1,
                    np.ones(1000) + 2])
```

Las variables `X` y `y` contiene la información que conforma el conjunto
$$\mathcal D = (\mathcal X, \mathcal Y)$$ donde cada renglón de `X`
es una realización de la variable aleatoria $$\mathcal X$$ y equivalentemente
cada elemento en `y` es una realización de $$\mathcal Y.$$

## Conjunto de Entrenamiento y Prueba
{: #sec:conjunto-entre-prueaba }

[Anteriormente](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-distribucion-gausiana) se había utilizado a $$\mathcal D$$ en el procedimiento
de maximizar la verosimilitud, esto porque el objetivo en ese procedimiento
es estimar los parámetros de la distribución. Pero el objetivo en 
aprendizaje supervisado es diseñar un algoritmo (función en este caso)
que modele la relación entre $$\mathcal X$$ y $$\mathcal Y$$. 
Para conocer esto es necesario medir el rendimiento del algoritmo
en instancias que no han sido vistas en el entrenamiento.

En el caso de esta unidad, **entrenar** se refiere a la estimación 
de los parámetros del modelo. En consecuencia, se requieren contar 
con datos para medir el rendimiento, a este conjunto de datos se le 
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
[Teorema de Bayes](/AprendizajeComputacional/capitulos/02Teoria_Decision/#teorema-de-bayes) $$\mathbb P(\mathcal Y \mid \mathcal X) = \frac{ \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)}{\mathbb P(\mathcal X)}$$ donde se usa la verosilimitud $$\mathbb P(\mathcal X \mid \mathcal Y)$$ y
el prior $$\mathbb P(\mathcal Y)$$ para definir la probabilidad a posteriori $$\mathbb P(\mathcal Y \mid \mathcal X)$$. En métodos paramétricos se asume que se puede modelar la verosimilitud
con una distribución particular, que por lo generar es una distribución Gausiana multivariada. 
Es decir, la variable aleatoria $$\mathcal X$$ dado $$\mathcal Y$$ ($$\mathcal X_{\mid \mathcal Y}$$) es $$\mathcal X_{\mid \mathcal Y} \sim \mathcal N(\mu_{\mathcal Y}, \Sigma_{\mathcal Y}),$$ donde
se observar que los parámetros de la distribución Gausina dependen de la variable 
aleatoría $$\mathcal Y$$ y estos pueden ser identificados 
cuando $$\mathcal Y$$ tiene un valor específico. 

## Estimación de Parámetros
{: #sec:estimacion-parametros }

Dado que por [definición del problema](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales) se conoce que la verosimilitud para cada clase proviene de una 
Gausiana, i.e., $$\mathcal X_{\mid \mathcal Y} \sim \mathcal N(\mu_{\mathcal Y}, \Sigma_{\mathcal Y}),$$ en esta sección se estimarán los parámetros utilizando este conocimiento. 

El primer paso en la estimación de parámetros es estimar el prior $$\mathbb P(\mathcal Y) $$,
el cual corresponde a clasificar el evento sin ver el valor de la 
entrada $$\mathcal X.$$ Esto se puede modelar
mediante una distribución Categorica con parámetros $$p_i$$ donde $$\sum_i^K p_i = 1$$. 
Estos parámetros se pueden estimar utilizando la función `np.unique` de la siguiente manera


```python
labels, counts = np.unique(y_t, return_counts=True)
prior = counts / counts.sum()
```
La variable `prior` contiene en el primer elemento $$\mathbb P(\mathcal Y=1) = 0.3292$$
, en el segundo $$\mathbb P(\mathcal Y=2) = 0.3412$$ y 
en el tercero $$\mathbb P(\mathcal Y=3) = 0.3296$$ que es
aproximadamente $$\frac{1}{3}$$ el cual es el valor real del prior. 

<!-- $$[0.3292, 0.3412, 0.3296]$$ -->

Siguiendo los pasos en [estimación de parámetros de una Gausiana](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-distribucion-gausiana) se pueden estimar los parámetros
para cada gausiana dada la clase. Es decir, se tiene que estimar los 
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
y $$\hat \mu_3 = [12.5064, -3.4501]^T$$. Para las covarianzas, los valores estimados
corresponden 
a $$\hat \Sigma_1 = \begin{pmatrix} 4.0896 & 0.0409 \\ 0.0409 & 1.9562 \\ \end{pmatrix},$$ $$\hat \Sigma_2 = \begin{pmatrix} 1.9533 & 1.0034 \\ 1.0034 & 2.8304 \\ \end{pmatrix}$$ 
y  $$\hat \Sigma_3 = \begin{pmatrix} 2.0451 & 3.0328 \\ 3.0328 & 6.8235 \\ \end{pmatrix}.$$
Estas estimaciones se pueden comparar con los 
[parámetros reales.](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
También se puede calcular su error estándar para identificar si el parámetro real, $$\theta$$, 
se encuentra en el intervalor definido 
por $$\hat \theta - 2\hat{se} \leq \hat \theta \leq \hat \theta + 2 \hat{se}$$ 
que corresponde aproximadamente al 95% de confianza del intervalo asumiendo que la distribución
de la estimación del parámetro es Gausiana. 

## Predicción
{: #sec:prediccion }

[predicción](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:prediccion-normal)

```python
def predict_prob(X, likelihood, prior):
    likelihood = [m.pdf(X) for m in likelihood]
    posterior = np.vstack(likelihood).T * prior
    evidence = posterior.sum(axis=1)
    return posterior / np.atleast_2d(evidence).T
```

```python
def predict(X, likelihood, prior, labels):
    _ = predict_prob(X, likelihood, prior)
    return labels[np.argmax(_, axis=1)]
```

## Rendimiento
{: #sec:rendimiento }

[error de clasificación](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:error-clasificacion)

```python
hy = predict(G, likelihood, prior, labels)

```

`error` es $$0.01$$ 


```python
error = (y_g != hy).mean()
```


```python
se_formula = np.sqrt(error * (1 - error) / y_g.shape[0])
```

`se_formula` es $$0.0041$$ 

# Clasificador Bayesiano Ingenuo

## Modelo

[Anteriormente](/AprendizajeComputacional/capitulos/03Parametricos/#sec:model-clasificacion)
se asumió que la variable aleatoria $$\mathcal X = (\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_d)$$
dado $$\mathcal Y$$ ($$\mathcal X_{\mid \mathcal Y}$$) 
es $$\mathcal X_{\mid \mathcal Y} \sim \mathcal N(\mu_{\mathcal Y}, \Sigma_{\mathcal Y}),$$ 
donde $$\mu_{\mathcal Y} \in \mathbb R^d$$, $$\Sigma_{\mathcal Y} \in \mathbb R^{d \times d}.$$ 
y $$f(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_d)$$ 
es la función de densidad de probabilidad conjunta.

En el clasificador Bayesiano Ingenuo se asume que las variables $$\mathcal X_i$$ 
y $$\mathcal X_j$$ para $$i \neq j$$ son independientes, esto trae como consecuencia
que $$f(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_d) = \prod_i^d f(\mathcal X_i).$$



extendiendo la función de densidad de probabilidad para una multiples variables se 
tiene 

## Estimación de Parámetros

```python
likelihood = []
for k in labels:
    mask = y_t == k
    mu = np.mean(T[mask], axis=0)
    var = np.var(T[mask], axis=0, ddof=1)
    likelihood_k = multivariate_normal(mean=mu, cov=var)
    likelihood.append(likelihood_k)
```

$$\hat \mu_1 = [5.1234, 5.0177]^T$$, $$\hat \mu_2 = [1.5229, -1.5553]^T$$ y
$$\hat \mu_3 = [12.5064, -3.4501]^T$$


$$\hat \Sigma_1 = \begin{pmatrix} 4.0896 & 0.0 \\ 0.0 & 1.9562 \\ 
\end{pmatrix}$$, 
$$\hat \Sigma_2 = \begin{pmatrix} 1.9533 & 0.0 \\ 0.0 & 2.8304 \\ 
\end{pmatrix}$$ y 
$$\hat \Sigma_3 = \begin{pmatrix} 2.0451 & 0.0 \\ 0.0 & 6.8235 \\ 
\end{pmatrix}$$

## Predicción

```python
hy_ingenuo = predict(G, likelihood, prior, labels)
```

## Rendimiento

con un `error` de $$0.0133$$ y un error estándar (`se_formula`)
de $$0.0047$$

```python
error = (y_g != hy_ingenuo).mean()
se_formula = np.sqrt(error * (1 - error) / y_g.shape[0])
```

# Clase: Clasificador Bayesiano Gausiano

```python
class GaussianBayes(object):
    def __init__(self, naive=False) -> None:
        self._naive = naive

    @property
    def naive(self):
        return self._naive
    
    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels        
```

## Estimación de Parámetros

```python
    def fit(self, X, y):
        self.prior = y
        self.likelihood = (X, y)
        return self
```

```python
    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, y):
        labels, counts = np.unique(y, return_counts=True)
        prior = counts / counts.sum()        
        self.labels = labels
        self._prior = np.log(prior)
```


```python
    @property
    def likelihood(self):
        return self._likelihood

    @likelihood.setter
    def likelihood(self, D):
        X, y = D
        likelihood = []
        for k in self.labels:
            mask = y == k
            mu = np.mean(X[mask], axis=0)
            if self.naive:
                cov = np.var(X[mask], axis=0, ddof=1)
            else:
                cov = np.cov(X[mask], rowvar=False)
            _ = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
            likelihood.append(_)
        self._likelihood = likelihood
```

## Predicción

```python
    def predict(self, X):
        hy = self.predict_log_proba(X)
        _ = np.argmax(hy, axis=1)
        return self.labels[_]
```

```python
    def predict_proba(self, X):
        _ = self.predict_log_proba(X)
        return np.exp(_)
```

```python
    def predict_log_proba(self, X):
        log_likelihood = np.vstack([m.logpdf(X) 
                                    for m in self.likelihood]).T
        prior = self.prior
        posterior = log_likelihood + prior
        evidence = np.atleast_2d(logsumexp(posterior, axis=1)).T
        return posterior - evidence
```

## Uso

```python
bayes = GaussianBayes().fit(T, y_t)
hy = bayes.predict(G)
```

# Ejemplo: Breast Cancer Wisconsin


```python
X, y = datasets.load_breast_cancer(return_X_y=True)
```


[entrenamiento y prueba](/AprendizajeComputacional/capitulos/03Parametricos/#conjunto-de-entrenamiento-y-prueba)

```python
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
```

## Entrenamiento

```python
gaussian = GaussianBayes().fit(X, y)
naive = GaussianBayes(naive=True).fit(T, y_t)
```

## Predicción

```python
hy_gaussian = gaussian.predict(G)
hy_naive = naive.predict(G)
```

## Rendimiento

```python
error_gaussian = (y_g != hy_gaussian).mean()
error_naive = (y_g != hy_naive).mean()
```

clasificador gausiano $$0.0351$$ 
ingenuo $$0.0614$$


# Diferencias en Rendimiento 

diferencia $$0.0263$$ 

```python
diff = (y_g != hy_naive).mean() -  (y_g != hy_gaussian).mean()
```

con un error estándar de $$0.0226$$
cociente entre la diferencia y el error estándar es de 
$$1.1650$$

[Bootstrap](/AprendizajeComputacional/capitulos/14Estadistica/#sec:bootstrap)


```python
S = np.random.randint(y_g.shape[0],
                      size=(500, y_g.shape[0]))
B = [(y_g[s] != hy_naive[s]).mean() -  (y_g[s] != hy_gaussian[s]).mean()
     for s in S]
se = np.std(B, axis=0)
```

```python
sns.displot(B, kde=True)
```

<!--
plt.savefig('comp_bayes_breast_cancer.png', dpi=300)
-->

![Diferencia entre Clasificadores Bayesianos](/AprendizajeComputacional/assets/images/comp_bayes_breast_cancer.png)


El área bajo la curva a la izquierda del cero es $$0.1220$$ 
la cual se puede calcular con el siguiente código

```python
dist = norm(loc=diff, scale=se)
dist.cdf(0)
```

que usa la función cumulativa de densidad para calcular la probabilidad.





# Regresión

Hasta este momento se han revisado métodos paramétricos en clasificación, ahora es el turno de abordar 
esto en el problema de regresión.

Recordando, regresión es un problema de aprendizaje supervisado, es decir se cuenta con un conjunto de 
entrenamiento, $$\mathcal X = \{ (x_1, y_1), \ldots, (x_N, y_N )\}$$, de pares entrada y salida; la 
salida es $$ y_i \in \mathbb R$$.

Entonces se busca una función con la forma $$ f: \mathbb{ R^d } \rightarrow \mathbb R $$ y que se 
comporte como: $$ \forall_{(x, y) \in \mathcal X} f(x) = y  + \epsilon $$. 

Este problema se puede plantear como un problema de optimización o como un problema de algebra lineal. 
Viéndolo como un problema de algebra lineal lo que se tiene es 

$$ X w = y $$

donde $$ X $$ son las observaciones, entradas o combinación de entradas, $$ w $$, son los pesos 
asociados y $$y$$ es el vector que contiene las variables dependientes. 

Tanto $$X$$ como $$y$$ son datos que se obtienen de $$\mathcal X$$ entonces lo que hace falta 
identificar es $$w$$, lo cual se puede realizar de la siguiente manera

$$ X^T X w = X^T y $$

donde $$X^T$$ es la transpuesta de $$X$$. Despejando $$w$$ se tiene

$$w = (X^T X)^{-1} X^T y.$$

