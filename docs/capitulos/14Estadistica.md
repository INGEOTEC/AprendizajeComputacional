---
layout: default
title: Estadística (Apéndice)
nav_order: 15
---

# Estadística
{: .fs-10 .no_toc }

El **objectivo** de este apéndice es complementar la información de algunos procedimientos
estadísticos usados en el curso. 

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from scipy.stats import norm
from matplotlib import pylab as plt
import numpy as np
import seaborn as sns
sns.set_theme()
```

# Error estandar
{: #sec:error-estandar}

El **error estandar** está definido como $$\sqrt{\mathbb V(\hat \theta)}$$ donde 
$$\hat \theta$$ es el valor estimado. No para todas las estadísticas es posible
tener una ecuación analítica para calcular $$\sqrt{\mathbb V(\hat \theta)}$$ y 
en los casos donde el valor analítico no se puede calcular se puede usar 
la técnica de Bootstrap. 

## Media
{: #sec:error-estandar-media}

Una de las estadísticas donde si se puede calcular analíticamente 
$$\sqrt{\mathbb V(\hat \theta)}$$ es la media, es decir, 
se tiene una muestra $$\mathcal D$$ con $$N$$ elementos independientes y identicamente 
distribuidos, entonces la media corresponde a

$$\hat \mu = \frac{1}{N} \sum_{x \in \mathcal D} x.$$

El error estandar de $$\hat \mu$$ es $$\sqrt{\mathbb V(\hat \mu)}$$. Para derivar
el valor análitico de este error estandar es necesario utilizar la siguiente 
propiedad de la varianza:

$$\mathbb V(\sum_i a_i \mathcal X_i) = \sum_i a_i^2 \mathbb V(\mathcal X_i),$$

donde $$a_i$$ representa una constante y las variables aleatorias $$\mathcal X$$ son
independientes. En estas condiciones se observa que para el caso del error estandar de
la media la constante es $$\frac{1}{N}$$ y las variables son independientes de acuerdo a la forma
que se construyó $$\mathcal D$$ entonces

$$\begin{eqnarray}
\sqrt{\mathbb V(\hat \mu)} &=& \sqrt{\mathbb V(\frac{1}{N} \sum_{x \in \mathcal D} x)} \\
&=& \sqrt{ \sum_{x \in \mathcal D} \frac{1}{N^2}  \mathbb V( x)}\\
&=& \sqrt{ \frac{1}{N^2} \sum_{x \in \mathcal D} \sigma^2} \\
&=& \sqrt{ \frac{N}{N^2} \sigma^2} \\
&=& \sqrt{ \frac{\sigma^2}{N}}, 
\end{eqnarray}$$

donde $$\sigma^2$$ es la varianza de la distribución usada para generar $$\mathcal D$$. 

## Ejemplo

El siguiente ejemplo complementa la información al presentar el
error estandar de la media cuando los datos vienen de una distribución
Gausiana. Suponiendo que se tiene $$1000$$ muestras de una
distribución Gausiana $$\mathcal N(1, 4),$$ i.e., $$\mu=1$$ y $$\sigma=2$$.
La error estandar de estimar la media con esos datos está dado por
$$\mathbb V(\hat \mu) = \sqrt{\frac{\sigma^2}{N}} = \sqrt{\frac{4}{1000}}=0.0632.$$ 

Continuado con el ejemplo, se simula la generación de esta población
de $$1000$$ elementos. El primer paso es iniciar la clase `norm` 
(que implementa una distribución Gausiana) para que se simule 
$$\mathcal N(1, 4).$$ Es importante notar que el parámetro `scale`
de `norm` corresponde a la desviación estandar $$\sigma.$$

```python
p1 = norm(loc=1, scale=2)
```

Usando `p1` se simulan 500 poblaciones de 1000 elementos cada una,
y para cada una de esas poblaciones se calcula su media. La primera
linea crea la muestra $$\mathcal D$$ y a continuación se calcula
la media por cada población, renglon de `D`.

```python
D = p1.rvs(size=(500, 1000))
mu = [x.mean() for x in D]
```

El error estandar es la desvicación estandar de `mu`,
el cual se puede calcular con la siguiente instrucción. `se` tiene 
un valor de $$0.0637$$, que es similar al obtenido mediante
$$\mathbb V(\hat \mu).$$

```python
se = np.std(mu)
```

Para complementar la información se presenta el histograma 
de `mu` donde se puede observar la distribución de estimar 
la media de una población. 

```python
sns.histplot(mu)
```

<!--
plt.savefig('normal_mean.png', dpi=300)
-->

<!--
![Histograma del error](/AprendizajeComputacional/assets/images/normal_mean.png)

# Bootstrap
{: #sec:bootstrap }

```python
D = X[0]
```

```python
S = np.random.randint(D.shape[0], size=(500, D.shape[0]))
B = [(D[s]).mean() for s in S]
se = np.std(B)
```

$$0.0623$$

-->