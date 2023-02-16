---
layout: default
title: Conjunto de Datos (Apéndice)
nav_order: 17
---

# Conjunto de Datos
{: .fs-10 .no_toc }

El **objetivo** de este apéndice es listar los conjuntos de datos utilizados en el curso.  

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from scipy.stats import multivariate_normal
import numpy as np
```

# Problema Sintético Mezcla de Clases

```python
p1 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]])
X_1 = p1.rvs(size=1000)
p2 = multivariate_normal(mean=[1.5, -1.5], cov=[[2, 1], [1, 3]])
X_2 = p2.rvs(size=1000)
p3 = multivariate_normal(mean=[12.5, -3.5], cov=[[2, 3], [3, 7]])
X_3 = p3.rvs(size=1000)
```

La siguiente figura muestra estas tres distribuciones. 

![Tres clases generadas por tres distribuciones gausianas multivariadas](/AprendizajeComputacional/assets/images/gaussian_3classes.png)

# Problema Sintético 3 Clases Separadas

```python
X_1 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]]).rvs(1000)
X_2 = multivariate_normal(mean=[-5, -10], cov=[[2, 1], [1, 3]]).rvs(1000)
X_3 = multivariate_normal(mean=[15, -6], cov=[[2, 3], [3, 7]]).rvs(1000)
```

Este problema se muestra en la siguiente figura. 

![Tres Distribuciones Gausianas](/AprendizajeComputacional/assets/images/clases3-arboles.png)

# Breast Cancer Wisconsin

El conjunto de datos de Breast Cancer Wisconsin es un problema de clasificación
que se obtiene con el siguiente código. 

```python
X, y = load_breast_cancer(return_X_y=True)
```

# Iris

Un conjunto clásico en problemas de clasificación es el problema del 
Iris que se encuentra con las siguientes instrucciones.

```python
X, y = load_iris(return_X_y=True)
```

# Números

El conjunto de Digits es un conjunto de clasificación donde 
se trata de identificar el número escrito en una imagen; este conjunto
de datos se descarga utilizando las siguientes instrucciones. 

```python
X, y = load_digits(return_X_y=True)
```

# Diabetes

El conjunto de datos Diabetes es un problema de regresión que se puede
recuperar usando el siguiente código. 

```python
X, y = load_diabetes(return_X_y=True)
```