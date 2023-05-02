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

La siguiente figura muestra como cambia el accuracy, cuando el número de clasificadores se incrementa, cada uno de esos clasificadores son independientes y tiene un accuracy de $$p=0.51,$$ se puede observar que cuando se 
tienen $$10,000$$ clasificadores independientes, se tiene un accuracy de $$0.977.$$

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


$$\hat y_i = \mathbb E[\mathcal N(\mathbf w \cdot \mathbf x_i + \epsilon, \sigma^2)]$$
$$\mathbb V() = \sum_i^N (y_i - \hat y_i)^2$$

# Bagging

Bagging es un método sencillo para la creación de un ensamble, la idea 
original es hacer un muestreo con repetición y así generar un nuevo conjunto 
de entrenamiento. 

Se ha probado que en promedio bagging es equivalente al seleccionar el 50% de 
los datos sin reemplazo y entrenar con este subconjunto, en el siguiente 
video vemos los efectos de bagging utilizando una máquina de soporte 
vectorial lineal.

{%include bagging.html %}

Bagging es una técnica que es muy utilizada en Árboles de Decisión, estos
algoritmos se les conoce como bosques, dado que son un conjunto de árboles. 
En particular Random Forest utiliza bagging y además una selección aleatoria
de las entradas para realizar los árboles que componen al ensamble. 

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