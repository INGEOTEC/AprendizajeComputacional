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
