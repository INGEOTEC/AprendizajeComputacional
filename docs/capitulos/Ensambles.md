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

---

# Introducción

Una manera de reducir la varianza de un clasificador o regresor es mediante 
el uso de ensambles. Los ensambles combinan algoritmos de aprendizaje que 
presentan algunas diferencias, es decir, estos pueden ser entrenados con 
diferentes particiones del conjunto de entrenamiento, o tienen diferentes 
parámetros o son diferentes algoritmos. 

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