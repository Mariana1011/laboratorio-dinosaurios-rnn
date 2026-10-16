# Resultados experimentales

## Configuración 1: RNN

- Arquitectura: RNN
- Embedding: 32
- Estado oculto: 128
- Capas recurrentes: 1
- Épocas: 30
- Optimizador: Adam
- Learning rate: 0.001
- Función de pérdida: CrossEntropyLoss ignorando <PAD>
- Pérdida final de validación: aproximadamente 1.60

La pérdida de entrenamiento y validación disminuyó de forma estable.
Las dos curvas se mantuvieron relativamente cercanas, por lo que no se observó un sobreajuste fuerte.

## Configuración 2: GRU

- Arquitectura: GRU
- Embedding: 64
- Estado oculto: 256
- Capas recurrentes: 2
- Épocas experimentales: 30
- Optimizador: Adam
- Learning rate: 0.001
- Función de pérdida: CrossEntropyLoss ignorando <PAD>

La GRU alcanzó una pérdida de validación menor que la RNN, cercana a 1.50.
Sin embargo, después de aproximadamente 15 épocas la pérdida de entrenamiento continuó disminuyendo mientras la pérdida de validación comenzó a aumentar.

Esto indicó sobreajuste.

Por esta razón se seleccionó como modelo final una GRU entrenada durante 15 épocas.

## Configuración final seleccionada

- Modelo: GRU
- Embedding: 64
- Estado oculto: 256
- Capas: 2
- Épocas: 15
- Optimizador: Adam
- Learning rate: 0.001
- Muestreo final: Top-p = 0.9

La pérdida final de validación fue aproximadamente 1.51.

## Estrategias de muestreo probadas

Se probaron:

- Temperatura = 0.7
- Temperatura = 1.0
- Temperatura = 1.3
- Top-k = 5
- Top-p = 0.9

### Análisis

La temperatura 0.7 produjo nombres más conservadores y cercanos a los patrones aprendidos.

La temperatura 1.0 produjo mayor variedad manteniendo nombres razonablemente coherentes.

La temperatura 1.3 aumentó la creatividad, pero produjo nombres menos previsibles.

Top-k restringió la generación a las opciones más probables y produjo nombres coherentes.

Top-p = 0.9 ofreció un buen equilibrio entre variedad y coherencia, por lo que fue seleccionado para la aplicación final.

## Resultado

Se generaron diez nombres que no coinciden exactamente con los nombres del conjunto de entrenamiento.

Para la primera demostración se seleccionó:

Dadosaurus

La aplicación final también puede generar nuevas especies automáticamente mediante el botón "Nuevo Dinosaurio", reutilizando la GRU entrenada sin volver a entrenarla.
