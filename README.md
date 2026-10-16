# 🦖 Isla de Dinosaurios IA
Laboratorio de Procesamiento de Lenguaje Natural.

La aplicación genera una especie ficticia mediante el flujo:

**GRU → Nombre → Ollama → Identidad → Imagen → Chat**

## Funcionalidades

- Generación de nombres de dinosaurios con una GRU entrenada a nivel de caracteres.
- Verificación para evitar nombres idénticos a los del conjunto de entrenamiento.
- Generación de identidad mediante Ollama y Llama 3.2 3B.
- Generación de imagen con SD-Turbo.
- Chat con el dinosaurio conservando su identidad.
- Botón **Nuevo Dinosaurio** para generar una nueva especie sin reentrenar la GRU.
- Visualización del nombre, descripción, imagen y configuración básica del modelo.

## Modelo de nombres

Modelo final seleccionado:

- Arquitectura: GRU
- Capas recurrentes: 2
- Embedding: 64
- Estado oculto: 256
- Épocas: 15
- Optimizador: Adam
- Learning rate: 0.001
- Función de pérdida: CrossEntropyLoss ignorando `<PAD>`
- Muestreo final: Top-p = 0.9

Se compararon previamente una RNN y una GRU.

## Dataset

Se utilizó el conjunto de nombres de dinosaurios del ejercicio **Dinosaurus Island** de DeepLearning.AI, convertido a formato CSV.

El conjunto contiene 1536 nombres.

## Ollama

Modelo utilizado:

`llama3.2:3b`

Ollama se ejecuta mediante Docker.

Para comprobar los modelos disponibles:

```bash
docker exec ollama ollama list