# 🦖 Isla de Dinosaurios IA

Laboratorio de Procesamiento de Lenguaje Natural.

La aplicación genera una especie ficticia mediante el flujo:

**GRU → Nombre → Ollama → Identidad → Imagen → Chat**

## Funcionalidades

- Generación de nombres con una GRU a nivel de caracteres.
- Verificación para evitar nombres idénticos al conjunto de entrenamiento.
- Generación de identidad con Ollama y Llama 3.2 3B.
- Generación de imagen con SD-Turbo.
- Chat conservando la identidad del dinosaurio.
- Botón **Nuevo Dinosaurio** sin reentrenar la GRU.
- Visualización de nombre, descripción, imagen y datos básicos del modelo.

## Modelo de nombres

Modelo final:

- Arquitectura: GRU
- Capas recurrentes: 2
- Embedding: 64
- Estado oculto: 256
- Épocas: 15
- Optimizador: Adam
- Learning rate: 0.001
- Pérdida: CrossEntropyLoss ignorando `<PAD>`
- Muestreo final: Top-p = 0.9

Se comparó previamente una RNN con una GRU.

## Dataset

Se utilizó el conjunto de nombres de dinosaurios del ejercicio **Dinosaurus Island** de DeepLearning.AI, convertido a formato CSV.

El conjunto contiene 1536 nombres.

## Ollama

Modelo utilizado:

`llama3.2:3b`

Ollama se ejecuta mediante Docker.

Comprobar modelos:

`docker exec ollama ollama list`

Si el modelo no está descargado:

`docker exec ollama ollama pull llama3.2:3b`

## Aplicación web

Ejecutar:

`cd /home/ec2-user/SageMaker`

`python app.py`

La aplicación queda disponible localmente en:

`http://localhost:7860`

## Acceso público

Se utiliza ngrok para exponer la aplicación de Gradio por el puerto 7860.

Las credenciales y tokens no están incluidos en este repositorio.

## Flujo

1. La GRU genera un nombre carácter por carácter.
2. Se verifica que el nombre no exista exactamente en `dinos.csv`.
3. Ollama crea la identidad del dinosaurio con:
   - apariencia;
   - hábitat;
   - alimentación;
   - comportamiento;
   - característica distintiva.
4. SD-Turbo genera una imagen basada en la identidad.
5. La identidad se conserva como contexto durante el chat.
6. Si el usuario pide una descripción o presentación, se reutiliza la imagen ya generada.
7. El botón **Nuevo Dinosaurio** repite el proceso sin reentrenar la GRU.

## Archivos principales

- `01_generador_nombres.ipynb`: entrenamiento, comparación de modelos y experimentación.
- `modelo_gru_dinosaurios.pth`: pesos del modelo final.
- `config_modelo.json`: configuración del modelo.
- `app.py`: aplicación web.
- `start_ollama.sh`: configuración e inicio de Ollama mediante Docker.
- `dinos.csv`: dataset utilizado por la aplicación.
- `dinos.txt`: dataset original utilizado como base.
- `evidencias/`: resultados experimentales.
- `imagenes/`: imagen utilizada como evidencia de la generación.
- `requirements.txt`: dependencias principales del proyecto.

## Evidencias experimentales

En la carpeta `evidencias/` se encuentran:

- `curva_rnn.png`
- `curva_gru.png`
- `nombres_generados.txt`
- `resultados_experimentos.md`

Se probaron diferentes estrategias de muestreo:

- Temperatura = 0.7
- Temperatura = 1.0
- Temperatura = 1.3
- Top-k = 5
- Top-p = 0.9

Se seleccionó **Top-p = 0.9** por ofrecer un buen equilibrio entre variedad y coherencia.

También se generaron diez nombres que fueron comparados con el conjunto de entrenamiento para verificar que no fueran copias exactas.

## Modelo seleccionado

La GRU obtuvo un mejor desempeño que la RNN simple.

Durante el experimento de 30 épocas se observó que la pérdida de entrenamiento continuaba disminuyendo mientras la pérdida de validación comenzaba a aumentar después de aproximadamente 15 épocas.

Por esta razón se utilizó como modelo final una GRU entrenada durante 15 épocas.

La pérdida final de validación fue aproximadamente **1.51**.

## Generación de imágenes

Para la generación de imágenes se utiliza:

`stabilityai/sd-turbo`

La imagen se genera una sola vez para cada dinosaurio y luego se reutiliza durante el chat cuando el usuario solicita una descripción o presentación.

## Chat

El chat utiliza la identidad generada previamente por Ollama como contexto.

El dinosaurio puede ampliar sus respuestas, pero debe conservar sus características principales y evitar contradicciones con:

- su apariencia;
- su hábitat;
- su alimentación;
- su comportamiento;
- su característica distintiva.

## Nuevo Dinosaurio

El botón **Nuevo Dinosaurio**:

1. genera un nuevo nombre usando la GRU ya entrenada;
2. genera una nueva identidad con Ollama;
3. genera una nueva imagen;
4. actualiza el contexto del chat.

La GRU no se vuelve a entrenar.

## Seguridad

El repositorio no incluye:

- credenciales de AWS;
- authtoken de ngrok;
- archivos `.env`;
- claves privadas;
- contraseñas.

## Tecnologías utilizadas

- Python
- PyTorch
- Pandas
- Ollama
- Llama 3.2 3B
- Hugging Face Diffusers
- SD-Turbo
- Gradio
- Docker
- AWS SageMaker
- ngrok

## Resultado final

El proyecto integra un modelo recurrente a nivel de caracteres con un modelo de lenguaje local, un modelo de generación de imágenes y una aplicación web interactiva.

El flujo final permite generar una especie ficticia completa y conversar con ella manteniendo una identidad coherente.