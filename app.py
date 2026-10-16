import os
import json
import random
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import gradio as gr

from diffusers import AutoPipelineForText2Image


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = "/home/ec2-user/SageMaker"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

MODEL_PATH = os.path.join(BASE_DIR, "modelo_gru_dinosaurios.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "config_modelo.json")
DINOS_PATH = os.path.join(BASE_DIR, "dinos.csv")

DEVICE = torch.device("cpu")


# ============================================================
# CARGAR CONFIGURACIÓN DEL MODELO
# ============================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

vocab = config["vocab"]
char_to_id = config["char_to_id"]
id_to_char = {int(k): v for k, v in config["id_to_char"].items()}

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"


# ============================================================
# MODELO GRU
# ============================================================

class CharGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=64,
        hidden_size=256,
        num_layers=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=char_to_id[PAD]
        )

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.gru(x)
        logits = self.fc(out)
        return logits


modelo_gru = CharGRU(
    vocab_size=len(vocab),
    emb_dim=config["embedding_dim"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"]
).to(DEVICE)

modelo_gru.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

modelo_gru.eval()


# ============================================================
# NOMBRES REALES
# ============================================================

df = pd.read_csv(DINOS_PATH)

nombres_reales = set(
    df["name"]
    .astype(str)
    .str.strip()
    .str.lower()
    .tolist()
)


# ============================================================
# MUESTREO
# ============================================================

def sample_next_token(logits, temperature=1.0, top_p=0.9):

    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True
    )

    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs <= top_p
    mask[0] = True

    filtered_probs = sorted_probs[mask]
    filtered_indices = sorted_indices[mask]

    filtered_probs = filtered_probs / filtered_probs.sum()

    idx = torch.multinomial(filtered_probs, 1)

    return filtered_indices[idx].item()


def generar_nombre(max_length=30):

    input_id = torch.tensor(
        [[char_to_id[BOS]]],
        dtype=torch.long,
        device=DEVICE
    )

    nombre = ""

    with torch.no_grad():

        for _ in range(max_length):

            logits = modelo_gru(input_id)

            next_logits = logits[0, -1]

            next_id = sample_next_token(
                next_logits,
                temperature=1.0,
                top_p=0.9
            )

            next_char = id_to_char[next_id]

            if next_char == EOS:
                break

            if next_char not in [PAD, BOS]:
                nombre += next_char

            nuevo = torch.tensor(
                [[next_id]],
                dtype=torch.long,
                device=DEVICE
            )

            input_id = torch.cat(
                [input_id, nuevo],
                dim=1
            )

    return nombre.strip().lower()


def generar_nombre_nuevo():

    for _ in range(100):

        nombre = generar_nombre()

        if (
            len(nombre) >= 4
            and nombre not in nombres_reales
        ):
            return nombre.capitalize()

    return "Dinogenerado"


# ============================================================
# OLLAMA - IDENTIDAD
# ============================================================

def crear_identidad(nombre):

    prompt = f"""
Eres un experto en criaturas prehistóricas ficticias.

Crea una especie ficticia llamada {nombre}.

Responde en español y utiliza exactamente estos apartados:

Nombre:
Apariencia:
Hábitat:
Alimentación:
Comportamiento:
Característica distintiva:

La especie debe ser coherente.
Haz la descripción clara y relativamente breve.
"""

    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=180
    )

    r.raise_for_status()

    return r.json()["response"]


# ============================================================
# MODELO DE IMAGEN
# ============================================================

print("Cargando SD-Turbo...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

print("SD-Turbo cargado.")


def crear_prompt_imagen(nombre, identidad):

    return f"""
A fictional dinosaur called {nombre}.
Full body prehistoric creature.
Use the following description as visual reference:

{identidad}

Realistic detailed digital illustration,
prehistoric environment,
cinematic lighting,
full body visible.
"""


def generar_imagen(nombre, identidad):

    prompt = crear_prompt_imagen(
        nombre,
        identidad
    )

    imagen = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=512,
        height=512
    ).images[0]

    ruta = os.path.join(
        BASE_DIR,
        f"{nombre.lower()}_actual.png"
    )

    imagen.save(ruta)

    return ruta


# ============================================================
# CREAR DINOSAURIO COMPLETO
# ============================================================

def nuevo_dinosaurio():

    nombre = generar_nombre_nuevo()

    identidad = crear_identidad(nombre)

    imagen_path = generar_imagen(
        nombre,
        identidad
    )

    info_modelo = f"""
### Modelo generador de nombre

- Arquitectura: **GRU**
- Capas recurrentes: **2**
- Estado oculto: **256**
- Embedding: **64**
- Épocas modelo final: **15**
- Optimizador: **Adam**
- Muestreo: **Top-p = 0.9**
"""

    chat_inicial = [
        {
            "role": "assistant",
            "content": f"Ya puedes conversar con **{nombre}**. Pregúntale algo sobre su especie."
        }
    ]

    return (
        nombre,
        identidad,
        imagen_path,
        info_modelo,
        nombre,
        identidad,
        imagen_path,
        chat_inicial
    )


# ============================================================
# CHAT
# ============================================================

def responder_chat(
    mensaje,
    historial,
    nombre,
    identidad,
    imagen_path
):

    if not mensaje.strip():
        return historial, ""

    historial = historial or []

    historial.append({
        "role": "user",
        "content": mensaje
    })

    prompt = f"""
Actúa como el dinosaurio ficticio llamado {nombre}.

Esta es tu identidad oficial:

{identidad}

Reglas:
- Responde en primera persona como {nombre}.
- No contradigas la identidad anterior.
- Puedes ampliar detalles coherentes.
- Responde en español.
- Sé claro y breve.

Conversación actual:
"""

    for msg in historial[-8:]:

        if isinstance(msg["content"], str):
            prompt += (
                f"\n{msg['role']}: "
                f"{msg['content']}"
            )

    prompt += "\nassistant:"

    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=180
    )

    r.raise_for_status()

    respuesta = r.json()["response"]

    historial.append({
        "role": "assistant",
        "content": respuesta
    })

    # Si pide presentación o descripción,
    # también mostramos la imagen en el chat.
    texto = mensaje.lower()

    palabras_imagen = [
        "quién eres",
        "quien eres",
        "descríbete",
        "describete",
        "descripción",
        "descripcion",
        "preséntate",
        "presentate",
        "cómo eres",
        "como eres"
    ]

    if any(p in texto for p in palabras_imagen):

        historial.append({
            "role": "assistant",
            "content": gr.Image(
                value=imagen_path
            )
        })

    return historial, ""


# ============================================================
# INTERFAZ
# ============================================================

with gr.Blocks(
    title="Isla de Dinosaurios IA"
) as demo:

    gr.Markdown(
        """
# 🦖 Isla de Dinosaurios IA

Generación de una especie ficticia usando:

**GRU → Ollama → Text-to-Image → Chat**
"""
    )

    nombre_state = gr.State()
    identidad_state = gr.State()
    imagen_state = gr.State()

    with gr.Row():

        with gr.Column():

            nombre = gr.Textbox(
                label="Nombre",
                interactive=False
            )

            identidad = gr.Markdown()

            info_modelo = gr.Markdown()

            nuevo_btn = gr.Button(
                "🦕 Nuevo Dinosaurio",
                variant="primary"
            )

        with gr.Column():

            imagen = gr.Image(
                label="Imagen del dinosaurio",
                type="filepath",
                interactive=False
            )

    gr.Markdown("## 💬 Habla con tu dinosaurio")

    chatbot = gr.Chatbot(
        height=400
    )

    mensaje = gr.Textbox(
        label="Tu mensaje",
        placeholder="Pregúntale algo al dinosaurio..."
    )

    enviar = gr.Button("Enviar")

    nuevo_btn.click(
        fn=nuevo_dinosaurio,
        outputs=[
            nombre,
            identidad,
            imagen,
            info_modelo,
            nombre_state,
            identidad_state,
            imagen_state,
            chatbot
        ]
    )

    enviar.click(
        fn=responder_chat,
        inputs=[
            mensaje,
            chatbot,
            nombre_state,
            identidad_state,
            imagen_state
        ],
        outputs=[
            chatbot,
            mensaje
        ]
    )

    mensaje.submit(
        fn=responder_chat,
        inputs=[
            mensaje,
            chatbot,
            nombre_state,
            identidad_state,
            imagen_state
        ],
        outputs=[
            chatbot,
            mensaje
        ]
    )


# ============================================================
# ARRANQUE
# ============================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)