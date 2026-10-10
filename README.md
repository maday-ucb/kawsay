# Kawsay · Recuperación semántica de documentos de revitalización lingüística

Sistema de búsqueda semántica sobre un corpus de documentos académicos y de
campo sobre revitalización de lenguas indígenas. Permite consultar en lenguaje
natural (español o inglés), recupera los fragmentos más relevantes y puede
redactar una respuesta citando sus fuentes.

El código está en inglés. La interfaz, la documentación y los comentarios
orientados al usuario están en español.

---

## Qué hace el sistema

```
PDFs originales → fragmentos Markdown → embeddings + índice BM25
                                                  ↓
Consulta en lenguaje natural → ranking semántico → filtro → fragmento → LLM → verificar citas
                           └→ ranking BM25 para comparar
```

| Componente | Implementación | Función |
|---|---|---|
| Línea base | BM25 (rank-bm25) | Recuperación léxica estándar |
| Búsqueda semántica | multilingual-e5-base preentrenado | Embeddings de consultas y fragmentos |
| Índice | Matriz de vectores normalizados (20161×768) | Similitud coseno por producto punto |
| Generador | Ollama · qwen2.5:1.5b local | Redactar respuesta basada en el fragmento |
| Verificación | Citas literales en la fuente | Rechazar respuestas con citas inventadas |
| Backend | FastAPI | API REST con validación y manejo de errores |
| Frontend | HTML / CSS / JavaScript | Interfaz de consulta, corpus y evaluación |

**Indexar no es entrenar.** El modelo de embeddings está preentrenado y no se
ajusta con este corpus. BM25 sí ajusta su vocabulario al corpus. El umbral de
paso a generación se elige usando las consultas de validación.

---

## Requisitos del sistema

- Python 3.14 (o 3.12+)
- Windows 10/11, macOS o Linux
- 4 GB de RAM mínimo (8 GB recomendado)
- [Ollama](https://ollama.com/download) instalado y corriendo (para generación)
- Espacio en disco: ~600 MB para el modelo de embeddings + tamaño del corpus

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd kawsay
```

### 2. Crear entorno virtual e instalar dependencias

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar token de Hugging Face (recomendado)

Crea una cuenta en https://huggingface.co y genera un token de lectura en
https://huggingface.co/settings/tokens. Configúralo para evitar límites de
descarga:

Windows PowerShell:
```powershell
$env:HF_TOKEN = "hf_tu_token_aqui"
```

macOS / Linux:
```bash
export HF_TOKEN="hf_tu_token_aqui"
```

### 4. Instalar Ollama y descargar el modelo generador

1. Descarga e instala Ollama desde https://ollama.com/download
2. Abre Ollama (queda corriendo en segundo plano)
3. En una terminal:

```bash
ollama pull qwen2.5:1.5b
```

4. Verifica que todo esté listo:

```bash
python -m scripts.check_ollama
```

---

## Preparar el corpus

### Opción A — Desde tus PDFs (flujo completo)

1. Copia tus PDFs a `data/raw_pdfs/`

2. Genera la planilla de auditoría:
```bash
python -m scripts.audit_pdfs
```
Abre `audit.csv` en Excel, revisa la columna `include` y elimina de
`data/raw_pdfs/` los documentos que no quieras incluir.

3. Convierte los PDFs seleccionados a Markdown:
```bash
python -m scripts.ingest_pdfs --dry-run   # previsualizar sin escribir
python -m scripts.ingest_pdfs             # ejecutar
```

Los archivos `D001_xxx.md`, `D002_xxx.md`, etc. quedarán en `data/documents/`.

### Opción B — Documentos Markdown directos

Coloca archivos con este formato en `data/documents/`:

```markdown
# Título del documento

## Nombre de la sección

Texto del contenido. Cada sección debe tener entre 20 y 90 palabras
aproximadamente para respetar el límite de tokens del modelo de embeddings.
```

Reglas obligatorias:
- Nombre de archivo: `D001_nombre.md`, `D002_nombre.md`, etc.
- Primera línea: `# Título` (obligatorio)
- Secciones: `## Encabezado` seguido del texto

---

## Construir el índice

```bash
python -m app.download_model   # descarga el modelo de embeddings (~280 MB, solo una vez)
python -m app.build_index      # genera embeddings + índice BM25 + métricas
```

Si tienes `data/queries.json`, `build_index` calculará automáticamente
NDCG@5, Recall@5, Precision@5 y MRR@5 para semántica y BM25.

---

## Ejecutar el sistema

```bash
python run.py
```

Abre **http://127.0.0.1:8995** en el navegador. Mantén la terminal abierta.
`Ctrl+C` detiene el servidor.

Opciones adicionales:
```bash
python run.py --port 8997      # si el puerto 8995 está ocupado
python run.py --rebuild        # reindexar y arrancar en un solo paso
```

---

## Evaluación

### Preparar queries.json

Crea `data/queries.json` con este formato:

```json
[
  {
    "query_id":    "Q01",
    "family_id":   "F01",
    "split":       "test",
    "query":       "Experiencias de revitalización del quechua mediante educación.",
    "answerable":  true,
    "relevant_ids": ["D03-S2", "D07-S1"]
  },
  {
    "query_id":    "Q02",
    "family_id":   "F01",
    "split":       "validation",
    "query":       "Programas educativos bilingües para preservar el quechua.",
    "answerable":  true,
    "relevant_ids": ["D03-S2"]
  }
]
```

Reglas:
- `family_id` agrupa paráfrasis del mismo hecho
- Una familia no puede aparecer en validación y test a la vez
- `answerable: false` cuando `relevant_ids` está vacío
- Incluir al menos 6 consultas sin respuesta para evaluar abstención

### Ejecutar evaluación completa

```bash
python -m app.build_index
```

Los resultados quedan en `artifacts/metrics.json` y `artifacts/evaluation.md`.

---

## Pruebas

```bash
python -m pytest -q                          # todos los tests
python -m pytest -v tests/test_kawsay.py     # con detalle
```

Los tests no llaman a Ollama: usan un stub del generador. El índice real
sí debe estar construido antes de correr los tests.

Cobertura de la suite (67 tests):

| Grupo | Qué verifica |
|---|---|
| Corpus | Carga, unicidad de IDs, formato, hash de cambios |
| Recuperación | Embeddings normalizados, score = producto punto, TOP_K=5, tie-breaking |
| Semántica vs BM25 | Semántica cae menos que BM25 al pasar de consulta directa a paráfrasis |
| Métricas | NDCG@5, Recall@5, Precision@5, MRR@5; gate de umbral |
| Validación de generación | 7 casos de fallo (chunk desconocido, cita inventada, JSON inválido, etc.) |
| Contratos de API | 200 en endpoints de lectura; 422 en entradas inválidas; 503 sin índice; 429 concurrente |
| Casos del enunciado | Caso válido, caso difícil (paráfrasis), entradas inválidas, flujo completo |
| Métricas del sistema | `/api/metrics` expone datos reales del corpus indexado |

---

## Qué probar en la interfaz

1. Escribe **"Experiencias de revitalización del quechua mediante educación"**
   y pulsa **Solo buscar**. Compara las pestañas Semántica y BM25.
2. Pulsa **Consultar con fuentes** para ejecutar el generador local (requiere Ollama).
3. Lee la respuesta, el extracto citado y abre la fuente con **Ver fuente**.
4. Prueba una paráfrasis: **"Programas bilingües para preservar lenguas originarias"**.
5. Prueba una consulta difícil: **"Uso de tecnología móvil para enseñar aymara"**.
6. Prueba un dato ausente: **"Número exacto de hablantes de quechua en 2024"**.
7. Prueba una consulta fuera del dominio: **"unicornio minotauro"** — observa el banner de baja confianza y que el gate bloquea la generación.
8. Abre **Corpus** para explorar los documentos indexados.
9. Abre **Evaluación** para ver las métricas de recuperación.
10. Usa el botón ☀/🌙 en el header para alternar entre modo oscuro y claro.

---

## Estructura del proyecto

```
kawsay/
├── run.py                      # punto de entrada
├── requirements.txt
├── app/
│   ├── settings.py             # rutas y parámetros
│   ├── documents.py            # lectura y validación del corpus
│   ├── retrieval.py            # embeddings + BM25
│   ├── build_index.py          # construir índice y calcular métricas
│   ├── download_model.py       # descargar modelo de embeddings
│   ├── evaluation.py           # NDCG, Recall, MRR, gate
│   ├── generation.py           # Ollama + validación de citas
│   └── main.py                 # FastAPI backend
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── scripts/
│   ├── audit_pdfs.py           # planilla de auditoría del corpus
│   ├── ingest_pdfs.py          # conversión PDF → Markdown
│   └── check_ollama.py         # verificar Ollama
├── data/
│   ├── raw_pdfs/               # PDFs originales (excluidos de git)
│   ├── documents/              # corpus Markdown generado
│   └── queries.json            # consultas etiquetadas para evaluación
├── artifacts/
│   ├── index.joblib            # índice generado (excluido de git)
│   ├── metrics.json            # métricas de evaluación
│   └── evaluation.md           # informe de recuperación
├── tests/
│   ├── conftest.py
│   └── test_kawsay.py
└── docs/
```

---

## Resultados de recuperación

| Métrica | Semántica | BM25 |
|---|---:|---:|
| NDCG@5 | 0.5624 | 0.2508 |
| Recall@5 | 0.5083 | 0.2250 |
| Precision@5 | 0.4400 | 0.2000 |
| MRR@5 | 0.7833 | 0.4333 |

Conjunto de prueba: 13 consultas (10 respondibles, 3 sin respuesta en el corpus).
Umbral de generación: 0.8641 · F1 del filtro: 0.818.

La búsqueda semántica supera a BM25 en todas las métricas (~2x en NDCG@5).
BM25 bajo es esperado: las consultas son temáticas y formuladas como paráfrasis,
no como búsquedas de palabras clave exactas.

---

## Limitaciones

- El modelo de embeddings tiene un límite de 512 tokens (~380 palabras) por fragmento.
  Los chunks están configurados a 400 palabras para mantenerse dentro del límite.
- El generador recibe solo el fragmento mejor rankeado. Consultas que requieren
  combinar varios documentos quedan fuera del alcance validado.
- BM25 no captura sinónimos ni variantes morfológicas entre lenguas.
- El corpus es un subconjunto seleccionado; documentos no incluidos no son recuperables.
- La verificación de citas es literal, no semántica: una cita correcta no garantiza
  que la respuesta completa esté respaldada por el fragmento.
- Ollama corre en CPU en este entorno; la generación puede tardar 10-30 segundos.

---

## Problemas frecuentes

**Puerto ocupado:**
```bash
python run.py --port 8997
```

**Índice desactualizado tras agregar documentos:**
```bash
python -m app.build_index
```

**Ollama no responde:**
Abre la aplicación Ollama y espera a que aparezca el ícono en la barra de tareas.

**PDF sin texto extraíble (escaneado):**
El pipeline no incluye OCR. Estos documentos generarán chunks vacíos y serán
omitidos automáticamente. Usa una herramienta de OCR externa primero.

**Memoria insuficiente durante indexación:**
Reduce el batch size en `build_index.py` (línea `batch_size=32`) a `batch_size=8`.

**Indexación muy lenta (solo CPU):**
Configura las variables de entorno antes de correr `build_index`:
```powershell
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
python -m app.build_index
```
