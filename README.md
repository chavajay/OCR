# OCR desde Cero — Examen Final IA

Sistema de Reconocimiento Óptico de Caracteres (OCR) implementado completamente desde cero usando, NumPy y PyTorch. Sin Tesseract, EasyOCR ni ninguna librería de alto nivel para reconocimiento de texto.

---

## Características

- **Pipeline modular:** 4 componentes independientes (preprocesador, segmentador, clasificador, exportador).
- **Preprocesamiento robusto:** Threshold fijo 127, denoising adaptativo, corrección de skew.
- **Segmentación por componentes conectados:** Detección de líneas, caracteres y espacios entre palabras.
- **Fusión inteligente de diacríticos:** Une puntos de 'i'/'j' con sus astas sin fusionar caracteres adyacentes.
- **Clasificación CNN residual:** Red convolucional con 4 niveles de características (64→128→256→256 canales), batch normalization, dropout y conexiones residuales.
- **75 clases:** Dígitos (0-9), mayúsculas (A-Z), minúsculas (a-z), puntuación (.,;:!?-()/"') y espacio.
- **Entrenamiento con datos sintéticos:** 32 fuentes TrueType + 8 fuentes OpenCV, degradaciones realistas, aumento de datos.
- **Post-procesamiento:** Reglas heurísticas para corregir confusiones l/I, O/0, c/C según contexto.
- **Exportación dual:** Texto plano (.txt) y Markdown (.md).

---

## Arquitectura

```
Imagen → ImagePreprocessor → ProjectionSegmenter → OCRClassifier → ResultExporter → TXT/MD
```

### Módulos

| Módulo | Archivo | Responsabilidad |
|---|---|---|
| `ImagePreprocessor` | `src/preprocessor.py` | Conversión a grises, denoising, binarización (threshold fijo 127), corrección de skew |
| `ProjectionSegmenter` | `src/segmenter.py` | Componentes conectados, detección de líneas, fusión de diacríticos, normalización 28×28 |
| `OCRClassifier` | `src/classifier.py` | CNN residual con 75 clases, entrenamiento, predicción con probabilidades |
| `ResultExporter` | `src/exporter.py` | Exportación a .txt y .md |

---

## Requisitos

- **Python 3.10+**
- **pip**
- **~100 MB** de espacio en disco
- **~2 GB de RAM** (mínimo)

### Dependencias

| Paquete | Versión mínima | Propósito |
|---|---|---|---|
| numpy | ≥ 1.24.0 | Operaciones matriciales |
| opencv-python | — | Procesamiento de imágenes, componentes conectados |
| torch | ≥ 2.0.0 | Red neuronal CNN |
| Pillow | ≥ 10.0.0 | Renderizado de texto TrueType |

---

## Instalación

```bash
# 1. Clonar o copiar el proyecto
cd /ruta/al/proyecto

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar resto de dependencias
pip install -r requirements.txt

# 5. Verificar instalación
python -c "import cv2; import torch; import numpy; print('OK:', cv2.__version__, torch.__version__, numpy.__version__)"
```

---

## Estructura del proyecto

```
IA/
├── main.py                          ← Punto de entrada del pipeline OCR
├── requirements.txt                 ← Dependencias
├── test.png                         ← Imagen de prueba incluida
├── DOCUMENTO_TECNICO.md             ← Documentación técnica completa
│
├── src/                             ← Código fuente
│   ├── preprocessor.py              ← Preprocesamiento (threshold fijo, denoising, deskew)
│   ├── segmenter.py                 ← Segmentación (componentes conectados, líneas, espacios)
│   ├── classifier.py                ← CNN residual (entrenamiento, predicción)
│   └── exporter.py                  ← Exportación a .txt y .md
│
├── models/                          ← Modelos entrenados
│   ├── ocr_printed_best.pth         ← Mejor modelo (76.6% val_acc) ~8.8 MB
│   └── ocr_printed.pth              ← Último modelo guardado
│
├── scripts/                         ← Scripts de entrenamiento y utilidades
│   ├── train_final.py               ← Entrenamiento final (32 TTF + 8 OpenCV, 800 muestras/clase)
│   ├── generate_printed_dataset.py  ← Generación de dataset sintético
│   ├── evaluate_ocr.py              ← Suite de evaluación cuantitativa
│   └── generate_test_images.py      ← Generación de imágenes de prueba
│
├── tests/                           ← Imágenes de prueba
│   └── images/                      ← 17 imágenes generadas para validación
│
├── output/                          ← Resultados del OCR
│   ├── output.txt
│   └── output.md
│
└── logs/                            ← Logs de entrenamiento
    └── training.log
```

---

## Cómo ejecutar

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar OCR sobre la imagen de prueba
python main.py test.png

# Ejecutar con ruta personalizada
python main.py /ruta/a/mi_imagen.jpg

# Especificar modelo y directorio de salida
python main.py test.png --model models/ocr_printed_best.pth --output-dir output
```

### Opciones de línea de comandos

| Opción | Default | Descripción |
|---|---|---|
| `image` | (obligatorio) | Ruta a la imagen de entrada |
| `--model` | `models/ocr_printed_best.pth` | Ruta al modelo CNN |
| `--output-dir` | `output` | Directorio para archivos de salida |

---

## Imágenes de prueba

### Ubicación recomendada

Puedes colocar tus imágenes de prueba en la raíz del proyecto o en cualquier ruta accesible:

```
IA/
├── test.png                         ← Imagen incluida
├── tests/
│   └── images/                      ← Imágenes generadas por generate_test_images.py
│       ├── test_simple_lower.png
│       ├── test_uppercase.png
│       └── ... (17 en total)
```

Para generar las imágenes de prueba incluidas:

```bash
python scripts/generate_test_images.py
```

### Formatos soportados

| Formato | Extensión | Soportado |
|---|---|---|
| PNG | `.png` | ✅ |
| JPEG | `.jpg`, `.jpeg` | ✅ |
| BMP | `.bmp` | ✅ |
| TIFF | `.tiff`, `.tif` | ✅ |

### Recomendaciones para mejores resultados

| Hacer ✅ | Evitar ❌ |
|---|---|
| Imágenes con buen contraste | Imágenes borrosas |
| Texto horizontal (sin rotación) | Texto inclinado >10° |
| Resolución ≥ 150 DPI | Imágenes <100px de ancho |
| Texto impreso claro | Texto manuscrito |
| Fondo blanco uniforme | Fondos con patrones o texturas |

---

## Ejemplos de uso

### Ejemplo 1: Imagen de prueba incluida

```bash
python main.py test.png
```

**Salida esperada (precisión actual ~87%):**

```
Loaded image: (400, 800, 3)
Preprocessed: (400, 800)
Detected 2 lines
Detected 31 characters
  Line 1: Hello OCR World
  Line 2: Testing pipeline

=== OCR Results ===
Line 1: Hello OCR World
Line 2: Testing pipeline

Text exported to: .../output/output.txt
Markdown exported to: .../output/output.md
```

> *Los errores 'I' por 'l' son la confusión principal. En fuentes serif como DejaVu Serif, la precisión alcanza el 100%.*

### Ejemplo 2: Probar todas las imágenes generadas

```bash
for img in tests/images/test_*.png; do
    echo "=== $(basename $img) ==="
    python main.py "$img" --output-dir output/test_results 2>&1 | grep "Line"
    echo ""
done
```

---

## Entrenamiento

El entrenamiento principal se realiza con `scripts/train_final.py`:

```bash
# Entrenar modelo desde cero (toma ~2 horas en CPU)
python scripts/train_final.py
```

Este script:
1. Genera ~60,000 muestras sintéticas (800 por cada una de 75 clases)
2. Renderiza con 32 fuentes TrueType + 8 fuentes OpenCV
3. Aplica degradaciones realistas (ruido, blur, JPEG, contraste)
4. Aplica aumento de datos (variación de grosor, variación de threshold)
5. Entrena una CNN residual durante máximo 80 épocas con early stopping
6. Guarda el mejor modelo en `models/ocr_printed_best.pth`


### Evaluación cuantitativa

```bash
python scripts/evaluate_ocr.py
```

Evalúa el modelo contra 15 textos de prueba en múltiples fuentes, condiciones de ruido, rotación y escalado.

---

## Solución de problemas

### Error: `ModuleNotFoundError: No module named 'torch'`

```bash
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Error: `FileNotFoundError: Image file not found`

```bash
ls -la tu_imagen.png                     # Verificar que existe
python main.py /ruta/completa/imagen.png # Usar ruta absoluta
```

### Error: `Model file not found: models/ocr_printed_best.pth`

```bash
ls -lh models/
# Si no existe, entrenar:
python scripts/train_final.py
```

### La precisión es baja

1. Verificar que la imagen tenga buen contraste y texto horizontal.
2. Usar el threshold fijo (predeterminado en `main.py`).
3. Si el texto está rotado >10°, aplicar enderezamiento manual primero.
4. El modelo está entrenado para texto impreso sans-serif. Fuentes decorativas o manuscritas no funcionarán bien.

### Los caracteres no se detectan correctamente

El segmentador depende del threshold fijo 127. Si la imagen tiene sombras o iluminación no uniforme, probar:

```python
# En main.py, cambiar a threshold adaptativo:
binary = pre.preprocess_adaptive(image, deskew=True, binarization="adaptive")
```

### Los espacios entre palabras no aparecen

Si las palabras aparecen pegadas, el detector de espacios puede no estar activándose. Ajustar el threshold en `src/segmenter.py` línea 285:

```python
# Reducir el threshold de detección de espacios
if gap > max(avg_width * 0.3, 4):  # más sensible
```

---

## Licencia

Proyecto académico — Examen Final de Inteligencia Artificial — Realizado por Francisco Chavajay
# OCR
