# Clasificación Multiclase: Scarlett Johansson vs Natalie Portman

Este proyecto implementa un **modelo de clasificación de imágenes multiclase** para distinguir entre Scarlett Johansson y Natalie Portman. Incluye evaluación del modelo, análisis de error y una interfaz web para consumirlo.
## 🔹 Objetivo

- Clasificar imágenes entre **Scarlett Johansson** y **Natalie Portman**.
- Evaluar el desempeño mediante **matriz de confusión**.
- Analizar errores y determinar mejoras.
- Desarrollar una **interfaz web** para consumir el modelo.

---

## ⚙️ Preparación de Datos

- Organizar las imágenes en carpetas por celebridad.
- Solo se permiten imágenes de: `Scarlett` y `Natalie`.
- Transformaciones aplicadas:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [...]  # etiquetas reales
y_pred = [...]  # etiquetas predichas

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Scarlett", "Natalie"],
            yticklabels=["Scarlett", "Natalie"])
plt.show()
🌐 Interfaz Web

Se implementa con Gradio para predecir nuevas imágenes:

import gradio as gr
from PIL import Image

def predict(image):
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
    return ["Scarlett", "Natalie"][pred.item()]

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             title="Clasificación Scarlett vs Natalie").launch()
