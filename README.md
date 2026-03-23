

### 1. Clonar y Entrar
```powershell
git clone [https://github.com/OvDiego/YOLO_model.git](https://github.com/OvDiego/YOLO_model.git)
cd YOLO_model


### 2. Entorno virtual
# Crear entorno
python -m venv venv_yolo

# Activar entorno
.\venv_yolo\Scripts\activate

# Instalar librerías necesarias
pip install ultralytics opencv-python labelme

### 3. Validar gpu
python gpu.py

### 4. Extracciòn, etiquetado y conversion.
python fin/extract_fin.py
labelme
Para que el modelo aprenda correctamente, sigue este proceso riguroso:

1. Lanzar herramienta: Ejecuta labelme en la terminal.

2. Cargar imágenes: Clic en Open Dir y selecciona la carpeta de imágenes crudas (raw_img_fin o raw_img).

3. Configurar guardado: Ve al menú File y activa Save Automatically.

4. Dibujar cuadros: Presiona la tecla R para crear un rectángulo alrededor del objeto.

5. Etiquetas exactas: Usa exclusivamente estos nombres según el objeto: red_stone, blue_stone, green_stone o fin_flag.

Navegación: Usa D para la siguiente imagen y A para la anterior.
python fin/convertir_fin.py


python fin/train_fin.py
