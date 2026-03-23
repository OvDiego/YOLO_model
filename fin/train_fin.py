from ultralytics import YOLO
import os

def main():
    # Usamos el modelo nano (yolov8n.pt) para que la detección sea 
    # ultra rápida en el hardware del Rover durante la misión
    model = YOLO('yolov8n.pt') 

    # Definimos la ruta de salida personalizada que pediste
    output_base = r'C:\Users\diego\Documents\YOLO\YOLO_model\fin\entrenamiento_resultados'

    model.train(
        data='fin.yaml',        # El archivo que configuramos arriba
        epochs=100,             # 100 vueltas son ideales para este volumen de fotos
        imgsz=640,              # Resolución estándar de entrenamiento
        batch=16,               # Ajuste óptimo para tus 6GB de VRAM
        device=0,               # Forzamos el uso de la RTX 4050
        project=output_base,    # Carpeta raíz de este entrenamiento
        name='detector_banderin' # Subcarpeta específica del experimento
    )

if __name__ == '__main__':
    main()