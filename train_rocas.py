from ultralytics import YOLO

def main():
    
    model_path = r'C:\Users\diego\Documents\YOLO\YOLO_model\runs\detect\train3\weights\best.pt'

    # Cargamos el modelo con el conocimiento de train3
    model = YOLO(model_path)

    # Entrenamos con los datos nuevos que acabas de etiquetar
    model.train(
        data='rocas.yaml',
        epochs=150,    
        imgsz=640,
        batch=16,
        device=0,       # RTX 4050 dándolo todo
        project='runs/detect', # Lo mandamos a la carpeta estándar
        name='train4' 
    )

if __name__ == '__main__':
    main()