import cv2
import os
import glob

def extraer_todos_los_frames(salto=20):
    # RUTAS CONFIRMADAS EN TU DIAGNÓSTICO
    videos_dir = r"C:\Users\diego\Documents\YOLO\YOLO_model\fin\videos"
    output_dir = r"C:\Users\diego\Documents\YOLO\YOLO_model\fin\raw_img_fin"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Buscamos todos los mp4
    archivos = glob.glob(os.path.join(videos_dir, "*.mp4")) + glob.glob(os.path.join(videos_dir, "*.MP4"))
    
    print(f"🚀 Iniciando extracción de {len(archivos)} videos...")

    total_imagenes = 0

    for video_path in archivos:
        nombre_video = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        
        frame_id = 0
        saved_id = 0

        print(f"📸 Procesando: {nombre_video}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            # Lógica del salto para no saturar de fotos iguales
            if frame_id % salto == 0:
                # Nombre único: nombre_video + id
                filename = os.path.join(output_dir, f"{nombre_video}_frame_{saved_id:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_id += 1
                total_imagenes += 1

            frame_id += 1

        cap.release()
        print(f"✅ Finalizado: {nombre_video} ({saved_id} imágenes extraídas)")

    print(f"\n--- RESUMEN FINAL ---")
    print(f"📂 Carpeta: {output_dir}")
    print(f"🖼️ Total de imágenes nuevas: {total_imagenes}")

if __name__ == "__main__":
    # Salto de 20 es ideal para que el banderín cambie de posición entre fotos
    extraer_todos_los_frames(salto=20)