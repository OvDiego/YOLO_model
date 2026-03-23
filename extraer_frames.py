import cv2
import os
import glob

def extraer_frames_rocas(carpeta_videos=r"C:\Users\diego\Documents\YOLO\YOLO_model\rocas_crudo", salto=15):
    carpeta_videos = os.path.abspath(carpeta_videos)
    
    archivos_video = glob.glob(os.path.join(carpeta_videos, "*.mp4"))
    
    if not archivos_video:
        print(f"No se encontraron videos en: {carpeta_videos}")
        return

    for video_path in archivos_video:
        nombre_video = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(carpeta_videos, nombre_video)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        saved_id = 0

        print(f"📸 Procesando: {nombre_video}...")

        while True:
            ret, frame = cap.read()
            if not ret: break 

            if frame_id % salto == 0:
                
                filename = os.path.join(output_folder, f"{nombre_video}_{saved_id:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_id += 1
            frame_id += 1

        cap.release()
        print(f"✅ {saved_id} imágenes listas en {output_folder}")

if __name__ == "__main__":
    # Salto de 20 para evitar tener 500 fotos casi iguales
    extraer_frames_rocas(salto=20)