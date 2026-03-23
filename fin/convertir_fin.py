import os
import json
import shutil
import random

# --- CONFIGURACIÓN DE RUTAS ---
source_folder = r"C:\Users\diego\Documents\YOLO\YOLO_model\fin\raw_img_fin"
output_dir = r"C:\Users\diego\Documents\YOLO\YOLO_model\fin\Dataset_Banderin"
train_ratio = 0.8  

class_map = {"fin_flag": 0}

def convert_to_yolo(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0
    w, h = box[1] - box[0], box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_fin():
    print(f"🔍 Buscando archivos JSON en: {source_folder}")
    
    if not os.path.exists(source_folder):
        print(f"❌ ERROR: La carpeta {source_folder} no existe.")
        return

    # Obtener lista de JSONs
    json_files = [f for f in os.listdir(source_folder) if f.endswith('.json')]
    print(f"📄 Se encontraron {len(json_files)} archivos JSON para procesar.")

    if len(json_files) == 0:
        print("⚠️ No hay nada que convertir. ¿Seguro que guardaste las etiquetas en Labelme?")
        return

    # Limpiar y crear carpetas de salida
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    total_procesados = 0

    for json_file in json_files:
        json_path = os.path.join(source_folder, json_file)
        base_name = os.path.splitext(json_file)[0]
        
        # Buscar imagen correspondiente
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            temp = os.path.join(source_folder, base_name + ext)
            if os.path.exists(temp):
                img_path = temp
                break
        
        if not img_path:
            print(f"❓ Imagen no encontrada para: {json_file}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        im_w, im_h = data['imageWidth'], data['imageHeight']
        yolo_lines = []
        
        for shape in data['shapes']:
            # IMPORTANTE: El nombre debe ser EXACTO al que pusiste en Labelme
            if shape['label'] == "flag":
                p = shape['points']
                box = (min(p[0][0], p[1][0]), max(p[0][0], p[1][0]), 
                       min(p[0][1], p[1][1]), max(p[0][1], p[1][1]))
                bb = convert_to_yolo((im_w, im_h), box)
                yolo_lines.append(f"0 {bb[0]} {bb[1]} {bb[2]} {bb[3]}")

        if yolo_lines:
            split = 'train' if random.random() < train_ratio else 'val'
            shutil.copy(img_path, os.path.join(output_dir, 'images', split, os.path.basename(img_path)))
            with open(os.path.join(output_dir, 'labels', split, base_name + ".txt"), 'w') as out_f:
                out_f.write('\n'.join(yolo_lines))
            total_procesados += 1

    print(f"✅ ¡ÉXITO! Se generaron {total_procesados} imágenes en el Dataset.")

if __name__ == '__main__':
    process_fin()