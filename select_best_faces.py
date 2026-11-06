import os
import glob
import shutil
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ExifTags
import pillow_heif

pillow_heif.register_heif_opener()

# ================= CONFIGURAZIONE =================
RAW_POOL_DIR = "./pool_2000_foto"   # Cartella con tutte le ~2000 foto
SELECTED_DIR = "./foto_antonia"     # Cartella di output (pronta per morph_evolution.py)

BUCKET_MODE = "HALF_YEAR"           # "YEAR" (1-2 foto/anno) oppure "HALF_YEAR" (1-2 foto ogni 6 mesi)
TOP_K_PER_BUCKET = 2                # Quante foto migliori salvare per ogni periodo (per darti margine di scelta)

# Soglie geometriche (più basse sono, più il volto deve essere dritto)
MAX_ROLL_DEG = 15.0                 # Inclinazione laterale massima della testa (in gradi)
MAX_YAW_ASYMMETRY = 0.22            # Rotazione destra/sinistra (0 = perfettamente frontale)
MAX_PITCH_DEVIATION = 0.28          # Inclinazione alto/basso rispetto al piano neutro
MIN_EYE_OPENNESS = 0.18             # Soglia apertura occhi (scarta occhi chiusi/ammiccamenti)
MIN_FACE_SIZE_PX = 120              # Larghezza minima del volto in pixel
# ==================================================

RIGHT_IRIS_IDX = [469, 470, 471, 472]
LEFT_IRIS_IDX = [474, 475, 476, 477]


def get_photo_datetime(filepath):
    """Estrae data e ora dallo scatto EXIF; usa mtime del file come fallback."""
    try:
        img = Image.open(filepath)
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTime"):
                    clean_str = str(value).strip().split(".")[0]
                    return datetime.strptime(clean_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(os.path.getmtime(filepath))


def load_image_bgr(filepath):
    """Carica l'immagine (inclusi HEIC) rispettando la rotazione EXIF."""
    pil_img = Image.open(filepath)
    try:
        exif = pil_img.getexif()
        orientation = exif.get(274, 1)
        if orientation == 3:
            pil_img = pil_img.rotate(180, expand=True)
        elif orientation == 6:
            pil_img = pil_img.rotate(270, expand=True)
        elif orientation == 8:
            pil_img = pil_img.rotate(90, expand=True)
    except Exception:
        pass
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def compute_eye_ear(pts, top_idx, bottom_idx, inner_idx, outer_idx):
    """Calcola l'Eye Aspect Ratio per verificare che gli occhi siano ben aperti."""
    v_dist = np.linalg.norm(pts[top_idx] - pts[bottom_idx])
    h_dist = np.linalg.norm(pts[inner_idx] - pts[outer_idx]) + 1e-6
    return v_dist / h_dist


def evaluate_face(img_bgr, face_mesh):
    """
    Valuta un'immagine. Restituisce (score, metrics_dict) se supera i filtri
    di frontalità e qualità, altrimenti (None, motivo_scarto).
    """
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, "Nessun volto"

    # Se ci sono più volti, prende il più grande (il soggetto in primo piano)
    best_pts = None
    max_area = -1
    for face_lms in results.multi_face_landmarks:
        pts = np.array([(lm.x * w, lm.y * h) for lm in face_lms.landmark], dtype=np.float32)
        area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        if area > max_area:
            max_area = area
            best_pts = pts

    pts = best_pts
    face_w = pts[:, 0].max() - pts[:, 0].min()
    face_h = pts[:, 1].max() - pts[:, 1].min()

    if face_w < MIN_FACE_SIZE_PX:
        return None, f"Volto troppo piccolo ({int(face_w)} px)"

    # 1. Calcolo ROLL (inclinazione testa sugli occhi)
    eye_l = np.mean(pts[RIGHT_IRIS_IDX], axis=0)
    eye_r = np.mean(pts[LEFT_IRIS_IDX], axis=0)
    roll_deg = abs(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])))
    if roll_deg > MAX_ROLL_DEG:
        return None, f"Testa inclinata (Roll: {roll_deg:.1f}°)"

    # 2. Calcolo YAW (simmetria punta del naso [1] rispetto ai margini zigomatici [234, 454])
    nose_tip = pts[1]
    left_cheek = pts[234]
    right_cheek = pts[454]
    dist_l = np.linalg.norm(nose_tip - left_cheek)
    dist_r = np.linalg.norm(nose_tip - right_cheek)
    yaw_asym = abs(dist_l - dist_r) / (dist_l + dist_r + 1e-6)
    if yaw_asym > MAX_YAW_ASYMMETRY:
        return None, f"Non frontale (Yaw asym: {yaw_asym:.2f})"

    # 3. Calcolo PITCH (proporzione verticale occhi-naso rispetto a naso-mento)
    eyes_mid_y = 0.5 * (eye_l[1] + eye_r[1])
    chin_y = pts[152][1]
    nose_ratio = (nose_tip[1] - eyes_mid_y) / (chin_y - eyes_mid_y + 1e-6)
    pitch_dev = abs(nose_ratio - 0.44)  # ~0.44 è il rapporto canonico frontale
    if pitch_dev > MAX_PITCH_DEVIATION:
        return None, f"Mento troppo alto/basso (Pitch dev: {pitch_dev:.2f})"

    # 4. Apertura occhi (EAR medio destro e sinistro)
    ear_r = compute_eye_ear(pts, 159, 145, 133, 33)
    ear_l = compute_eye_ear(pts, 386, 374, 362, 263)
    ear_avg = 0.5 * (ear_r + ear_l)
    if ear_avg < MIN_EYE_OPENNESS:
        return None, f"Occhi socchiusi (EAR: {ear_avg:.2f})"

    # 5. Nitidezza (Laplaciano) calcolata SOLO sul ritaglio del volto normalizzato a 256x256
    x1 = max(0, int(pts[:, 0].min()))
    y1 = max(0, int(pts[:, 1].min()))
    x2 = min(w, int(pts[:, 0].max()))
    y2 = min(h, int(pts[:, 1].max()))
    face_crop = img_bgr[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None, "Crop non valido"

    gray_crop = cv2.cvtColor(cv2.resize(face_crop, (256, 256)), cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

    # 6. Punteggio composito: premia frontalità assoluta, nitidezza e risoluzione del volto
    frontal_score = (1.0 - (yaw_asym / MAX_YAW_ASYMMETRY)) * (1.0 - (pitch_dev / MAX_PITCH_DEVIATION))
    sharpness_norm = min(sharpness / 300.0, 2.0)  # Normalizza il Laplaciano
    resolution_norm = min(face_w / 500.0, 1.5)

    total_score = (frontal_score * 55.0) + (sharpness_norm * 30.0) + (resolution_norm * 15.0)

    return total_score, {
        "yaw": yaw_asym,
        "roll": roll_deg,
        "sharpness": sharpness,
        "face_w": int(face_w)
    }


def get_bucket_key(dt):
    """Raggruppa la data per Anno o per Semestre (S1 = Gen-Giu, S2 = Lug-Dic)."""
    if BUCKET_MODE == "YEAR":
        return f"{dt.year}"
    semester = "S1" if dt.month <= 6 else "S2"
    return f"{dt.year}_{semester}"


def main():
    os.makedirs(SELECTED_DIR, exist_ok=True)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.JPG", "*.JPEG", "*.PNG", "*.HEIC")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(RAW_POOL_DIR, "**", ext), recursive=True))

    files = sorted(set(files))
    if not files:
        print(f"Nessuna immagine trovata in '{RAW_POOL_DIR}'. Controlla il percorso.")
        return

    print(f"Avvio scansione di {len(files)} foto in '{RAW_POOL_DIR}'...\n")

    buckets = {}
    discarded = 0

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        for idx, fpath in enumerate(files, 1):
            try:
                img = load_image_bgr(fpath)
            except Exception:
                discarded += 1
                continue

            score, info = evaluate_face(img, face_mesh)
            if score is None:
                discarded += 1
                if idx % 100 == 0:
                    print(f"  Analizzate {idx}/{len(files)} foto...")
                continue

            dt = get_photo_datetime(fpath)
            b_key = get_bucket_key(dt)
            buckets.setdefault(b_key, []).append((score, dt, fpath, info))

            if idx % 100 == 0:
                print(f"  Analizzate {idx}/{len(files)} foto...")

    print(f"\nScansione terminata! Foto valide: {sum(len(v) for v in buckets.values())} | Scartate: {discarded}")
    print(f"Seleziono le migliori {TOP_K_PER_BUCKET} per ciascuno dei {len(buckets)} periodi trovati:\n")

    copied_count = 0
    for b_key in sorted(buckets.keys()):
        candidates = sorted(buckets[b_key], key=lambda x: x[0], reverse=True)
        winners = candidates[:TOP_K_PER_BUCKET]

        for rank, (score, dt, fpath, info) in enumerate(winners, 1):
            ext = os.path.splitext(fpath)[1].lower()
            date_prefix = dt.strftime("%Y%m%d_%H%M%S")
            out_name = f"{date_prefix}_{b_key}_top{rank}_score{int(score)}{ext}"
            out_path = os.path.join(SELECTED_DIR, out_name)

            shutil.copy2(fpath, out_path)
            copied_count += 1
            print(
                f"  [{b_key} #{rank}] Score: {score:5.1f} | "
                f"Yaw: {info['yaw']:.2f} | Sharp: {info['sharpness']:5.0f} | "
                f"Orig: {os.path.basename(fpath)}"
            )

    print(f"\nOperazione completata: {copied_count} foto copiate e ordinate in '{SELECTED_DIR}'.")


if __name__ == "__main__":
    main()