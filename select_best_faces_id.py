import os
import glob
import shutil
import urllib.request
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mediapipe_tasks
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis
from PIL import Image, ExifTags
import pillow_heif

pillow_heif.register_heif_opener()

# ================= CONFIGURAZIONE =================
RAW_POOL_DIR = "./EvoluzioneAntonia/Photos-1-001"  # Cartella con le ~2000 foto da filtrare
REF_DIR = "./EvoluzioneAntonia/ref_antonia"           # Cartella con 3-4 foto di riferimento di Antonia a varie età
SELECTED_DIR = "./EvoluzioneAntonia/foto_antonia"     # Cartella di output (pronta per morph_evolution.py)

# Il Face Landmarker moderno usa un modello esterno. Se manca, viene scaricato
# automaticamente nella cartella "models" accanto a questo script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_LANDMARKER_MODEL = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

BUCKET_MODE = "HALF_YEAR"           # "YEAR" oppure "HALF_YEAR"
TOP_K_PER_BUCKET = 2                # Quante foto migliori salvare per ogni finestra temporale

# Soglia di similarità coseno ArcFace (range tipico per la stessa persona: 0.35 - 0.85)
# 0.36 è calibrato per tollerare i cambiamenti di crescita pur escludendo sorella/amici
SIMILARITY_THRESHOLD = 0.36

# Soglie geometriche e ottiche
MAX_ROLL_DEG = 15.0                 # Inclinazione laterale massima della testa (in gradi)
MAX_YAW_ASYMMETRY = 0.22            # Rotazione destra/sinistra (0 = perfettamente frontale)
MAX_PITCH_DEVIATION = 0.28          # Inclinazione alto/basso rispetto al piano neutro
MIN_EYE_OPENNESS = 0.18             # Soglia apertura occhi (scarta occhi chiusi)
MIN_FACE_SIZE_PX = 110              # Larghezza minima del volto di Antonia in pixel
# ==================================================

RIGHT_IRIS_IDX = [469, 470, 471, 472]
LEFT_IRIS_IDX = [474, 475, 476, 477]


def ensure_face_landmarker_model():
    """Scarica il modello ufficiale MediaPipe se non e gia presente."""
    if os.path.isfile(FACE_LANDMARKER_MODEL):
        return

    os.makedirs(os.path.dirname(FACE_LANDMARKER_MODEL), exist_ok=True)
    temporary_path = f"{FACE_LANDMARKER_MODEL}.download"
    print("Download del modello MediaPipe Face Landmarker...")

    try:
        with urllib.request.urlopen(FACE_LANDMARKER_MODEL_URL, timeout=60) as response:
            with open(temporary_path, "wb") as output_file:
                shutil.copyfileobj(response, output_file)
        os.replace(temporary_path, FACE_LANDMARKER_MODEL)
    except Exception as exc:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise RuntimeError(
            "Impossibile scaricare il modello Face Landmarker. "
            f"Scaricalo manualmente da {FACE_LANDMARKER_MODEL_URL} e salvalo in "
            f"'{FACE_LANDMARKER_MODEL}'."
        ) from exc


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


def load_reference_embeddings(ref_dir, face_app):
    """Estrae i vettori biometrici normalizzati (512-D) dalle foto campione di Antonia."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.JPG", "*.JPEG", "*.PNG", "*.HEIC")
    ref_files = []
    for ext in extensions:
        ref_files.extend(glob.glob(os.path.join(ref_dir, ext)))

    ref_embeddings = []
    for fpath in sorted(set(ref_files)):
        img = load_image_bgr(fpath)
        faces = face_app.get(img)
        if not faces:
            print(f"  [ATTENZIONE] Nessun volto trovato nel riferimento: {os.path.basename(fpath)}")
            continue
        # Se nella foto di riferimento ci sono più volti, prende il più grande
        main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        ref_embeddings.append(main_face.normed_embedding)
        print(f"  [REF OK] Caricato profilo biometrico da: {os.path.basename(fpath)}")

    return np.array(ref_embeddings)


def compute_eye_ear(pts, top_idx, bottom_idx, inner_idx, outer_idx):
    """Calcola l'Eye Aspect Ratio per verificare che gli occhi siano ben aperti."""
    v_dist = np.linalg.norm(pts[top_idx] - pts[bottom_idx])
    h_dist = np.linalg.norm(pts[inner_idx] - pts[outer_idx]) + 1e-6
    return v_dist / h_dist


def evaluate_antonia_in_photo(img_bgr, face_app, ref_embeddings, face_landmarker):
    """
    1. Cerca tutti i volti nella foto con InsightFace (ArcFace).
    2. Identifica quale volto appartiene ad Antonia confrontando la similarità coseno.
    3. Isola un crop largo intorno al volto di Antonia e calcola su di esso
       i 478 landmark MediaPipe e la nitidezza Laplaciana.
    """
    h, w = img_bgr.shape[:2]
    faces = face_app.get(img_bgr)
    if not faces:
        return None, "Nessun volto rilevato"

    # Trova il volto che ha la massima similarità con almeno una delle foto di riferimento
    best_face = None
    best_sim = -1.0

    for face in faces:
        emb = face.normed_embedding
        # Prodotto scalare tra vettori normalizzati = Similarità Coseno
        sims = np.dot(ref_embeddings, emb)
        max_sim_for_face = float(np.max(sims))
        if max_sim_for_face > best_sim:
            best_sim = max_sim_for_face
            best_face = face

    if best_sim < SIMILARITY_THRESHOLD:
        return None, f"Antonia non presente (Max Sim: {best_sim:.2f})"

    # Coordinate del bounding box di Antonia trovato da InsightFace
    bx1, by1, bx2, by2 = best_face.bbox.astype(int)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < MIN_FACE_SIZE_PX:
        return None, f"Volto di Antonia troppo piccolo ({bw} px)"

    # Ritaglia un'area con margine del 40% intorno al viso di Antonia per passarla a MediaPipe
    # Così MediaPipe analizza sempre e solo Antonia anche nelle foto di gruppo
    pad_x, pad_y = int(bw * 0.4), int(bh * 0.4)
    cx1, cy1 = max(0, bx1 - pad_x), max(0, by1 - pad_y)
    cx2, cy2 = min(w, bx2 + pad_x), min(h, by2 + pad_y)
    antonia_roi = img_bgr[cy1:cy2, cx1:cx2]

    roi_h, roi_w = antonia_roi.shape[:2]
    rgb_roi = np.ascontiguousarray(cv2.cvtColor(antonia_roi, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
    results = face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None, "Landmark MediaPipe non estratti sul crop"

    # Prende la mesh principale nel ritaglio centrato su Antonia
    landmarks = results.face_landmarks[0]
    if len(landmarks) <= max(LEFT_IRIS_IDX):
        return None, f"Mesh MediaPipe incompleta ({len(landmarks)} landmark)"

    pts = np.array(
        [(lm.x * roi_w, lm.y * roi_h) for lm in landmarks],
        dtype=np.float32
    )

    # 1. Controllo ROLL
    eye_l = np.mean(pts[RIGHT_IRIS_IDX], axis=0)
    eye_r = np.mean(pts[LEFT_IRIS_IDX], axis=0)
    roll_deg = abs(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])))
    if roll_deg > MAX_ROLL_DEG:
        return None, f"Testa inclinata (Roll: {roll_deg:.1f}°)"

    # 2. Controllo YAW (frontalità orizzontale)
    nose_tip = pts[1]
    dist_l = np.linalg.norm(nose_tip - pts[234])
    dist_r = np.linalg.norm(nose_tip - pts[454])
    yaw_asym = abs(dist_l - dist_r) / (dist_l + dist_r + 1e-6)
    if yaw_asym > MAX_YAW_ASYMMETRY:
        return None, f"Non frontale (Yaw: {yaw_asym:.2f})"

    # 3. Controllo PITCH (inclinazione verticale)
    eyes_mid_y = 0.5 * (eye_l[1] + eye_r[1])
    chin_y = pts[152][1]
    nose_ratio = (nose_tip[1] - eyes_mid_y) / (chin_y - eyes_mid_y + 1e-6)
    pitch_dev = abs(nose_ratio - 0.44)
    if pitch_dev > MAX_PITCH_DEVIATION:
        return None, f"Mento alto/basso (Pitch: {pitch_dev:.2f})"

    # 4. Apertura occhi (EAR)
    ear_r = compute_eye_ear(pts, 159, 145, 133, 33)
    ear_l = compute_eye_ear(pts, 386, 374, 362, 263)
    if 0.5 * (ear_r + ear_l) < MIN_EYE_OPENNESS:
        return None, "Occhi socchiusi"

    # 5. Nitidezza (Laplaciano) sul box stretto del viso di Antonia
    fx1, fy1 = max(0, bx1), max(0, by1)
    fx2, fy2 = min(w, bx2), min(h, by2)
    face_tight = img_bgr[fy1:fy2, fx1:fx2]
    gray_crop = cv2.cvtColor(cv2.resize(face_tight, (256, 256)), cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

    # 6. Punteggio finale (combina frontalità, nitidezza, certezza dell'identità e risoluzione)
    frontal_score = (1.0 - (yaw_asym / MAX_YAW_ASYMMETRY)) * (1.0 - (pitch_dev / MAX_PITCH_DEVIATION))
    sharpness_norm = min(sharpness / 300.0, 2.0)
    sim_bonus = min((best_sim - SIMILARITY_THRESHOLD) / 0.4, 1.0)
    res_norm = min(bw / 500.0, 1.5)

    total_score = (frontal_score * 45.0) + (sharpness_norm * 30.0) + (sim_bonus * 15.0) + (res_norm * 10.0)

    return total_score, {
        "sim": best_sim,
        "yaw": yaw_asym,
        "sharpness": sharpness,
        "face_w": int(bw)
    }


def get_bucket_key(dt):
    if BUCKET_MODE == "YEAR":
        return f"{dt.year}"
    semester = "S1" if dt.month <= 6 else "S2"
    return f"{dt.year}_{semester}"


def main():
    os.makedirs(SELECTED_DIR, exist_ok=True)
    if not os.path.isdir(RAW_POOL_DIR):
        print(f"Errore: cartella delle foto non trovata: '{RAW_POOL_DIR}'.")
        return

    if not os.path.exists(REF_DIR):
        os.makedirs(REF_DIR, exist_ok=True)
        print(f"Cartella '{REF_DIR}' creata. Inserisci 3-4 foto di Antonia a varie età e rilancia.")
        return

    print("Inizializzazione InsightFace (ArcFace) su Apple Silicon...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    ref_embeddings = load_reference_embeddings(REF_DIR, face_app)
    if len(ref_embeddings) == 0:
        print(f"Errore: nessuna foto di riferimento valida trovata in '{REF_DIR}'.")
        return

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.JPG", "*.JPEG", "*.PNG", "*.HEIC")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(RAW_POOL_DIR, "**", ext), recursive=True))
    files = sorted(set(files))

    if not files:
        print(f"Errore: nessuna foto supportata trovata in '{RAW_POOL_DIR}'.")
        return

    print(f"\nAvvio scansione biometrica su {len(files)} foto...\n")
    buckets = {}
    discarded = 0

    ensure_face_landmarker_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mediapipe_tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        for idx, fpath in enumerate(files, 1):
            try:
                img = load_image_bgr(fpath)
            except Exception:
                discarded += 1
                continue

            score, info = evaluate_antonia_in_photo(
                img, face_app, ref_embeddings, face_landmarker
            )
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

    print(f"\nScansione conclusa! Foto valide di Antonia: {sum(len(v) for v in buckets.values())} | Scartate: {discarded}\n")

    copied_count = 0
    for b_key in sorted(buckets.keys()):
        winners = sorted(buckets[b_key], key=lambda x: x[0], reverse=True)[:TOP_K_PER_BUCKET]
        for rank, (score, dt, fpath, info) in enumerate(winners, 1):
            ext = os.path.splitext(fpath)[1].lower()
            out_name = f"{dt.strftime('%Y%m%d_%H%M%S')}_{b_key}_top{rank}_sim{int(info['sim']*100)}{ext}"
            shutil.copy2(fpath, os.path.join(SELECTED_DIR, out_name))
            copied_count += 1
            print(
                f"  [{b_key} #{rank}] Score: {score:5.1f} | Sim: {info['sim']:.2f} | "
                f"Yaw: {info['yaw']:.2f} | Sharp: {info['sharpness']:5.0f} | {os.path.basename(fpath)}"
            )

    print(f"\nCompletato! {copied_count} foto di Antonia copiate in '{SELECTED_DIR}'.")


if __name__ == "__main__":
    main()
