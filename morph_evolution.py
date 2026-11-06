import os
import glob
import shutil
import subprocess
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mediapipe_tasks
from mediapipe.tasks.python import vision
from PIL import Image, ExifTags
import pillow_heif

# Supporto nativo per i file .HEIC di iPhone/Mac
pillow_heif.register_heif_opener()

# ================= CONFIGURAZIONE =================
INPUT_DIR = "./EvoluzioneAntonia/foto_antonia"       # Cartella con le foto selezionate
ALIGNED_DIR = "./EvoluzioneAntonia/aligned_debug"    # Cartella dove vengono salvati i volti allineati
OUTPUT_VIDEO = "./EvoluzioneAntonia/evoluzione_antonia.mp4"
AUDIO_FILE = "./EvoluzioneAntonia/OST/forever_young.mp3"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_LANDMARKER_MODEL = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

OUT_W, OUT_H = 1080, 1350          # Risoluzione 4:5 (ottima per i ritratti, es. 1080x1350 px)
FPS = 60                           # 60 fps garantisce una fluidità setosa
HOLD_SECONDS = 0.4                 # Pausa sulla foto originale nitida (in secondi)
MORPH_SECONDS = 1.6                # Durata della trasformazione tra due foto (in secondi)

# Posizione fissa degli occhi nel frame finale (in percentuale rispetto a OUT_W e OUT_H)
LEFT_EYE_TARGET = (0.37, 0.40)     # Occhio sinistro nello schermo (destro del soggetto)
RIGHT_EYE_TARGET = (0.63, 0.40)    # Occhio destro nello schermo (sinistro del soggetto)
# ==================================================

# Indici MediaPipe FaceMesh per il contorno dell'iride
RIGHT_IRIS_IDX = [469, 470, 471, 472]
LEFT_IRIS_IDX = [474, 475, 476, 477]

# Sottoinsieme di ~110 landmark chiave (contorno viso, occhi, sopracciglia, naso, labbra)
# Evita triangoli degeneri o sovrapposti rispetto all'uso di tutti i 468 punti
SUBSET_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276,
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    468, 473 # Centri iride
]


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


def get_audio_duration(filepath):
    """Legge la durata della traccia audio in secondi tramite ffprobe."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Traccia audio non trovata: '{filepath}'.")

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(result.stdout.strip())
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe non trovato. Installa FFmpeg con Homebrew.") from exc
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Impossibile leggere la durata di '{filepath}'.") from exc

    if duration <= 0:
        raise RuntimeError(f"Durata audio non valida per '{filepath}'.")
    return duration


def build_timeline(num_images, audio_duration):
    """Distribuisce esattamente i frame audio tra pause e transizioni."""
    weights = []
    for _ in range(num_images - 1):
        weights.extend((HOLD_SECONDS, MORPH_SECONDS))
    weights.append(HOLD_SECONDS * 2.5)

    target_frames = round(audio_duration * FPS)
    if target_frames < len(weights):
        raise RuntimeError("La traccia audio e troppo corta per il numero di foto selezionate.")

    total_weight = sum(weights)
    segment_frames = []
    previous_boundary = 0
    cumulative_weight = 0.0
    for weight in weights:
        cumulative_weight += weight
        boundary = round(target_frames * cumulative_weight / total_weight)
        segment_frames.append(boundary - previous_boundary)
        previous_boundary = boundary

    transition_frames = [
        (segment_frames[i], segment_frames[i + 1])
        for i in range(0, len(segment_frames) - 1, 2)
    ]
    return transition_frames, segment_frames[-1], target_frames


def get_exif_date_or_name(filepath):
    """Ordina le foto per data di scatto EXIF; se assente, usa il nome del file."""
    try:
        img = Image.open(filepath)
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTime"):
                    return (0, str(value))
    except Exception:
        pass
    return (1, os.path.basename(filepath))


def load_image_bgr(filepath):
    """Carica qualsiasi formato (incluso HEIC) e corregge l'orientamento EXIF."""
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


def extract_landmarks(img_bgr, face_landmarker):
    """Estrae i 478 landmark e seleziona il volto principale (più grande)."""
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_landmarker.detect(mp_image)
    if not results.face_landmarks:
        return None

    best_face = None
    max_area = -1
    for face_lms in results.face_landmarks:
        if len(face_lms) <= max(max(SUBSET_INDICES), max(LEFT_IRIS_IDX)):
            continue
        pts = np.array([(lm.x * w, lm.y * h) for lm in face_lms], dtype=np.float32)
        area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        if area > max_area:
            max_area = area
            best_face = pts
    return best_face


def align_face(img_bgr, pts, out_w, out_h):
    """Ruota, scala e trasla la foto fissando le pupille sulle coordinate target."""
    # Occhio sinistro sullo schermo = iride destra anatomica del soggetto
    eye_left_screen = np.mean(pts[RIGHT_IRIS_IDX], axis=0)
    eye_right_screen = np.mean(pts[LEFT_IRIS_IDX], axis=0)

    dY = eye_right_screen[1] - eye_left_screen[1]
    dX = eye_right_screen[0] - eye_left_screen[0]
    angle = np.degrees(np.arctan2(dY, dX))

    dist = np.hypot(dX, dY)
    desired_dist = (RIGHT_EYE_TARGET[0] - LEFT_EYE_TARGET[0]) * out_w
    scale = desired_dist / dist

    eyes_center = (
        float((eye_left_screen[0] + eye_right_screen[0]) * 0.5),
        float((eye_left_screen[1] + eye_right_screen[1]) * 0.5),
    )

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    target_center_x = out_w * (LEFT_EYE_TARGET[0] + RIGHT_EYE_TARGET[0]) * 0.5
    target_center_y = out_h * LEFT_EYE_TARGET[1]
    M[0, 2] += (target_center_x - eyes_center[0])
    M[1, 2] += (target_center_y - eyes_center[1])

    # BORDER_REFLECT_101 evita bordi neri se una foto originale è tagliata stretta
    aligned_img = cv2.warpAffine(
        img_bgr, M, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Trasforma anche i landmark con la stessa matrice M
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_homo = np.hstack([pts, ones])
    aligned_pts = (M @ pts_homo.T).T

    return aligned_img, aligned_pts


def get_boundary_points(w, h):
    """Aggiunge 16 punti fissi lungo il perimetro per deformare anche sfondo e spalle."""
    w1, h1 = w - 1, h - 1
    return np.array([
        [0, 0], [w1 // 4, 0], [w1 // 2, 0], [3 * w1 // 4, 0], [w1, 0],
        [w1, h1 // 4], [w1, h1 // 2], [w1, 3 * h1 // 4], [w1, h1],
        [3 * w1 // 4, h1], [w1 // 2, h1], [w1 // 4, h1], [0, h1],
        [0, 3 * h1 // 4], [0, h1 // 2], [0, h1 // 4]
    ], dtype=np.float32)


def compute_delaunay_triangles(pts, w, h):
    """Calcola la triangolazione di Delaunay e restituisce le triplette di indici."""
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    pts_clipped = np.clip(pts, [0, 0], [w - 1.001, h - 1.001])

    for p in pts_clipped:
        subdiv.insert((float(p[0]), float(p[1])))

    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []

    for t in triangle_list:
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        idx1 = np.argmin(np.linalg.norm(pts_clipped - pt1, axis=1))
        idx2 = np.argmin(np.linalg.norm(pts_clipped - pt2, axis=1))
        idx3 = np.argmin(np.linalg.norm(pts_clipped - pt3, axis=1))
        if len({idx1, idx2, idx3}) == 3:
            delaunay_tri.append((idx1, idx2, idx3))

    return delaunay_tri


def warp_triangle(img1, img2, img_morph, t1, t2, t, alpha):
    """Deforma e fonde un singolo triangolo tra img1 e img2."""
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    if r[2] <= 0 or r[3] <= 0 or r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_rect = np.float32([(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)])
    t2_rect = np.float32([(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)])
    t_rect = np.float32([(t[i][0] - r[0], t[i][1] - r[1]) for i in range(3)])

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    M1 = cv2.getAffineTransform(t1_rect, t_rect)
    M2 = cv2.getAffineTransform(t2_rect, t_rect)

    warp1 = cv2.warpAffine(img1_rect, M1, (r[2], r[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warp2 = cv2.warpAffine(img2_rect, M2, (r[2], r[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    img_rect = (1.0 - alpha) * warp1 + alpha * warp2

    roi = img_morph[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
    img_morph[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = roi * (1.0 - mask) + img_rect * mask


def morph_pair(img1, img2, pts1, pts2, triangles, alpha):
    """Genera un fotogramma intermedio dato alpha in [0, 1]."""
    img1_f = np.float32(img1)
    img2_f = np.float32(img2)
    pts_inter = (1.0 - alpha) * pts1 + alpha * pts2
    img_morph = np.zeros(img1.shape, dtype=np.float32)

    for idx1, idx2, idx3 in triangles:
        t1 = [pts1[idx1], pts1[idx2], pts1[idx3]]
        t2 = [pts2[idx1], pts2[idx2], pts2[idx3]]
        t = [pts_inter[idx1], pts_inter[idx2], pts_inter[idx3]]
        warp_triangle(img1_f, img2_f, img_morph, t1, t2, t, alpha)

    return np.uint8(np.clip(img_morph, 0, 255))


def main():
    os.makedirs(ALIGNED_DIR, exist_ok=True)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.JPG", "*.JPEG", "*.PNG", "*.HEIC")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    files = sorted(set(files), key=get_exif_date_or_name)
    if len(files) < 2:
        print(f"Errore: servono almeno 2 foto nella cartella '{INPUT_DIR}'.")
        return

    try:
        audio_duration = get_audio_duration(AUDIO_FILE)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Errore audio: {exc}")
        return

    audio_minutes, audio_seconds = divmod(audio_duration, 60)
    print(
        f"Traccia audio: {os.path.basename(AUDIO_FILE)} "
        f"({int(audio_minutes)}:{audio_seconds:04.1f})"
    )
    print(f"Trovate {len(files)} immagini. Allineamento volti in corso...")

    aligned_images = []
    aligned_landmarks = []
    boundary_pts = get_boundary_points(OUT_W, OUT_H)

    ensure_face_landmarker_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mediapipe_tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        for idx, fpath in enumerate(files):
            img = load_image_bgr(fpath)
            pts = extract_landmarks(img, face_landmarker)
            if pts is None:
                print(f"  [SCARTATA] Nessun volto rilevato in: {os.path.basename(fpath)}")
                continue

            al_img, al_pts = align_face(img, pts, OUT_W, OUT_H)
            subset_pts = np.vstack([al_pts[SUBSET_INDICES], boundary_pts])

            aligned_images.append(al_img)
            aligned_landmarks.append(subset_pts)

            # Salva la foto allineata per eventuale controllo visivo
            debug_name = f"{idx:03d}_{os.path.splitext(os.path.basename(fpath))[0]}.jpg"
            cv2.imwrite(os.path.join(ALIGNED_DIR, debug_name), al_img)
            print(f"  [OK] Allineata ({ len(aligned_images) }): {os.path.basename(fpath)}")

    if len(aligned_images) < 2:
        print("Non ci sono abbastanza volti validi per creare il video.")
        return

    transition_frames, final_hold_frames, target_frames = build_timeline(
        len(aligned_images), audio_duration
    )
    print(
        f"Timeline sincronizzata: {target_frames} frame, "
        f"{target_frames / FPS:.2f} secondi."
    )

    # Generazione video temporaneo ed encoding H.264/AAC finale
    temp_video = os.path.join(os.path.dirname(OUTPUT_VIDEO), "temp_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, FPS, (OUT_W, OUT_H))
    if not writer.isOpened():
        print(f"Errore: impossibile creare il video temporaneo '{temp_video}'.")
        return

    print("\nGenerazione del morphing geometrico...")
    for i in range(len(aligned_images) - 1):
        img1, img2 = aligned_images[i], aligned_images[i + 1]
        pts1, pts2 = aligned_landmarks[i], aligned_landmarks[i + 1]
        hold_frames, morph_frames = transition_frames[i]

        # Pausa iniziale sulla foto i-esima
        for _ in range(hold_frames):
            writer.write(img1)

        # Calcolo triangolazione di Delaunay sulla geometria media tra i due volti
        pts_mid = 0.5 * (pts1 + pts2)
        triangles = compute_delaunay_triangles(pts_mid, OUT_W, OUT_H)

        # Interpolazione con curva ease-in-out cosenoidale (molto più naturale di quella lineare)
        for f in range(1, morph_frames + 1):
            t_lin = f / (morph_frames + 1)
            alpha = 0.5 * (1.0 - np.cos(np.pi * t_lin))
            frame = morph_pair(img1, img2, pts1, pts2, triangles, alpha)
            writer.write(frame)

        print(f"  Transizione {i + 1}/{len(aligned_images) - 1} completata")

    # Pausa finale sull'ultima foto
    for _ in range(final_hold_frames):
        writer.write(aligned_images[-1])

    writer.release()

    # Ricodifica il video e aggiunge la traccia MP3 come audio AAC.
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", AUDIO_FILE,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            OUTPUT_VIDEO
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_video)
        print(f"\nVideo sincronizzato con l'audio: {OUTPUT_VIDEO}")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"\nErrore FFmpeg: {exc}")
        print(f"Il video temporaneo senza audio e disponibile in: {temp_video}")


if __name__ == "__main__":
    main()
