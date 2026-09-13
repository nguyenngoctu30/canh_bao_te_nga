"""Fall detection server: RTSP -> YOLO26-Pose -> MJPEG/API (PTZ optional).
"""

import os
import time
import json
import math
import threading
import queue
from collections import deque
from datetime import timedelta, datetime
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask, Response, request, jsonify, session, redirect,
    url_for, send_from_directory, render_template
)
from flask_cors import CORS
from ultralytics import YOLO

# =====================================================================
# CONFIG
# =====================================================================
# Co the set RTSP_URL day du (uu tien), hoac chi sua IP/USER/PASS/PATH ben duoi.
_RTSP_IP   = os.environ.get("RTSP_IP", "192.168.1.8")
_RTSP_PORT = os.environ.get("RTSP_PORT", "554")
_RTSP_USER = os.environ.get("RTSP_USER", "admin")
_RTSP_PASS = os.environ.get("RTSP_PASS", "JuLdN5Qv")
# Duong dan sau cong 554 — TUY CAMERA, khong phai camera nao cung nhu nhau.
# Neu chua biet path dung, chay file test_rtsp.py truoc de do tu dong.
_RTSP_PATH = os.environ.get("RTSP_PATH", "/live/ch00_0")

RTSP_URL = os.environ.get(
    "RTSP_URL",
    f"rtsp://{_RTSP_USER}:{_RTSP_PASS}@{_RTSP_IP}:{_RTSP_PORT}{_RTSP_PATH}",
)

# Timeout ket noi RTSP (giay). Mac dinh FFmpeg cho phep toi 30s truoc khi bao
# "Stream timeout" — qua lau khi dang do sai URL. Giam xuong de phat hien
# nhanh cau hinh sai va thu lai som hon.
RTSP_CONNECT_TIMEOUT_SEC = int(os.environ.get("RTSP_CONNECT_TIMEOUT_SEC", "6"))

MODEL_PATH = os.environ.get("MODEL_PATH", "yolo26n-pose.pt")   # yolo26n/s/m/l/x-pose.pt
TRACKER_CFG = "bytetrack.yaml"          # co san trong ultralytics, khong can file rieng
CONF_THRES = 0.45
IOU_THRES = 0.5
TARGET_FPS = 15                          # fps xu ly/hien thi (khong phai fps model)
RESIZE_TO = (960, 540)                   # None neu muon giu nguyen do phan giai goc

# Nguong phat hien te nga
FALL_ASPECT_RATIO   = 1.35   # bbox.width / bbox.height >= nguong -> nam ngang
FALL_ANGLE_DEG       = 55     # goc than (vai-hong) so voi phuong thang dung >= nguong -> nam
FALL_CONFIRM_FRAMES  = 12     # so frame lien tiep thoa dieu kien moi bao "TE NGA" (chong bao gia)
FALL_RECOVER_FRAMES  = 20     # so frame lien tiep KHONG thoa dieu kien moi go trang thai
TRACK_TIMEOUT_SEC    = 5.0    # qua thoi gian nay khong thay ID -> xoa khoi bo nho

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ALERT_DIR  = os.path.join(BASE_DIR, "alerts")
ALERTS_LOG = os.path.join(ALERT_DIR, "alerts.json")
os.makedirs(ALERT_DIR, exist_ok=True)

WEB_USERNAME = os.environ.get("WEB_USERNAME", "admin")
WEB_PASSWORD = os.environ.get("WEB_PASSWORD", "changeme123")
# Optional public API key to allow cross-origin static frontend to access
# endpoints without session cookies. Set PUBLIC_API_KEY in env to enable.
PUBLIC_API_KEY = os.environ.get("PUBLIC_API_KEY")

# ---- Dieu khien goc camera (PTZ qua ONVIF) ----
# Nhieu camera PTZ dung chung IP nhung PORT ONVIF khac PORT RTSP (554).
# Neu camera khong ho tro PTZ/ONVIF, cu de ONVIF_ENABLED=false — web van
# chay binh thuong, chi an di khoi joystick.
ONVIF_ENABLED  = os.environ.get("ONVIF_ENABLED", "true").lower() in ("1", "true", "yes")
ONVIF_IP       = os.environ.get("ONVIF_IP", _RTSP_IP)
ONVIF_PORT     = int(os.environ.get("ONVIF_PORT", "8899"))
ONVIF_USERNAME = os.environ.get("ONVIF_USERNAME", _RTSP_USER)
ONVIF_PASSWORD = os.environ.get("ONVIF_PASSWORD", _RTSP_PASS)
PTZ_DEFAULT_SPEED = 0.5

# ---- Zoom so (crop + resize tren frame, khong can camera ho tro zoom quang hoc) ----
MAX_DIGITAL_ZOOM = 4.0
ZOOM_STEP = 0.2

# TCP + stimeout de khong treo 30s khi ket noi. LUU Y: truoc day chuoi nay
# co them buffer_size;0|max_delay;0|reorder_queue_size;0 — cac co nay tren
# mot so ban FFmpeg di kem OpenCV (vi du tren Windows) gay treo qua trinh mo
# luong toi khi het stimeout thay vi bao loi ngay. Da rut gon lai con bo
# options da kiem chung hoat dong on dinh.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    f"|stimeout;{RTSP_CONNECT_TIMEOUT_SEC * 1_000_000}"
    "|fflags;nobuffer"
)

# =====================================================================
# FLASK
# =====================================================================
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
app.permanent_session_lifetime = timedelta(days=7)

# Enable CORS for frontend hosting (e.g., Vercel). If using credentials, set
# a specific origin and supports_credentials=True. For simple public API key
# usage we allow all origins here.
CORS(app)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Allow session-based login
        if session.get("logged_in"):
            return f(*args, **kwargs)
        # Allow API key in header or query string for static frontends
        key = None
        if PUBLIC_API_KEY:
            key = request.headers.get('X-API-KEY') or request.args.get('api_key')
            if key == PUBLIC_API_KEY:
                return f(*args, **kwargs)
        if request.path in ("/", "/video_feed"):
            return redirect(url_for("login_page", next=request.path))
        return jsonify({"status": "error", "message": "Chua dang nhap"}), 401
    return decorated


# =====================================================================
# COCO-17 KEYPOINTS (chuan YOLO-Pose) + skeleton de ve
# =====================================================================
KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]
KP_CONF_MIN = 0.30

# =====================================================================
# MODEL — nap YOLO26-Pose o thread rieng, khong chan khoi dong server
# =====================================================================
MODEL_OK = False
model = None


def _load_model():
    global model, MODEL_OK
    try:
        print(f"[YOLO] Dang nap model {MODEL_PATH} ...")
        model = YOLO(MODEL_PATH)
        # warm-up 1 lan cho GPU/CPU "nong may"
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        MODEL_OK = True
        print("[YOLO] San sang.")
    except Exception as e:
        print(f"[YOLO] Loi nap model: {e}")


threading.Thread(target=_load_model, daemon=True, name="yolo-loader").start()


# =====================================================================
# ONVIF / PTZ — dieu khien goc camera tu web (joystick)
# =====================================================================
PTZ_OK = False
ptz_svc = None
token = None


def _onvif_connect():
    global PTZ_OK, ptz_svc, token
    if not ONVIF_ENABLED:
        print("[ONVIF] Da tat trong cau hinh (ONVIF_ENABLED=false) — bo qua PTZ.")
        return
    try:
        from onvif import ONVIFCamera
    except ImportError:
        print("[ONVIF] Chua cai thu vien 'onvif-zeep'. Chay: pip install onvif-zeep")
        print("[ONVIF] Web van chay binh thuong, chi khong dieu khien duoc goc camera.")
        return
    try:
        # Try to detect wsdl folder from installed onvif package and pass it
        wsdl_dir = None
        try:
            import onvif as _onvif_mod
            candidate = os.path.join(os.path.dirname(_onvif_mod.__file__), 'wsdl')
            if os.path.isdir(candidate):
                wsdl_dir = candidate
        except Exception:
            wsdl_dir = None

        if wsdl_dir:
            cam = ONVIFCamera(ONVIF_IP, ONVIF_PORT, ONVIF_USERNAME, ONVIF_PASSWORD, wsdl_dir=wsdl_dir)
        else:
            cam = ONVIFCamera(ONVIF_IP, ONVIF_PORT, ONVIF_USERNAME, ONVIF_PASSWORD)
        media = cam.create_media_service()
        # create PTZ service and token
        ptz = cam.create_ptz_service()
        tkn = media.GetProfiles()[0].token
        # assign to globals
        globals()['ptz_svc'] = ptz
        globals()['token'] = tkn
        PTZ_OK = True
        print(f"[ONVIF] Da ket noi PTZ tai {ONVIF_IP}:{ONVIF_PORT}")
    except Exception as e:
        print(f"[ONVIF] Khong ket noi duoc PTZ ({ONVIF_IP}:{ONVIF_PORT}): {e}")
        print("[ONVIF] Kiem tra lai ONVIF_PORT — thuong khac cong RTSP (554).")


threading.Thread(target=_onvif_connect, daemon=True, name="onvif-connect").start()

# Worker rieng gui lenh PTZ — route HTTP chi ghi "van toc muon", worker doc va
# gui ONVIF, tach hoan toan khoi vong doi request HTTP (phan hoi ngay lap tuc).
_ptz_lock = threading.Lock()
_ptz_vx = 0.0
_ptz_vy = 0.0
_ptz_event = threading.Event()


def _ptz_worker():
    sent_vx = sent_vy = 0.0
    IDLE_TIMEOUT = 0.3
    last_nonzero_t = 0.0

    while True:
        _ptz_event.wait(timeout=IDLE_TIMEOUT)
        _ptz_event.clear()

        if not PTZ_OK:
            continue

        with _ptz_lock:
            vx, vy = _ptz_vx, _ptz_vy

        now = time.monotonic()
        moving = (vx != 0.0 or vy != 0.0)
        if moving:
            last_nonzero_t = now
        idle = (now - last_nonzero_t) >= IDLE_TIMEOUT

        if (vx, vy) == (sent_vx, sent_vy):
            continue

        if idle and vx == 0.0 and vy == 0.0:
            if sent_vx != 0.0 or sent_vy != 0.0:
                try:
                    if ptz_svc is not None and token is not None:
                        ptz_svc.Stop({"ProfileToken": token})
                except Exception as e:
                    print(f"[PTZ] Loi Stop: {e}")
                sent_vx = sent_vy = 0.0
            continue
        try:
            if ptz_svc is not None and token is not None:
                req = ptz_svc.create_type("ContinuousMove")
                req.ProfileToken = token
                req.Velocity = {"PanTilt": {"x": vx, "y": vy}}
                ptz_svc.ContinuousMove(req)
                sent_vx, sent_vy = vx, vy
        except Exception as e:
            print(f"[PTZ] Loi ContinuousMove: {e}")


threading.Thread(target=_ptz_worker, daemon=True, name="ptz-worker").start()


def ptz_set(vx: float, vy: float):
    """Cập nhật target velocity + wake worker ngay lập tức.

    vx, vy: float in [-1.0, 1.0] representing pan (x) và tilt (y) velocities.
    """
    global _ptz_vx, _ptz_vy
    with _ptz_lock:
        _ptz_vx, _ptz_vy = float(vx), float(vy)
    try:
        print(f"[PTZ] set vx={_ptz_vx:.3f}, vy={_ptz_vy:.3f}")
    except Exception:
        pass
    # Wake worker thread to process the new velocity immediately
    _ptz_event.set()


def ptz_move(x=0.0, y=0.0):
    """Alias to set velocity (x=pan, y=tilt)."""
    ptz_set(x, y)


def ptz_stop():
    """Alias to stop movement immediately."""
    ptz_set(0.0, 0.0)


# =====================================================================
# FALL DETECTION — trang thai theo tung track ID
# =====================================================================
_track_lock = threading.Lock()
_track_state: dict[int, dict] = {}
# moi ID: {fall_count, recover_count, is_fallen, fall_since, last_seen}


def _angle_from_vertical(p_top, p_bottom):
    """Goc (do) cua doan thang p_top->p_bottom so voi phuong thang dung.
    ~0 do = dung thang, ~90 do = nam ngang."""
    dx = p_bottom[0] - p_top[0]
    dy = p_bottom[1] - p_top[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def _midpoint_if_visible(a, b):
    if a[2] >= KP_CONF_MIN and b[2] >= KP_CONF_MIN:
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return None


def _evaluate_fall(track_id, bbox, keypoints):
    """Tra ve (is_fallen, newly_triggered, aspect_ratio, torso_angle)."""
    x1, y1, x2, y2 = bbox
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    aspect = w / h

    angle = None
    if keypoints is not None:
        shoulder_mid = _midpoint_if_visible(keypoints[5], keypoints[6])
        hip_mid = _midpoint_if_visible(keypoints[11], keypoints[12])
        if shoulder_mid and hip_mid:
            angle = _angle_from_vertical(shoulder_mid, hip_mid)

    fallen_now = (aspect >= FALL_ASPECT_RATIO) or (angle is not None and angle >= FALL_ANGLE_DEG)

    with _track_lock:
        st = _track_state.setdefault(track_id, {
            "fall_count": 0, "recover_count": 0,
            "is_fallen": False, "fall_since": None,
        })
        st["last_seen"] = time.time()

        if fallen_now:
            st["fall_count"] += 1
            st["recover_count"] = 0
        else:
            st["recover_count"] += 1
            st["fall_count"] = 0

        newly_triggered = False
        if not st["is_fallen"] and st["fall_count"] >= FALL_CONFIRM_FRAMES:
            st["is_fallen"] = True
            st["fall_since"] = time.time()
            newly_triggered = True
        elif st["is_fallen"] and st["recover_count"] >= FALL_RECOVER_FRAMES:
            st["is_fallen"] = False
            st["fall_since"] = None

        return st["is_fallen"], newly_triggered, round(aspect, 2), (round(angle, 1) if angle is not None else None)


def _purge_stale_tracks():
    now = time.time()
    with _track_lock:
        stale = [tid for tid, st in _track_state.items() if now - st.get("last_seen", now) > TRACK_TIMEOUT_SEC]
        for tid in stale:
            del _track_state[tid]


# =====================================================================
# PHAN TICH DANG NGUOI (posture) — chi tiet hon co/khong te nga
# Dung cho hien thi web: Dung / Ngoi / Nam-te nga / Khong ro
# =====================================================================
def _classify_posture(is_fallen: bool, aspect: float, angle, bbox, keypoints):
    """Phan loai dang nguoi de hien thi chi tiet tren web.
    is_fallen: ket qua da qua debounce (dung de bao dong, giu nguyen).
    Ham nay chi phuc vu HIEN THI, khong anh huong logic canh bao."""
    if is_fallen:
        return "Nam / Te nga"

    if angle is not None and angle >= 35:
        return "Cui / Nghieng nguoi"

    # Uoc luong Ngoi: hong va goi gan nhau theo truc doc (dui gap lai)
    # so voi chieu cao khung — dac trung cua tu the ngoi.
    if keypoints is not None:
        x1, y1, x2, y2 = bbox
        h = max(y2 - y1, 1)
        hip_mid = _midpoint_if_visible(keypoints[11], keypoints[12])
        knee_mid = _midpoint_if_visible(keypoints[13], keypoints[14])
        if hip_mid and knee_mid:
            vertical_gap = abs(knee_mid[1] - hip_mid[1])
            if vertical_gap < 0.28 * h:
                return "Ngoi"

    if angle is None and aspect < 0.9:
        # Khong du keypoint dang tin cay nhung bbox cao/hep -> nhieu kha nang dung
        return "Dung"

    return "Dung"


# =====================================================================
# ALERTS — luu anh + log khi phat hien te nga
# =====================================================================
_alerts_lock = threading.Lock()
_alerts: deque = deque(maxlen=300)


def _load_alerts():
    global _alerts
    if os.path.exists(ALERTS_LOG):
        try:
            with open(ALERTS_LOG, "r", encoding="utf-8") as f:
                _alerts = deque(json.load(f), maxlen=300)
        except Exception as e:
            print(f"[ALERTS] Khong doc duoc log cu: {e}")


def _save_alerts():
    try:
        with open(ALERTS_LOG, "w", encoding="utf-8") as f:
            json.dump(list(_alerts), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ALERTS] Khong ghi duoc log: {e}")


_load_alerts()


def _record_alert(track_id, frame_bgr):
    ts = datetime.now()
    filename = f"fall_id{track_id}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(os.path.join(ALERT_DIR, filename), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    entry = {"id": int(track_id), "time": ts.strftime("%Y-%m-%d %H:%M:%S"), "filename": filename}
    with _alerts_lock:
        _alerts.appendleft(entry)
        _save_alerts()
    print(f"[ALERT] Phat hien TE NGA — ID {track_id} luc {entry['time']}")
    return entry


# =====================================================================
# SHARED STATE (camera + ket qua nhan dien moi nhat)
# =====================================================================
state_lock = threading.Lock()
connected = False
last_frame_time = 0.0
frame_timestamps: deque = deque(maxlen=60)
persons_state: list = []     # danh sach nguoi o frame moi nhat, dung cho /status
zoom_level = 1.0
ptz_speed = PTZ_DEFAULT_SPEED

# (No ngrok variables here)

cap_lock = threading.Lock()
cap = None

_clients_lock = threading.Lock()
_clients: list[queue.Queue] = []


def _broadcast(jpeg: bytes):
    with _clients_lock:
        dead = []
        for q in _clients:
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(jpeg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _clients.remove(q)


_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 78]
_FRAME_INTERVAL = 1.0 / TARGET_FPS


def apply_zoom(img, zoom):
    if zoom <= 1.0:
        return img
    h, w = img.shape[:2]
    nw = max(int(w / zoom), 10)
    nh = max(int(h / zoom), 10)
    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    return cv2.resize(img[y1:y1 + nh, x1:x1 + nw], (w, h))


def _draw_overlay(img, persons, fps):
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        color = (0, 0, 255) if p["fallen"] else (0, 200, 90)   # BGR: do neu te nga

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f'ID {p["id"]}' + ("  TE NGA!" if p["fallen"] else f'  {p.get("posture","")}')
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly1 = max(y1 - th - 8, 0)
        cv2.rectangle(img, (x1, ly1), (x1 + tw + 8, ly1 + th + 8), color, -1)
        cv2.putText(img, label, (x1 + 4, ly1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

        kpts = p.get("keypoints")
        if kpts:
            for a, b in SKELETON:
                if kpts[a][2] >= KP_CONF_MIN and kpts[b][2] >= KP_CONF_MIN:
                    pa = (int(kpts[a][0]), int(kpts[a][1]))
                    pb = (int(kpts[b][0]), int(kpts[b][1]))
                    cv2.line(img, pa, pb, (255, 180, 0), 2, cv2.LINE_AA)
            for kp in kpts:
                if kp[2] >= KP_CONF_MIN:
                    cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1, cv2.LINE_AA)

    # HUD goc tren trai
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Nguoi: {len(persons)}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
    if any(p["fallen"] for p in persons):
        cv2.putText(img, "CANH BAO TE NGA", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def _run_yolo_track(img):
    """Chay YOLO26-Pose + tracking tren 1 frame, tra ve list persons (dict)."""
    persons = []
    if not MODEL_OK:
        return persons
    try:
        results = model.track(
            img, persist=True, tracker=TRACKER_CFG,
            conf=CONF_THRES, iou=IOU_THRES, verbose=False,
        )
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return persons

        ids = r.boxes.id.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None

        kpts_xy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        kpts_conf = (r.keypoints.conf.cpu().numpy()
                     if (r.keypoints is not None and r.keypoints.conf is not None) else None)

        for i, tid in enumerate(ids):
            tid = int(tid)
            bbox = tuple(int(v) for v in boxes[i])

            kp = None
            if kpts_xy is not None:
                n = kpts_xy.shape[1]
                conf_arr = kpts_conf[i] if kpts_conf is not None else np.ones(n)
                kp = [(float(kpts_xy[i][j][0]), float(kpts_xy[i][j][1]), float(conf_arr[j]))
                      for j in range(n)]

            is_fallen, newly, aspect, angle = _evaluate_fall(tid, bbox, kp)
            posture = _classify_posture(is_fallen, aspect, angle, bbox, kp)

            persons.append({
                "id": tid,
                "bbox": list(bbox),
                "keypoints": kp,
                "fallen": is_fallen,
                "posture": posture,
                "aspect": aspect,
                "angle": angle,
                "det_conf": round(float(confs[i]), 2) if confs is not None else None,
            })

            if newly:
                _record_alert(tid, img.copy())

    except Exception as e:
        print(f"[YOLO] Loi khi inference/tracking: {e}")

    return persons


def camera_loop():
    """Vong lap chinh: doc RTSP -> YOLO tracking -> fall-detect -> ve overlay -> broadcast."""
    global cap, connected, last_frame_time, persons_state

    frame_count = 0
    was_connected = False
    last_log_t = 0.0

    while True:
        # ---- Ket noi RTSP ----
        with cap_lock:
            need_reconnect = (cap is None or not cap.isOpened())

        if need_reconnect:
            with state_lock:
                connected = False
            was_connected = False
            safe_url = RTSP_URL.replace(_RTSP_PASS, "****") if _RTSP_PASS else RTSP_URL
            print(f"[CAM] Dang ket noi RTSP: {safe_url}")
            c = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            with cap_lock:
                cap = c
            if not c.isOpened():
                print(
                    "[CAM] Khong mo duoc luong RTSP. Kiem tra: (1) URL/path dung camera "
                    "chua — chay test_rtsp.py de do; (2) camera va may chay app.py cung mang; "
                    "(3) RTSP da duoc bat trong app dieu khien camera chua."
                )
                time.sleep(2)
            else:
                print("[CAM] Da mo duoc luong RTSP, dang cho frame dau tien...")
            continue

        # ---- Xa buffer cu (giam do tre) ----
        grabbed = False
        with cap_lock:
            for _ in range(4):
                if cap.grab():
                    grabbed = True
                else:
                    break

        if not grabbed:
            print("[CAM] Grab that bai — ket noi lai...")
            with state_lock:
                connected = False
            with cap_lock:
                if cap:
                    cap.release()
                cap = None
            time.sleep(1)
            continue

        with cap_lock:
            ok, img = cap.retrieve()
        if not ok or img is None:
            continue

        loop_start = time.monotonic()

        with state_lock:
            connected = True
            last_frame_time = time.time()
            frame_timestamps.append(last_frame_time)

        if not was_connected:
            print("[CAM] Da nhan duoc frame dau tien — camera dang chay binh thuong.")
            was_connected = True

        with state_lock:
            zl = zoom_level
        img = apply_zoom(img, zl)

        if RESIZE_TO:
            img = cv2.resize(img, RESIZE_TO, interpolation=cv2.INTER_LINEAR)

        persons = _run_yolo_track(img)

        with state_lock:
            persons_state = persons
            ts = list(frame_timestamps)

        fps = 0.0
        if len(ts) >= 2:
            span = ts[-1] - ts[0]
            if span > 0:
                fps = round((len(ts) - 1) / span, 1)

        now_t = time.time()
        if now_t - last_log_t >= 10:
            print(f"[CAM] Dang chay — FPS: {fps} | So nguoi: {len(persons)} | "
                  f"Te nga: {sum(1 for p in persons if p['fallen'])}")
            last_log_t = now_t

        img = _draw_overlay(img, persons, fps)

        ret, buf = cv2.imencode(".jpg", img, _ENCODE_PARAMS)
        if ret:
            _broadcast(buf.tobytes())

        frame_count += 1
        if frame_count % 30 == 0:
            _purge_stale_tracks()

        elapsed = time.monotonic() - loop_start
        wait = _FRAME_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)


threading.Thread(target=camera_loop, daemon=True, name="cam-loop").start()


# =====================================================================
# MJPEG GENERATOR (moi client 1 queue rieng)
# =====================================================================
def generate():
    q: queue.Queue = queue.Queue(maxsize=3)
    with _clients_lock:
        _clients.append(q)
    try:
        while True:
            try:
                jpeg = q.get(timeout=5.0)
            except queue.Empty:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
    finally:
        with _clients_lock:
            if q in _clients:
                _clients.remove(q)


# =====================================================================
# ROUTES — AUTH
# =====================================================================
@app.route("/login", methods=["GET", "POST"])
def login_page():
    error = None
    if request.method == "POST":
        if (request.form.get("username") == WEB_USERNAME and
                request.form.get("password") == WEB_PASSWORD):
            session.permanent = True
            session["logged_in"] = True
            return redirect(request.args.get("next") or url_for("index"))
        error = "Sai ten dang nhap hoac mat khau"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# =====================================================================
# ROUTES — TRANG CHINH & VIDEO
# =====================================================================
@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/video_feed")
@login_required
def video_feed():
    resp = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


# =====================================================================
# ROUTES — API TRANG THAI
# =====================================================================
@app.route("/status")
@login_required
def status():
    now = time.time()
    with state_lock:
        conn = connected
        lft = last_frame_time
        persons = list(persons_state)
        ts = list(frame_timestamps)

    fps = 0.0
    if len(ts) >= 2:
        span = ts[-1] - ts[0]
        if span > 0:
            fps = round((len(ts) - 1) / span, 1)

    age = (now - lft) if lft else None
    online = conn and age is not None and age < 3

    with state_lock:
        zl = zoom_level
        spd = ptz_speed

    out_persons = [{
        "id": p["id"],
        "bbox": p["bbox"],
        "fallen": p["fallen"],
        "posture": p.get("posture", "Khong ro"),
        "aspect_ratio": p["aspect"],
        "torso_angle": p["angle"],
        "det_conf": p["det_conf"],
        "keypoints": p["keypoints"],   # [[x,y,conf], ...] 17 diem — de web ve neu can
    } for p in persons]

    return jsonify({
        "online": online,
        "model_ok": MODEL_OK,
        "ptz_ok": PTZ_OK,
        "zoom": zl,
        "speed": spd,
        "fps": fps,
        "frame_age_ms": round(age * 1000) if age else None,
        "person_count": len(persons),
        "fall_count": sum(1 for p in persons if p["fallen"]),
        "persons": out_persons,
    })


@app.route("/reconnect")
@login_required
def reconnect():
    global cap
    with cap_lock:
        if cap:
            cap.release()
        cap = None
    # Also attempt ONVIF/PTZ reconnect in background (useful after installing onvif libs)
    def _try_onvif():
        try:
            _onvif_connect()
        except Exception as e:
            print(f"[ONVIF] Reconnect attempt failed: {e}")

    threading.Thread(target=_try_onvif, daemon=True, name="onvif-reconnect").start()
    return jsonify({"status": "reconnecting"})


# =====================================================================
# ROUTES — DIEU KHIEN GOC CAMERA (PTZ) + ZOOM
# =====================================================================
@app.route("/joystick")
@login_required
def joystick():
    try:
        x = max(-1.0, min(1.0, float(request.args.get("x", 0))))
        y = max(-1.0, min(1.0, float(request.args.get("y", 0))))
    except (TypeError, ValueError):
        x = y = 0.0
    with state_lock:
        spd = ptz_speed
    ptz_set(x * spd, y * spd)
    return "", 204


@app.route("/stop")
@login_required
def stop_ptz():
    ptz_set(0.0, 0.0)
    return "", 204


@app.route("/set_speed")
@login_required
def set_speed():
    global ptz_speed
    try:
        v = float(request.args.get("value", PTZ_DEFAULT_SPEED))
    except (TypeError, ValueError):
        v = PTZ_DEFAULT_SPEED
    v = max(0.05, min(v, 1.0))
    with state_lock:
        ptz_speed = v
    return jsonify({"speed": v})


@app.route("/zoom_in")
@login_required
def zoom_in():
    global zoom_level
    with state_lock:
        zoom_level = min(round(zoom_level + ZOOM_STEP, 2), MAX_DIGITAL_ZOOM)
        zl = zoom_level
    return jsonify({"zoom": zl})


@app.route("/zoom_out")
@login_required
def zoom_out():
    global zoom_level
    with state_lock:
        zoom_level = max(round(zoom_level - ZOOM_STEP, 2), 1.0)
        zl = zoom_level
    return jsonify({"zoom": zl})


@app.route("/zoom_reset")
@login_required
def zoom_reset():
    global zoom_level
    with state_lock:
        zoom_level = 1.0
    return jsonify({"zoom": 1.0})


# =====================================================================
# ROUTES — LICH SU CANH BAO TE NGA
# =====================================================================
@app.route("/alerts")
@login_required
def list_alerts():
    with _alerts_lock:
        return jsonify(list(_alerts))


@app.route("/alerts/<path:filename>")
@login_required
def get_alert_image(filename):
    return send_from_directory(ALERT_DIR, filename)


@app.route("/alerts/<path:filename>/delete", methods=["DELETE"])
@login_required
def delete_alert(filename):
    global _alerts
    path = os.path.join(ALERT_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    with _alerts_lock:
        _alerts = deque([a for a in _alerts if a["filename"] != filename], maxlen=300)
        _save_alerts()
    return jsonify({"status": "ok"})


# No /info route (ngrok support removed)


# =====================================================================
# MAIN — waitress (on dinh hon Flask dev server tren Windows)
# =====================================================================
if __name__ == "__main__":
    # Start server (no ngrok)

    try:
        from waitress import serve
        print("[SERVER] Chay voi waitress tren port 5000...")
        serve(app, host="0.0.0.0", port=5000, threads=8,
              channel_timeout=300, cleanup_interval=30)
    except ImportError:
        print("[SERVER] waitress chua cai — chay Flask dev server")
        print("[SERVER] De on dinh hon: pip install waitress")
        app.run(host="0.0.0.0", port=5000, threaded=True)