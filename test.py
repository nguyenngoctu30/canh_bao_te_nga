"""
Script do tim duong dan (path) RTSP dung cho camera.
Chay rieng, KHONG lien quan toi app.py — chi de tim ra URL RTSP dung.

Cach dung:
    python test_rtsp.py

Sua IP / USERNAME / PASSWORD ben duoi truoc khi chay.
"""

import cv2
import os
import time

IP = "192.168.1.8"
PORT = 554
USERNAME = "admin"
PASSWORD = "JuLdN5Qv"

# Timeout ngan (giay) cho moi lan thu — khong doi 30s nhu mac dinh
CONNECT_TIMEOUT_SEC = 6

# Cac duong dan RTSP pho bien cho camera dang V380 / OEM Trung Quoc /
# Hikvision-compatible / Dahua-compatible. Se thu lan luot.
CANDIDATE_PATHS = [
    "",                                 # rtsp://ip:554  (khong path)
    "/live/ch00_0",                     # V380 main stream
    "/live/ch00_1",                     # V380 sub stream
    "/live/0/MAIN",                     
    "/live/0/SUB",
    "/11",                              # mot so dong V380 Pro / IP cam OEM
    "/12",
    "/h264",
    "/h264_stream",
    "/stream1",
    "/stream2",
    "/onvif1",                          # ONVIF profile stream
    "/onvif2",
    "/cam/realmonitor?channel=1&subtype=0",   # Dahua-style
    "/cam/realmonitor?channel=1&subtype=1",
    "/Streaming/Channels/101",          # Hikvision-style (main)
    "/Streaming/Channels/102",          # Hikvision-style (sub)
    "/user=admin_password=_channel=1_stream=0.sdp",  # mot so OEM re
]

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    f"rtsp_transport;tcp|stimeout;{CONNECT_TIMEOUT_SEC * 1_000_000}"
)


def try_path(path: str):
    url = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}{path}"
    print(f"-> Dang thu: {url}")
    t0 = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    ok = cap.isOpened()
    if ok:
        ret, frame = cap.read()
        ok = ret and frame is not None
    elapsed = time.time() - t0
    cap.release()
    status = "THANH CONG ✔" if ok else "that bai"
    print(f"   {status}  ({elapsed:.1f}s)")
    return ok, url


def main():
    print(f"Dang do RTSP path cho {IP}:{PORT} (user={USERNAME})")
    print(f"Moi path timeout toi da ~{CONNECT_TIMEOUT_SEC}s\n")

    found = []
    for path in CANDIDATE_PATHS:
        ok, url = try_path(path)
        if ok:
            found.append(url)

    print("\n================ KET QUA ================")
    if found:
        print("Cac URL ket noi thanh cong:")
        for u in found:
            print(f"  {u}")
        print("\n=> Dung 1 trong cac URL tren de set RTSP_URL trong app.py")
    else:
        print("KHONG co path nao ket noi duoc.")
        print("Kha nang cao la:")
        print("  1) Sai username/password")
        print("  2) RTSP chua duoc bat trong app dieu khien camera (vi du app V380 Pro)")
        print("     -> vao Cai dat camera > tim muc 'RTSP' / 'ONVIF' va bat len")
        print("  3) Camera dung cong khac 554 (kiem tra trong app camera)")
        print("  4) May tinh va camera khong cung mang / bi tuong lua chan")
        print("     -> thu: ping", IP, " va Test-NetConnection", IP, "-Port", PORT)


if __name__ == "__main__":
    main()