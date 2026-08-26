import os
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
# ====== Static user inputs (edit these) ======
INPUT_DIR  = '' #Location of input data
OUTPUT_DIR = '' #Location to save attacked data video/images
PROCESS_MODE  = 'image'    # 'image' or 'video'

# RF parameters
RF_FREQ_HZ    =  11.465E6 #  carrier frequency (Hz)
AMPLITUDE     = 80  # base intensity offset [0..255]
AUTO_ROW_RATE = True    # if False, use ROW_RATE below
ROW_RATE      = 720*30    # rows/sec (height x fps)

# Bar rotation (degrees). 0 => horizontal bars; positive tilts upward to the right.
BAR_ANGLE_DEG = 90 #0-360

# Amplitude modulation (AM) controls
ENABLE_AM   = False
AM_DEPTH    = 0.5
AM_FREQ_HZ  = 20.0

# Frequency modulation (FM) controls
ENABLE_FM   = True
FM_DEV_HZ   = 500E3
FM_FREQ_HZ  = 194E3
# ============================================

def apply_rf_interference(frame_bgr: np.ndarray,
                          rf_freq: float,
                          row_rate: float,
                          amplitude: float,
                          t_seconds: float = 0.0,
                          bar_angle_deg: float = 0.0,
                          enable_am: bool = False,
                          am_depth: float = 0.0,
                          am_freq_hz: float = 0.0,
                          enable_fm: bool = False,
                          fm_dev_hz: float = 0.0,
                          fm_freq_hz: float = 0.0) -> np.ndarray:
    if frame_bgr is None:
        return frame_bgr

    img = frame_bgr
    gray_input = False
    if img.ndim == 2:
        gray_input = True
        img = img[:, :, None]

    h, w = img.shape[:2]
    y_coords = np.arange(h, dtype=np.float32).reshape(h, 1)
    x_coords = np.arange(w, dtype=np.float32).reshape(1, w)

    theta = np.deg2rad(float(bar_angle_deg))
    slope = math.tan(theta)
    y_eff = y_coords + slope * x_coords

    t_eff = np.float32(t_seconds) + (y_eff / np.float32(row_rate))

    if enable_fm and fm_freq_hz > 0.0 and fm_dev_hz != 0.0:
        beta = float(fm_dev_hz) / float(fm_freq_hz)
        phase = (2.0 * np.pi * np.float32(rf_freq) * t_eff) + (np.float32(beta) * np.sin(2.0 * np.pi * np.float32(fm_freq_hz) * t_eff))
    else:
        phase = 2.0 * np.pi * np.float32(rf_freq) * t_eff

    if enable_am and am_freq_hz > 0.0 and am_depth != 0.0:
        envelope = 1.0 + (np.float32(am_depth) * np.sin(2.0 * np.pi * np.float32(am_freq_hz) * t_eff))
        envelope = np.maximum(envelope, 0.0)
        amp_map = np.float32(amplitude) * envelope
    else:
        amp_map = np.float32(amplitude)

    offsets = amp_map * np.sin(phase)

    out = img.astype(np.float32)
    out += offsets[:, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    if gray_input:
        out = out[:, :, 0]
    return out

def choose_fourcc(output_path: str):
    ext = os.path.splitext(output_path)[1].lower()
    if ext in {'.mp4', '.m4v', '.mov'}:
        return cv2.VideoWriter_fourcc(*'mp4v')
    if ext in {'.avi'}:
        return cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter_fourcc(*'mp4v')

def process_image(in_path: str, out_dir: str):
    base, ext = os.path.splitext(os.path.basename(in_path))
    out_path = os.path.join(out_dir, f'{base}{ext or ".jpg"}')

    frame = cv2.imread(in_path)
    if frame is None:
        print(f'Failed to read image: {in_path}')
        return

    height, width = frame.shape[:2]
    # For a static image, row_rate is arbitrary unless AUTO_ROW_RATE is False.
    row_rate = (30.0 * height) if AUTO_ROW_RATE else float(ROW_RATE)

    attacked = apply_rf_interference(
        frame,
        rf_freq=RF_FREQ_HZ,
        row_rate=row_rate,
        amplitude=AMPLITUDE,
        t_seconds=0.0,
        bar_angle_deg=BAR_ANGLE_DEG,
        enable_am=ENABLE_AM,
        am_depth=AM_DEPTH,
        am_freq_hz=AM_FREQ_HZ,
        enable_fm=ENABLE_FM,
        fm_dev_hz=FM_DEV_HZ,
        fm_freq_hz=FM_FREQ_HZ,
    )
    cv2.imwrite(out_path, attacked)
    print(f'Done. Wrote attacked image to: {out_path}')

def process_video(in_path: str, out_dir: str):
    base, ext = os.path.splitext(os.path.basename(in_path))
    out_path = os.path.join(out_dir, f'{base}_rf{ext or ".mp4"}')

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f'Failed to open input: {in_path}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        print(f'Could not read video resolution: {in_path}')
        cap.release()
        return

    row_rate = (fps * height) if AUTO_ROW_RATE else float(ROW_RATE)
    fourcc = choose_fourcc(out_path)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        print(f'Failed to open output for writing: {out_path}')
        cap.release()
        return

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            t_seconds = frame_idx / float(fps)
            attacked = apply_rf_interference(
                frame,
                rf_freq=RF_FREQ_HZ,
                row_rate=row_rate,
                amplitude=AMPLITUDE,
                t_seconds=t_seconds,
                bar_angle_deg=BAR_ANGLE_DEG,
                enable_am=ENABLE_AM,
                am_depth=AM_DEPTH,
                am_freq_hz=AM_FREQ_HZ,
                enable_fm=ENABLE_FM,
                fm_dev_hz=FM_DEV_HZ,
                fm_freq_hz=FM_FREQ_HZ,
            )
            writer.write(attacked)
            frame_idx += 1

        print(f'Done. Wrote {frame_idx} frames to: {out_path}')
    finally:
        cap.release()
        writer.release()

def synth_signal(t: np.ndarray) -> np.ndarray:
    # AM envelope
    if ENABLE_AM and AM_FREQ_HZ > 0.0 and AM_DEPTH != 0.0:
        envelope = 1.0 + (AM_DEPTH * np.sin(2.0 * np.pi * AM_FREQ_HZ * t))
        envelope = np.maximum(envelope, 0.0)
    else:
        envelope = 1.0

    # FM phase
    if ENABLE_FM and FM_FREQ_HZ > 0.0 and FM_DEV_HZ != 0.0:
        beta = float(FM_DEV_HZ) / float(FM_FREQ_HZ)
        phase = (2.0 * np.pi * RF_FREQ_HZ * t) + (beta * np.sin(2.0 * np.pi * FM_FREQ_HZ * t))
    else:
        phase = 2.0 * np.pi * RF_FREQ_HZ * t

    return AMPLITUDE * envelope * np.sin(phase)

def plotting():
    # Two periods of the carrier
    T_c = 1.0 / float(RF_FREQ_HZ)
    t = np.linspace(0.0, 2.0 * T_c, 5000, endpoint=True)

    s = synth_signal(t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, s, lw=1.5)
    plt.title('RF interference signal over two carrier periods')
    plt.xlabel('Time [s]')
    plt.ylabel('Offset intensity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    in_dir = os.path.expanduser(INPUT_DIR)
    out_dir = os.path.expanduser(OUTPUT_DIR)

    if not os.path.isdir(in_dir):
        print(f'Input directory not found: {in_dir}')
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    if PROCESS_MODE == 'video':
        exts = {'.mp4', '.m4v', '.mov', '.avi', '.mkv'}
        process_func = process_video
        file_type = 'videos'
    elif PROCESS_MODE == 'image':
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        process_func = process_image
        file_type = 'images'
    else:
        print(f"Error: Unknown PROCESS_MODE '{PROCESS_MODE}'. Use 'image' or 'video'.")
        sys.exit(1)

    files = [f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in exts]

    if not files:
        print(f'No {file_type} found in: {in_dir}')
        sys.exit(0)

    for fname in files:
        in_path = os.path.join(in_dir, fname)
        print(f'Processing: {in_path}')
        process_func(in_path, out_dir)

if __name__ == '__main__':
    #plotting() #Uncomment for plotting the synthesized RF signal
    main()
