import os
import sys

# RF optimization doesn't require GPU - it's fast enough on CPU

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # -1 = disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TF warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

print("[RF_Opt] GPU disabled - running on CPU only", file=sys.stderr)
sys.stderr.flush()

# Import TensorFlow and disable GPU explicitly
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print("[RF_Opt] TensorFlow GPU disabled", file=sys.stderr)
sys.stderr.flush()

import cv2
import numpy as np
import math
import time

from deepface import DeepFace
import itertools
import pandas as pd
import argparse
from typing import Optional

print("[RF_Opt] Python script started", file=sys.stderr)
print(f"[RF_Opt] Python version: {sys.version}", file=sys.stderr)
print(f"[RF_Opt] OpenCV: {cv2.__version__}", file=sys.stderr)

# --- Default Configuration ---
DEFAULT_REF_IMG_DIR = '' #Baseline identity images
DEFAULT_MODEL_NAME = "VGG-Face"
DEFAULT_DISTANCE_METRIC = "euclidean"  # Options: 'euclidean', 'cosine', 'euclidean_l2'
DEFAULT_IMPOSTER_CSV = '' #Baseline impostor distriubtion csv
DEFAULT_FMR_MIN = 1
DEFAULT_FMR_MAX = 1
DEFAULT_FMR_STEP = 0.1

print("[RF_Opt] Default configuration loaded", file=sys.stderr)

# Parse command-line arguments
def parse_arguments():
    try:
        parser = argparse.ArgumentParser(
            description='RF interference optimization for face recognition spoofing'
        )
        parser.add_argument(
            '--ref_img_dir',
            type=str,
            default=DEFAULT_REF_IMG_DIR,
            help=f'Directory containing reference images (default: {DEFAULT_REF_IMG_DIR})'
        )
        parser.add_argument(
            '--model',
            type=str,
            default=DEFAULT_MODEL_NAME,
            help=f'Face recognition model (default: {DEFAULT_MODEL_NAME})'
        )
        parser.add_argument(
            '--distance_metric',
            type=str,
            choices=['euclidean', 'cosine', 'euclidean_l2'],
            default=DEFAULT_DISTANCE_METRIC,
            help=f'Distance metric (default: {DEFAULT_DISTANCE_METRIC})'
        )
        parser.add_argument(
            '--imposter_csv',
            type=str,
            default=DEFAULT_IMPOSTER_CSV,
            help=f'Baseline imposter CSV (default: {DEFAULT_IMPOSTER_CSV})'
        )
        parser.add_argument(
            '--fmr_min',
            type=float,
            default=DEFAULT_FMR_MIN,
            help=f'Minimum FMR percentage (default: {DEFAULT_FMR_MIN})'
        )
        parser.add_argument(
            '--fmr_max',
            type=float,
            default=DEFAULT_FMR_MAX,
            help=f'Maximum FMR percentage (default: {DEFAULT_FMR_MAX})'
        )
        parser.add_argument(
            '--fmr_step',
            type=float,
            default=DEFAULT_FMR_STEP,
            help=f'FMR step size (default: {DEFAULT_FMR_STEP})'
        )
        parser.add_argument(
            '--output_csv',
            type=str,
            default=None,
            help='Output CSV path (default: RF_Optimization_Results_<MODEL>.csv)'
        )

        parser.add_argument(
            '--max_images',
            type=int,
            default=None,
            help='Optional cap on number of images to use from ref_img_dir (e.g., 10 for quick tests).'
        )

        parser.add_argument(
            '--progress_every',
            type=int,
            default=25,
            help='Print progress every N parameter combinations (default: 25).'
        )
        
        print("[RF_Opt] Argument parser created", file=sys.stderr)
        # Use parse_known_args so extra arguments don't hard-fail on clusters
        # where an older copy of this script might still be on $PATH.
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"[RF_Opt] Warning: ignoring unknown args: {unknown}", file=sys.stderr)
        print(f"[RF_Opt] Arguments parsed: model={args.model}, metric={args.distance_metric}", file=sys.stderr)
        return args
    except Exception as e:
        print(f"[RF_Opt] ERROR during argument parsing: {type(e).__name__}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# Parse arguments
try:
    print("[RF_Opt] Starting argument parsing...", file=sys.stderr)
    args = parse_arguments()
except Exception as e:
    print(f"[RF_Opt] FATAL: Failed to parse arguments: {e}", file=sys.stderr)
    sys.exit(1)

REF_IMG_DIR = args.ref_img_dir
MODEL_NAME = args.model
DISTANCE_METRIC = args.distance_metric
Imposter_baseline_csv = args.imposter_csv
fmr_min = args.fmr_min
fmr_max = args.fmr_max
fmr_step = args.fmr_step
MAX_IMAGES = args.max_images
PROGRESS_EVERY = max(1, int(args.progress_every))

print(f"[RF_Opt] Configuration: model={MODEL_NAME}, metric={DISTANCE_METRIC}, fmr_range={fmr_min}-{fmr_max}", file=sys.stderr)

# Set output CSV path
if args.output_csv:
    OUTPUT_CSV_PATH = args.output_csv
else:
    OUTPUT_CSV_PATH = f'RF_Optimization_Results_{MODEL_NAME}.csv'

print(f"[RF_Opt] Output CSV: {OUTPUT_CSV_PATH}", file=sys.stderr)

# Validate inputs
try:
    print(f"[RF_Opt] Validating inputs...", file=sys.stderr)
    if not os.path.isdir(REF_IMG_DIR):
        print(f"[Error] Reference image directory does not exist: {REF_IMG_DIR}")
        sys.exit(1)

    if not os.path.exists(Imposter_baseline_csv):
        print(f"[Error] Imposter baseline CSV does not exist: {Imposter_baseline_csv}")
        sys.exit(1)

    print(f"[RF_Opt] Inputs validated OK", file=sys.stderr)
except Exception as e:
    print(f"[RF_Opt] ERROR during validation: {type(e).__name__}: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print(f"\n{'='*70}")
print(f"[Config] Reference Image Dir: {REF_IMG_DIR}")
print(f"[Config] Model: {MODEL_NAME}")
print(f"[Config] Distance Metric: {DISTANCE_METRIC}")
print(f"[Config] Imposter Baseline CSV: {Imposter_baseline_csv}")
print(f"[Config] FMR Range: {fmr_min}% - {fmr_max}% (step: {fmr_step})")
print(f"[Config] Output CSV: {OUTPUT_CSV_PATH}")
print(f"{'='*70}\n")
# FNMR Threshold: A lower value makes it "harder" to verify, thus easier to get a high FNMR.

def build_fmr_grid(fmr_min, fmr_max, fmr_step=1.0):
    if fmr_step <= 0:
        raise ValueError('FMR step must be > 0')
    n = int(round((fmr_max - fmr_min) / fmr_step)) + 1
    return [round(fmr_min + i * fmr_step, 10) for i in range(n)]

def get_fmr_thresholds(csv_path, fmr_min=1, fmr_max=1, fmr_step=1):
    fmr_values = build_fmr_grid(fmr_min, fmr_max, fmr_step)

    if not os.path.exists(csv_path):
        print(f"Warning: Baseline {csv_path} not found. Using defaults.")
        return {fmr: (0.60, 0.85) for fmr in fmr_values}

    return {fmr: (0.60, 0.85) for fmr in fmr_values}

try:
    print("[RF_Opt] Initializing FNMR thresholds...", file=sys.stderr)
    FNMR_THRESHOLD = get_fmr_thresholds(Imposter_baseline_csv, fmr_min, fmr_max, fmr_step)
    print(f"[RF_Opt] FNMR thresholds initialized: {len(FNMR_THRESHOLD)} FMR values", file=sys.stderr)
except Exception as e:
    print(f"[RF_Opt] ERROR initializing FNMR: {type(e).__name__}: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

OUTPUT_CSV_PATH_OVERRIDE = OUTPUT_CSV_PATH

# --- Optimization Search Space ---
# Define the ranges for the parameters you want to optimize.
# Format: (start, stop, number_of_steps)
try:
    print("[RF_Opt] Initializing parameter ranges...", file=sys.stderr)
    RF_FREQ_RANGE = np.linspace(1e6, 25e6, 480)
    AMPLITUDE_RANGE = np.linspace(20, 80, 6)
    ANGLE_RANGE = np.linspace(0, 180, 3)
    print(f"[RF_Opt] Parameter ranges initialized OK", file=sys.stderr)
except Exception as e:
    print(f"[RF_Opt] ERROR initializing parameter ranges: {type(e).__name__}: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# --- Static RF Parameters (not being optimized in this loop) ---
ENABLE_AM = False
AM_DEPTH = 0.5
AM_FREQ_HZ = 20.0
ENABLE_FM = True
FM_DEV_HZ = 500E3
FM_FREQ_HZ = 194E3
AUTO_ROW_RATE = True
ROW_RATE = 720 * 30


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
                          fm_freq_hz: float = 0.0) -> Optional[np.ndarray]:
    """Applies simulated RF interference to an image."""
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    y_coords = np.arange(h, dtype=np.float32).reshape(h, 1)
    x_coords = np.arange(w, dtype=np.float32).reshape(1, w)

    theta = np.deg2rad(float(bar_angle_deg))
    slope = math.tan(theta)
    y_eff = y_coords + slope * x_coords
    t_eff = np.float32(t_seconds) + (y_eff / np.float32(row_rate))

    # FM phase calculation
    if enable_fm and fm_freq_hz > 0.0 and fm_dev_hz != 0.0:
        beta = float(fm_dev_hz) / float(fm_freq_hz)
        phase = (2.0 * np.pi * rf_freq * t_eff) + (beta * np.sin(2.0 * np.pi * fm_freq_hz * t_eff))
    else:
        phase = 2.0 * np.pi * rf_freq * t_eff

    # AM envelope calculation
    if enable_am and am_freq_hz > 0.0 and am_depth != 0.0:
        envelope = 1.0 + (am_depth * np.sin(2.0 * np.pi * am_freq_hz * t_eff))
        amp_map = amplitude * np.maximum(envelope, 0.0)
    else:
        amp_map = float(amplitude)

    offsets = amp_map * np.sin(phase)
    out = frame_bgr.astype(np.float32) + offsets[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def evaluate_attack(original_img_path, attacked_img):
    """
    Compares the original and attacked images to see if the attack caused a non-match.
    Returns True if it's a non-match (successful attack), False otherwise.
    """
    try:
        result = DeepFace.verify(
            img1_path=original_img_path,
            img2_path=attacked_img,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False
        )
        # A successful attack means the pair is NOT verified.
        return not result['verified']
    except Exception as e:
        print(f"\nError during verification for {os.path.basename(original_img_path)}: {e}")
        return False  # Treat errors as failed attacks


def _compute_distance(emb1: np.ndarray, emb2: np.ndarray, metric: str) -> float:
    """Compute embedding distance matching DeepFace metric conventions as closely as possible."""
    a = np.asarray(emb1, dtype=np.float32)
    b = np.asarray(emb2, dtype=np.float32)

    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return float("inf")
        cos_sim = float(np.dot(a, b) / denom)
        # DeepFace returns cosine *distance* (1 - similarity)
        return 1.0 - cos_sim

    if metric == "euclidean_l2":
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 0:
            a = a / na
        if nb > 0:
            b = b / nb
        return float(np.linalg.norm(a - b))

    # Default: euclidean
    return float(np.linalg.norm(a - b))


def _represent(img, model_name: str):
    """Compute embedding using DeepFace.represent for either file path or numpy image."""
    reps = DeepFace.represent(
        img_path=img,
        model_name=model_name,
        enforce_detection=False,
    )
    # DeepFace.represent returns list[dict] for a single image
    return np.asarray(reps[0]["embedding"], dtype=np.float32)


def _warmup_model(model_name: str) -> None:
    """Best-effort warmup: ensure model weights are loaded once."""
    try:
        _ = DeepFace.build_model(model_name)
    except Exception:
        # If build_model isn't available or fails, ignore.
        return


def main():
    """
    Main function to run the optimization loop.
    """
    global OUTPUT_CSV_PATH
    OUTPUT_CSV_PATH = OUTPUT_CSV_PATH_OVERRIDE  # Use the override value from args
    
    print(f"[Main] Starting main() function", file=sys.stderr)
    print(f"[Main] Output CSV path: {OUTPUT_CSV_PATH}", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        if not os.path.isdir(REF_IMG_DIR):
            print(f"Error: Reference directory not found at '{REF_IMG_DIR}'")
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [os.path.join(REF_IMG_DIR, f) for f in os.listdir(REF_IMG_DIR)
                       if os.path.splitext(f)[1].lower() in image_extensions]
        image_paths.sort()

        if MAX_IMAGES is not None:
            if MAX_IMAGES <= 0:
                raise ValueError(f"--max_images must be > 0, got {MAX_IMAGES}")
            image_paths = image_paths[:MAX_IMAGES]

        if not image_paths:
            print(f"Error: No images found in '{REF_IMG_DIR}'")
            return

        print(f"Found {len(image_paths)} images to process.")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        # -------------------------------------------------------------------
        # Load model once, then cache original embeddings
        # -------------------------------------------------------------------
        print(f"[Main] Warming up DeepFace model: {MODEL_NAME}", file=sys.stderr)
        sys.stderr.flush()
        _warmup_model(MODEL_NAME)

        print("[Main] Building original embedding cache...", file=sys.stderr)
        sys.stderr.flush()

        # Build once: original frame + embedding for each selected image
        originals = []
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Could not read image '{img_path}', skipping.")
                continue
            try:
                emb = _represent(img_path, MODEL_NAME)
            except Exception as e:
                print(f"Warning: embedding failed for '{img_path}': {type(e).__name__}: {e}")
                continue
            originals.append((img_path, frame, emb))

        if not originals:
            print("Error: No usable images after embedding cache build.")
            return

        try:
            # DeepFace.verify accepts two file paths; we provide same image twice to obtain threshold.
            # Threshold returned is model+metric specific.
            thr_probe = DeepFace.verify(
                img1_path=originals[0][0],
                img2_path=originals[0][0],
                model_name=MODEL_NAME,
                distance_metric=DISTANCE_METRIC,
                enforce_detection=False,
            )
            verify_threshold = float(thr_probe.get("threshold", 0.0))
        except Exception as e:
            print(f"Warning: could not retrieve DeepFace threshold; using 0.0. Error: {e}")
            verify_threshold = 0.0

        n_images = len(originals)
        print(f"[Main] Cached {n_images} embeddings. Verify threshold={verify_threshold}")

        param_grid = list(itertools.product(RF_FREQ_RANGE, AMPLITUDE_RANGE, ANGLE_RANGE))
        total_combinations = len(param_grid)
        print(f"Starting optimization search across {total_combinations} parameter combinations...")

        best_fnmr = -1.0
        best_params = {}
        results_log = []

        start_wall = time.time()
        last_report_wall = start_wall
        last_report_i = 0

        for i, (rf_freq, amplitude, angle) in enumerate(param_grid):
            successful_attacks_for_params = 0
            distances_this_combo = []

            for img_path, original_frame, original_emb in originals:
                height, _ = original_frame.shape[:2]
                row_rate = (30.0 * height) if AUTO_ROW_RATE else float(ROW_RATE)

                attacked_frame = apply_rf_interference(
                    original_frame,
                    rf_freq=rf_freq,
                    row_rate=row_rate,
                    amplitude=amplitude,
                    bar_angle_deg=angle,
                    enable_am=ENABLE_AM, am_depth=AM_DEPTH, am_freq_hz=AM_FREQ_HZ,
                    enable_fm=ENABLE_FM, fm_dev_hz=FM_DEV_HZ, fm_freq_hz=FM_FREQ_HZ
                )

                try:
                    attacked_emb = _represent(attacked_frame, MODEL_NAME)
                    dist = _compute_distance(original_emb, attacked_emb, DISTANCE_METRIC)
                    distances_this_combo.append(float(dist))
                    # DeepFace: verified if dist <= threshold. Successful attack => NOT verified.
                    if dist > verify_threshold:
                        successful_attacks_for_params += 1
                except Exception as e:
                    print(f"\nError during embedding/distance for {os.path.basename(img_path)}: {e}")
                    # Treat as failed attack (conservative)
                    continue

            # Calculate FNMR for this parameter set across all images
            current_fnmr = (successful_attacks_for_params / max(1, n_images)) * 100

            log_entry = {
                'rf_freq_hz': rf_freq,
                'amplitude': amplitude,
                'bar_angle_deg': angle,
                'successful_attacks': successful_attacks_for_params,
                'total_images': len(image_paths),
                'fnmr_percent': current_fnmr
            }
            results_log.append(log_entry)

            if current_fnmr > best_fnmr:
                best_fnmr = current_fnmr
                best_params = {
                    'rf_freq_hz': rf_freq,
                    'amplitude': amplitude,
                    'bar_angle_deg': angle,
                    'fnmr': best_fnmr
                }

            # Progress reporting (combo/sec)
            if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == total_combinations:
                now = time.time()
                dt = max(1e-9, now - last_report_wall)
                di = (i + 1) - last_report_i
                cps = di / dt
                elapsed = now - start_wall
                eta = (total_combinations - (i + 1)) / max(1e-9, cps)
                median_dist = float(np.median(distances_this_combo)) if distances_this_combo else float('nan')
                print(
                    f"\rProgress: {i + 1}/{total_combinations} | {cps:.2f} combos/s | "
                    f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | "
                    f"MedDist: {median_dist:.4f} | Thr: {verify_threshold:.4f} | "
                    f"Current FNMR: {current_fnmr:.2f}% | Best FNMR: {best_fnmr:.2f}%",
                    end="",
                    flush=True
                )
                last_report_wall = now
                last_report_i = (i + 1)

        # Save results to CSV
        df = pd.DataFrame(results_log)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n\nResults saved to '{OUTPUT_CSV_PATH}'")

        print("\n--- Optimization Complete ---")
        if not best_params:
            print("No successful attack parameters were found.")
        else:
            print("Best parameters found:")
            print(f"  - RF Frequency: {best_params['rf_freq_hz'] / 1e6:.4f} MHz")
            print(f"  - Amplitude: {best_params['amplitude']:.2f}")
            print(f"  - Bar Angle: {best_params['bar_angle_deg']:.2f} degrees")
            print(f"  - Achieved FNMR: {best_params['fnmr']:.2f}%")
    
    except Exception as e:
        print(f"\n[Error] Unexpected error in main(): {type(e).__name__}")
        print(f"[Error] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
