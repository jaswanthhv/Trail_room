import os
import random
import shutil
import string
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# --- Try-on tuning knobs ---
# If your desktop/front camera is mirrored, set this True to flip frames before pose/overlay.
SOURCE_MIRROR_FLIP = True   # <— turn ON for your Windows desktop capture

# If you don’t want to flip the whole frame, but only correct math, you can leave this False
# and set MIRROR_POLICY="force_on" in the renderer below. You can use both if needed.

# Scaling & smoothing
TOP_WIDTH_FACTOR = 1.08     # 1.0 = exactly shoulder span
BOT_WIDTH_FACTOR = 1.03     # relative to hip span
CHEST_Y_OFFSET   = 0.10     # fraction of frame height down from shoulder midpoint
HIP_Y_OFFSET     = 0.02     # fraction of frame height down from hip midpoint
SMOOTH_ALPHA     = 0.35     # 0..1 temporal smoothing (higher = smoother)
SPLIT_AT         = 0.55     # split point for single full outfit (top/bottom)



from flask import (
    Flask, request, jsonify, session, send_from_directory,
    render_template_string, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

# --- Video/image processing ---
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, ColorClip
from PIL import Image, ImageDraw
# Pillow 10+ compat for MoviePy
try:
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.LANCZOS  # type: ignore
except Exception:
    pass


# ---------------------------------------
# Flask setup & paths
# ---------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
app.permanent_session_lifetime = timedelta(days=2)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_ROOT = STATIC_DIR / "uploads"
RENDER_ROOT = STATIC_DIR / "renders"
for p in (UPLOAD_ROOT, RENDER_ROOT):
    p.mkdir(parents=True, exist_ok=True)

# Larger uploads OK
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

# Limits & formats
MAX_GARMENTS = 6
ALLOWED_VID = {"mp4", "mov", "webm", "m4v"}
ALLOWED_IMG = {"png", "jpg", "jpeg", "webp"}

# ---------------------------------------
# Helpers
# ---------------------------------------
def _ext_ok(filename, allowed) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def _get_sid() -> str:
    if "sid" not in session:
        session.permanent = True
        session["sid"] = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        app.logger.info(f"[SID] new session {session['sid']}")
    return session["sid"]

def _user_dir() -> Path:
    sid = _get_sid()
    d = UPLOAD_ROOT / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "garments").mkdir(exist_ok=True)
    (RENDER_ROOT / sid).mkdir(parents=True, exist_ok=True)
    return d

def _render_dir() -> Path:
    return RENDER_ROOT / _get_sid()

def _list_user_files() -> Tuple[str, List[Dict]]:
    d = _user_dir()
    video = None
    vids = sorted([p.name for p in d.glob("*") if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED_VID])
    if vids:
        video = f"/static/uploads/{_get_sid()}/{vids[-1]}"

    garments = []
    for p in sorted((d / "garments").glob("*")):
        if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED_IMG:
            garments.append({
                "id": p.stem,
                "name": p.stem,
                "url": f"/static/uploads/{_get_sid()}/garments/{p.name}"
            })
    return video, garments

# ---------------------------------------
# Sample asset generator
# ---------------------------------------
def _generate_sample_assets():
    d = _user_dir()
    gdir = d / "garments"
    gdir.mkdir(exist_ok=True)

    vid_path = d / "sample.mp4"
    if not vid_path.exists():
        app.logger.info("[SAMPLES] creating sample.mp4")
        base = ColorClip(size=(1280, 720), color=(18, 28, 48)).set_duration(4)
        grad = ColorClip(size=(1280, 720), color=(52, 94, 200)).set_duration(4).set_opacity(0.25)
        CompositeVideoClip([base, grad]).write_videofile(
            str(vid_path), codec="libx264", audio=False, fps=25, logger=None
        )

    shirt_path = gdir / "demo-shirt.png"
    if not shirt_path.exists():
        app.logger.info("[SAMPLES] creating demo-shirt.png")
        img = Image.new("RGBA", (700, 900), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        dr.rounded_rectangle([120, 280, 580, 820], radius=60, fill=(250, 250, 250, 235))
        dr.polygon([(120, 330), (40, 500), (120, 560), (120, 330)], fill=(240, 240, 240, 235))
        dr.polygon([(580, 330), (660, 500), (580, 560), (580, 330)], fill=(240, 240, 240, 235))
        dr.polygon([(260, 300), (350, 360), (450, 360), (540, 300), (260, 300)], fill=(230, 230, 230, 235))
        img.save(shirt_path)

    kurta_path = gdir / "demo-kurta.png"
    if not kurta_path.exists():
        app.logger.info("[SAMPLES] creating demo-kurta.png")
        img = Image.new("RGBA", (700, 900), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        dr.rounded_rectangle([180, 200, 520, 860], radius=80, fill=(210, 70, 100, 235))
        dr.polygon([(520, 210), (660, 320), (520, 420)], fill=(255, 210, 80, 220))
        img.save(kurta_path)

# ---------------------------------------
# “ML” stubs — wired to real renders
# ---------------------------------------
def mlstub_process(
    garment_ids: List[str],
    garment_map: Dict[str, Path],
    *,
    occasion: str,
    lighting: str,
    location_hint: str,
) -> List[Dict]:
    d = _user_dir()
    vids = sorted([p for p in d.glob("*") if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED_VID])
    if not vids:
        return []
    video_path = vids[-1]

    out_results = []
    for gid in garment_ids:
        g_img_path = garment_map.get(gid)
        if not g_img_path or not g_img_path.exists():
            continue
        out_path = _render_dir() / f"{gid}.mp4"

        need_render = True
        if out_path.exists():
            out_mtime = out_path.stat().st_mtime
            if g_img_path.stat().st_mtime < out_mtime and video_path.stat().st_mtime < out_mtime:
                need_render = False

        if need_render:
            try:
                _compose_preview(video_path, g_img_path, out_path, lighting)
                app.logger.info(f"[RENDER] {gid} -> {out_path.name}")
            except Exception as e:
                app.logger.warning(f"[RENDER][FAIL] {gid}: {e}")

        if out_path.exists():
            out_results.append({
                "id": gid,
                "preview_url": f"/static/renders/{_get_sid()}/{out_path.name}"
            })
    return out_results

def mlstub_score(
    garment_ids: List[str],
    *,
    occasion: str,
    lighting: str,
    location_hint: str,
) -> Dict[str, int]:
    seed_str = f"{sorted(garment_ids)}|{occasion}|{lighting}|{location_hint.strip().lower()}"
    seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
    rnd = random.Random(seed)
    base_by_occ = {"wedding": 72, "office": 68, "party": 70, "outdoor": 66}
    base = base_by_occ.get(occasion, 65)
    light_bump = {"auto": 2, "daylight": 3, "indoor": 1, "evening": 4}.get(lighting, 0)
    loc_bump = min(len(location_hint.strip()), 40) // 8
    scores = {}
    for gid in garment_ids:
        jitter = rnd.randint(-6, 12)
        score = max(40, min(100, base + light_bump + loc_bump + jitter))
        scores[gid] = score
    return scores

# ---------------------------------------
# Video composition (MoviePy)
# ---------------------------------------
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, ColorClip

def _safe_duration(v):
    # Try several ways to get a sensible clip length (1–6s)
    try:
        if v.duration and v.duration > 0:
            return max(1.0, min(6.0, float(v.duration)))
    except Exception:
        pass
    try:
        rd = getattr(v, "reader", None)
        if rd and getattr(rd, "duration", 0):
            return max(1.0, min(6.0, float(rd.duration)))
        nframes = getattr(rd, "nframes", 0)
        fps = getattr(v, "fps", None) or 25
        if nframes and fps:
            return max(1.0, min(6.0, float(nframes) / float(fps)))
    except Exception:
        pass
    return 4.0  # final fallback

##Compose Preview
# --- add near other imports ---
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
from PIL import Image

mp_pose = mp.solutions.pose
mp_selfie_seg = mp.solutions.selfie_segmentation

def _safe_duration(v):
    try:
        if v.duration and v.duration > 0:
            return max(1.0, min(6.0, float(v.duration)))
    except Exception:
        pass
    try:
        rd = getattr(v, "reader", None)
        if rd and getattr(rd, "duration", 0):
            return max(1.0, min(6.0, float(rd.duration)))
        nframes = getattr(rd, "nframes", 0)
        fps = getattr(v, "fps", None) or 25
        if nframes and fps:
            return max(1.0, min(6.0, float(nframes) / float(fps)))
    except Exception:
        pass
    return 4.0

def _load_rgba_np(path: Path):
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    rgb = arr[..., :3]
    a   = (arr[..., 3:4].astype(np.float32)) / 255.0
    return rgb, a

def _content_bbox(alpha01: np.ndarray, thresh=0.02):
    m = (alpha01[..., 0] > thresh).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1

def _crop_to_bbox(rgb, a, bbox):
    x0,y0,x1,y1 = bbox
    return rgb[y0:y1, x0:x1], a[y0:y1, x0:x1]

def _split_top_bottom(rgb: np.ndarray, a: np.ndarray):
    H, W = rgb.shape[:2]
    cut = int(H * SPLIT_AT)
    if H < 140:
        bb = _content_bbox(a)
        return {"top": _crop_to_bbox(rgb, a, bb) if bb else None, "bottom": None}
    # Crop halves to their content to avoid padding skew
    def crop_half(r, al):
        bb = _content_bbox(al)
        return _crop_to_bbox(r, al, bb) if bb else None
    top = crop_half(rgb[:cut], a[:cut])
    bot = crop_half(rgb[cut:], a[cut:])
    if top is None and bot is None:
        bb = _content_bbox(a)
        return {"top": _crop_to_bbox(rgb, a, bb) if bb else None, "bottom": None}
    return {"top": top, "bottom": bot}

def _resize_rgba(rgb, a, size_wh):
    W, H = size_wh
    rgb_r = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a_r   = cv2.resize(a,   (W, H), interpolation=cv2.INTER_LANCZOS4)[..., None]
    return rgb_r, a_r

def _flip_h_rgba(rgb, a):
    return np.ascontiguousarray(np.flip(rgb, 1)), np.ascontiguousarray(np.flip(a, 1))

def _affine_rgba(src_rgb, src_a, src_tri, dst_tri, out_wh):
    """
    Warp src (RGB+alpha) from src_tri -> dst_tri into an empty canvas (out_wh).
    Triangles: 3x2 float arrays in (x,y) pixel coords (relative to their images).
    """
    W, H = out_wh
    canvas_rgb = np.zeros((H, W, 3), np.uint8)
    canvas_a   = np.zeros((H, W, 1), np.float32)

    # Compute affine for RGB and A
    M = cv2.getAffineTransform(src_tri.astype(np.float32), dst_tri.astype(np.float32))
    dst_rgb = cv2.warpAffine(src_rgb, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    dst_a   = cv2.warpAffine((src_a*255).astype(np.uint8), M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dst_a   = (dst_a.astype(np.float32) / 255.0)[..., None]

    # Composite onto canvas (since we return the warped garment as an image)
    canvas_rgb = dst_rgb
    canvas_a   = dst_a
    return canvas_rgb, canvas_a

def _blend_under(dst_bgr, overlay_rgb, overlay_a, seg_mask, person_weight=0.35):
    """Alpha blend overlay (full-frame sized) under a bit of person mask."""
    h, w = dst_bgr.shape[:2]
    H, W = overlay_rgb.shape[:2]
    if (H, W) != (h, w):
        overlay_rgb = cv2.resize(overlay_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        overlay_a   = cv2.resize(overlay_a,   (w, h), interpolation=cv2.INTER_LANCZOS4)[..., None]
    roi_rgb = dst_bgr[:, :, ::-1]
    eff_a = np.clip(overlay_a * (1.0 - person_weight * seg_mask), 0.0, 1.0)
    out_rgb = (overlay_rgb * eff_a + roi_rgb * (1.0 - eff_a)).astype(np.uint8)
    dst_bgr[:, :, :] = out_rgb[:, :, ::-1]
    return dst_bgr

def _ema(prev, cur, alpha=SMOOTH_ALPHA):
    if prev is None: return cur
    return prev*(1-alpha) + cur*alpha

def _compose_preview(video_path: Path, garment_img: Path, out_path: Path, lighting: str):
    out_tmp = out_path.with_suffix(".tmp.mp4")
    full_rgb, full_a = _load_rgba_np(garment_img)
    parts = _split_top_bottom(full_rgb, full_a)  # {"top":(rgb,a) or None, "bottom":(rgb,a) or None}

    with VideoFileClip(str(video_path)) as v:
        duration = _safe_duration(v)
        vw, vh = v.w, v.h
        base_fps = v.fps or 25
        target_fps = min(15, int(base_fps))

        tint = None
        if lighting == "evening":
            tint = ColorClip(size=(vw, vh), color=(255, 200, 120)).set_opacity(0.06).set_duration(duration)
        elif lighting == "indoor":
            tint = ColorClip(size=(vw, vh), color=(255, 220, 180)).set_opacity(0.04).set_duration(duration)
        elif lighting == "daylight":
            tint = ColorClip(size=(vw, vh), color=(200, 220, 255)).set_opacity(0.04).set_duration(duration)

        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, smooth_landmarks=True)
        segm = mp_selfie_seg.SelfieSegmentation(model_selection=1)

        # Smoothed state
        s_shw = None; s_hipw = None
        s_LS = None;  s_RS = None
        s_LH = None;  s_RH = None
        s_angle = None

        def make_frame(t):
            nonlocal s_shw, s_hipw, s_LS, s_RS, s_LH, s_RH, s_angle

            frame = v.get_frame(t)            # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # --- Flip whole frame first if webcam/desktop is mirrored ---
            if SOURCE_MIRROR_FLIP:
                frame_bgr = cv2.flip(frame_bgr, 1)

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            res = pose.process(frame_rgb)
            seg = segm.process(frame_rgb)
            seg_mask = np.clip(seg.segmentation_mask, 0.0, 1.0)[..., None]

            # Defaults (fallbacks)
            LS = np.array([0.35*w, 0.35*h]); RS = np.array([0.65*w, 0.35*h])
            LH = np.array([0.42*w, 0.60*h]); RH = np.array([0.58*w, 0.60*h])
            angle = 0.0

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                def px(i): return np.array([lm[i].x*w, lm[i].y*h], np.float32), lm[i].visibility

                LS_, vLS = px(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                RS_, vRS = px(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                LH_, vLH = px(mp_pose.PoseLandmark.LEFT_HIP.value)
                RH_, vRH = px(mp_pose.PoseLandmark.RIGHT_HIP.value)

                # Ensure we assign anatomical left/right correctly even if camera flips
                # We keep the landmark semantics (LEFT == person’s left), but we don’t swap based on x.
                if vLS > 0.5: LS = LS_
                if vRS > 0.5: RS = RS_
                if vLH > 0.5: LH = LH_
                if vRH > 0.5: RH = RH_

                if vLS > 0.5 and vRS > 0.5:
                    dx, dy = (RS - LS)
                    angle = np.degrees(np.arctan2(dy, dx))

            # Smooth everything
            s_LS = _ema(s_LS, LS)
            s_RS = _ema(s_RS, RS)
            s_LH = _ema(s_LH, LH)
            s_RH = _ema(s_RH, RH)
            shw  = float(np.linalg.norm((s_RS - s_LS))) if (s_RS is not None and s_LS is not None) else 0.5*w
            hipw = float(np.linalg.norm((s_RH - s_LH))) if (s_RH is not None and s_LH is not None) else 0.55*w
            s_shw = _ema(s_shw, shw)
            s_hipw = _ema(s_hipw, hipw)
            s_angle = _ema(s_angle, angle)

            # Target triangles in output frame
            shoulder_mid = (s_RS + s_LS) / 2.0
            hip_mid = (s_RH + s_LH) / 2.0

            # Slight vertical offsets
            top_center = shoulder_mid + np.array([0, CHEST_Y_OFFSET*h], np.float32)
            bot_center = hip_mid      + np.array([0, HIP_Y_OFFSET*h],   np.float32)

            # Build destination triangles (L,R,center) for affine
            dst_top = np.stack([s_LS, s_RS, (s_LH + s_RH)/2.0], axis=0).astype(np.float32)
            dst_bot = np.stack([s_LH, s_RH, (s_LH + s_RH)/2.0 + np.array([0, 0.25*h], np.float32)], axis=0).astype(np.float32)

            # Shift dst_top so its centroid is at top_center (keeps vertical placement pleasant)
            dst_top_center = dst_top.mean(axis=0)
            dst_top += (top_center - dst_top_center)

            # Build full-size overlay canvases to composite once
            overlay_rgb = np.zeros((h, w, 3), np.uint8)
            overlay_a   = np.zeros((h, w, 1), np.float32)

            # --- TOP (affine) ---
            if parts["top"] is not None:
                t_rgb, t_a = parts["top"]
                Ht, Wt = t_rgb.shape[:2]
                # Source triangle: left-top, right-top, bottom-mid (tight to content)
                src_top = np.array([[0, 0], [Wt-1, 0], [Wt//2, Ht-1]], np.float32)

                # Adjust width vs shoulder with factor (scale the dst triangle around its center)
                top_scale = (TOP_WIDTH_FACTOR * float(s_shw or shw)) / max(1.0, np.linalg.norm(dst_top[1]-dst_top[0]))
                top_center_tri = dst_top.mean(axis=0, keepdims=True)
                dst_top_scaled = (dst_top - top_center_tri) * top_scale + top_center_tri

                wrgb, wa = _affine_rgba(t_rgb, t_a, src_top, dst_top_scaled, (w, h))
                # Composite into the big overlay (alpha max to accumulate)
                m = (wa > overlay_a).astype(np.float32)
                overlay_rgb = (overlay_rgb*(1-m) + wrgb*m).astype(np.uint8)
                overlay_a   = np.maximum(overlay_a, wa)

            # --- BOTTOM (affine) ---
            if parts["bottom"] is not None:
                b_rgb, b_a = parts["bottom"]
                Hb, Wb = b_rgb.shape[:2]
                # Source triangle: left-top, right-top, bottom-mid
                src_bot = np.array([[0, 0], [Wb-1, 0], [Wb//2, Hb-1]], np.float32)

                # Adjust vs hip width
                bot_scale = (BOT_WIDTH_FACTOR * float(s_hipw or hipw)) / max(1.0, np.linalg.norm(dst_bot[1]-dst_bot[0]))
                bot_center_tri = dst_bot.mean(axis=0, keepdims=True)
                dst_bot_scaled = (dst_bot - bot_center_tri) * bot_scale + bot_center_tri

                wrgb, wa = _affine_rgba(b_rgb, b_a, src_bot, dst_bot_scaled, (w, h))
                m = (wa > overlay_a).astype(np.float32)
                overlay_rgb = (overlay_rgb*(1-m) + wrgb*m).astype(np.uint8)
                overlay_a   = np.maximum(overlay_a, wa)

            # Blend under person a bit for occlusion
            composed = _blend_under(frame_bgr.copy(), overlay_rgb, overlay_a, seg_mask, person_weight=0.35)

            return cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)

        v_sub = v.subclip(0, duration)
        out_clip = v_sub.set_fps(target_fps).fl_image(lambda fr: fr)
        out_clip = out_clip.set_make_frame(make_frame)

        final = CompositeVideoClip([out_clip, tint] if tint is not None else [out_clip])
        final.write_videofile(
            str(out_tmp),
            codec="libx264",
            audio=False,
            fps=target_fps,
            preset="medium",
            threads=max(1, (os.cpu_count() or 2)),
            logger=None
        )

    if out_path.exists():
        out_path.unlink(missing_ok=True)
    out_tmp.rename(out_path)

# ---------------------------------------
# Routes — Pages
# ---------------------------------------
@app.get("/")
def index():
    _get_sid()
    video_url, garments = _list_user_files()
    session.setdefault("last_context", {"occasion": "wedding", "lighting": "auto", "location_hint": ""})

    # NOTE: all forms now have method="post" + action="...".
    # Even if JS fails, uploads will POST to the right endpoints.
    return render_template_string("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Virtual Try-On — Fixed Uploads + Renders</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --bg:#f8fafc; --card:#ffffff; --muted:#6b7280; --ring:#d1d5db; --text:#0f172a; --accent:#2563eb;}
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:var(--text); background: linear-gradient(#fff, #f5f7fb); }
    header { position: sticky; top:0; background: rgba(255,255,255,0.8); backdrop-filter: blur(8px); border-bottom:1px solid var(--ring); padding: 10px 16px; z-index:10;}
    .wrap { max-width: 1200px; margin: 0 auto; padding: 16px;}
    .grid { display:grid; grid-template-columns: 1fr; gap:16px; }
    @media (min-width: 1200px) { .grid { grid-template-columns: 420px 1fr; } }
    .card { background: var(--card); border:1px solid var(--ring); border-radius:16px; box-shadow: 0 6px 24px rgba(15,23,42,0.05); }
    .card .hd { padding:14px 16px; border-bottom: 1px solid var(--ring); font-weight:600; }
    .card .bd { padding:16px; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap: wrap;}
    .btn { background: var(--accent); color:#fff; padding:8px 12px; border-radius:10px; border:0; cursor:pointer; font-weight:600;}
    .btn.secondary { background:#e5e7eb; color:#111827; }
    .muted { color: var(--muted); font-size: 12px; }
    .field { display:block; width:100%; padding:10px 12px; border:1px solid var(--ring); border-radius:10px; }
    .badge { display:inline-flex; align-items:center; gap:6px; background:#eef2ff; color:#3730a3; font-size:12px; padding:4px 8px; border-radius:999px; border:1px solid #c7d2fe;}
    .video-wrap { position: relative; width: 100%; padding-top: 56.25%; background:#000; border-radius:16px; overflow:hidden; }
    .video-wrap video { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0; }
    .tab { padding:6px 10px; border:1px solid var(--ring); border-radius:10px; background:#fff; cursor:pointer;}
    .tab.active { background:#111827; color:#fff; border-color:#111827; }
  </style>
</head>
<body>
  <header>
    <div class="wrap row" style="justify-content:space-between;">
      <div class="row">
        <div style="font-weight:700;">Virtual Try-On — Fixed Uploads + Renders</div>
        <span class="badge">MoviePy</span>
      </div>
      <form action="/clear" method="post" onsubmit="return confirm('Clear current uploads & renders?');">
        <button class="btn secondary" type="submit">Clear Session</button>
      </form>
    </div>
  </header>

  <main class="wrap">
    <div class="grid">

      <!-- LEFT -->
      <div>
        <div class="card">
          <div class="hd">1) Upload Video (works without JS)</div>
          <div class="bd">
            <form id="videoForm" class="row" method="post" action="/upload_video" enctype="multipart/form-data">
              <input class="field" type="file" name="video" accept="video/*" required />
              <button class="btn secondary" type="submit">Upload</button>
              <button class="btn" formaction="/load_samples" formmethod="post">Load Test Assets</button>
            </form>
            <div class="muted" id="videoName">{{ 'Current: ' + video_url if video_url else 'No video yet' }}</div>
          </div>
        </div>

        <div class="card" style="margin-top:16px;">
          <div class="hd">2) Upload Garments (max 6) (works without JS)</div>
          <div class="bd">
            <form id="garmentForm" class="row" method="post" action="/upload_garments" enctype="multipart/form-data">
              <input class="field" type="file" name="garments" accept="image/*" multiple required />
              <button class="btn secondary" type="submit">Upload</button>
            </form>
            <div class="muted">Uploaded: {{ garments|length }} image(s)</div>
          </div>
        </div>

        <div class="card" style="margin-top:16px;">
          <div class="hd">3) Occasion & Location</div>
          <div class="bd">
            <form class="row" method="post" action="/save_context">
              <select name="occasion" class="field" style="flex:1;">
                <option value="wedding">Wedding</option>
                <option value="office">Office/Meeting</option>
                <option value="party">Party/Evening</option>
                <option value="outdoor">Outdoor/Day</option>
              </select>
              <select name="lighting" class="field" style="flex:1;">
                <option value="auto">Auto</option>
                <option value="daylight">Daylight</option>
                <option value="indoor">Indoor warm</option>
                <option value="evening">Evening/Golden</option>
              </select>
              <input name="location_hint" class="field" placeholder="Paste Google Maps URL or 'Venue lawn'" />
              <button class="btn" type="submit">Save Context</button>
            </form>

            <form style="margin-top:10px;" method="post" action="/process_and_score">
              <button class="btn" type="submit">Generate Previews (no JS)</button>
              <div class="muted">This button will render previews and reload the page to show them.</div>
            </form>
          </div>
        </div>
      </div>

      <!-- RIGHT -->
      <div>
        <div class="card">
          <div class="hd">Preview / Compare</div>
          <div class="bd">
            <div class="tabs" id="tabs">
              {% for g in garments %}
                <a class="tab" href="#look-{{ g.id }}">Look {{ loop.index }}</a>
              {% endfor %}
            </div>

            <!-- Base video -->
            <div class="video-wrap" style="margin-bottom:10px;">
              <video id="baseVid" autoplay loop muted playsinline controls {% if video_url %} src="{{ video_url }}?v={{ cachebust }}"{% endif %}></video>
            </div>

            <!-- Rendered overlays (server-side mp4s) -->
            {% if garments|length == 0 %}
              <div class="muted">Upload garments, then click "Generate Previews".</div>
            {% else %}
              {% for g in garments %}
                <div id="look-{{ g.id }}" style="margin-bottom:16px;">
                  <div style="font-weight:600;">Look {{ loop.index }} — {{ g.name }}</div>
                  <div class="video-wrap">
                    <video autoplay loop muted playsinline controls src="/static/renders/{{ sid }}/{{ g.id }}.mp4?v={{ cachebust }}"></video>
                  </div>
                </div>
              {% endfor %}
            {% endif %}
          </div>
        </div>
      </div>

    </div>
  </main>
</body>
</html>
    """, sid=_get_sid(), video_url=video_url, garments=garments, cachebust=random.randint(1, 1_000_000))

# ---------------------------------------
# Routes — Uploads / Session / Static
# ---------------------------------------
@app.post("/upload_video")
def upload_video():
    f = request.files.get("video")
    if not f or f.filename == "":
        flash("No video file provided", "error")
        return redirect(url_for("index"))
    if not _ext_ok(f.filename, ALLOWED_VID):
        flash(f"Unsupported video type. Allowed: {sorted(ALLOWED_VID)}", "error")
        return redirect(url_for("index"))
    d = _user_dir()
    for p in d.glob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED_VID:
            p.unlink(missing_ok=True)
    filename = secure_filename(f.filename)
    dst = d / filename
    f.save(dst)
    app.logger.info(f"[UPLOAD][VIDEO] saved {dst}")
    return redirect(url_for("index"))

@app.post("/upload_garments")
def upload_garments():
    files = request.files.getlist("garments")
    if not files:
        flash("No images provided", "error")
        return redirect(url_for("index"))
    d = _user_dir() / "garments"
    existing = list(d.glob("*"))
    if len(existing) >= MAX_GARMENTS:
        flash(f"Cart already has {MAX_GARMENTS} garments", "error")
        return redirect(url_for("index"))
    added = 0
    for f in files:
        if f and f.filename and _ext_ok(f.filename, ALLOWED_IMG):
            if len(existing) + added >= MAX_GARMENTS:
                break
            safe = secure_filename(f.filename)
            stem = Path(safe).stem
            ext = Path(safe).suffix.lower()
            final = (d / f"{stem}{ext}")
            ctr = 1
            while final.exists():
                final = d / f"{stem}-{ctr}{ext}"
                ctr += 1
            f.save(final)
            added += 1
            app.logger.info(f"[UPLOAD][GARMENT] saved {final}")
    return redirect(url_for("index"))

@app.post("/clear")
def clear_all():
    d = _user_dir()
    try:
        shutil.rmtree(d)
        app.logger.info("[CLEAR] uploads cleared")
    except Exception as e:
        app.logger.warning(f"[CLEAR] uploads failed: {e}")
    rd = _render_dir()
    try:
        shutil.rmtree(rd)
        app.logger.info("[CLEAR] renders cleared")
    except Exception as e:
        app.logger.warning(f"[CLEAR] renders failed: {e}")
    return redirect(url_for("index"))

@app.get("/static/uploads/<path:filename>")
def uploaded(filename):
    return send_from_directory(UPLOAD_ROOT, filename)

@app.get("/static/renders/<path:subpath>")
def rendered(subpath):
    return send_from_directory(RENDER_ROOT, subpath)

# ---------------------------------------
# Context + Samples + Process
# ---------------------------------------
@app.post("/save_context")
def save_context():
    # Accept both form and JSON
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict(flat=True)
    session["last_context"] = {
        "occasion": (data.get("occasion") or "wedding").lower(),
        "lighting": (data.get("lighting") or "auto").lower(),
        "location_hint": data.get("location_hint") or "",
    }
    session.modified = True
    app.logger.info(f"[CTX] {session['last_context']}")
    return redirect(url_for("index"))

@app.post("/load_samples")
def load_samples():
    _generate_sample_assets()
    app.logger.info("[SAMPLES] loaded")
    return redirect(url_for("index"))

@app.post("/process_and_score")
def process_and_score():
    # render previews for all current garments, then redirect to show them
    _, garments = _list_user_files()
    garment_ids = [g["id"] for g in garments]
    if not garment_ids:
        flash("No garments to process", "error")
        return redirect(url_for("index"))

    ctx = session.get("last_context") or {"occasion": "wedding", "lighting": "auto", "location_hint": ""}
    d = _user_dir() / "garments"
    gmap_all = {p.stem: p for p in d.glob("*") if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED_IMG}
    gmap = {gid: gmap_all[gid] for gid in garment_ids if gid in gmap_all}
    results = mlstub_process(garment_ids, gmap, **ctx)
    if not results:
        flash("Processing failed (no previews produced). Ensure you uploaded a video and at least one garment image.", "error")
    else:
        flash(f"Rendered {len(results)} preview(s).", "ok")
    return redirect(url_for("index"))

# Optional JSON API (kept for JS button “Generate Previews” if you add it back)
@app.post("/process")
def process_stub():
    data = request.get_json(silent=True) or {}
    garment_ids = data.get("garment_ids") or []
    occasion = (data.get("occasion") or "wedding").lower()
    lighting = (data.get("lighting") or "auto").lower()
    location_hint = (data.get("location_hint") or "")

    d = _user_dir() / "garments"
    gmap_all = {p.stem: p for p in d.glob("*") if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED_IMG}
    gmap = {gid: gmap_all[gid] for gid in garment_ids if gid in gmap_all}
    if not gmap:
        return jsonify(ok=False, error="No garments found to process"), 400
    results = mlstub_process(garment_ids, gmap, occasion=occasion, lighting=lighting, location_hint=location_hint)
    return jsonify(ok=True, results=results)

@app.post("/score")
def score_stub():
    data = request.get_json(silent=True) or {}
    garment_ids = data.get("garment_ids") or []
    occasion = (data.get("occasion") or "wedding").lower()
    lighting = (data.get("lighting") or "auto").lower()
    location_hint = (data.get("location_hint") or "")
    if not garment_ids:
        return jsonify(ok=False, error="No garments to score"), 400
    scores = mlstub_score(garment_ids, occasion=occasion, lighting=lighting, location_hint=location_hint)
    return jsonify(ok=True, scores=scores)

# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    app.logger.setLevel("INFO")
    app.run(host="0.0.0.0", port=port, debug=True)
