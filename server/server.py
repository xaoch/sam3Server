"""
SAM3 FastAPI Server (Transformers)
=================================================
Endpoints:
  GET  /health

Image segmentation:
  POST /segment/text    : PCS (Sam3Processor + Sam3Model) using text (+ optional pos/neg boxes via input_boxes_labels)
  POST /segment/visual  : PVS (Sam3TrackerProcessor + Sam3TrackerModel) using points and/or boxes (best-effort optional text)

Video workflow (tracking) — client owns the video UI:
  POST /video/upload            : upload a video -> {video_id}
  POST /video/session/init      : init a tracker video session -> {session_id}
  POST /video/session/prompt    : add points/boxes prompt on a frame (and obj_id) -> {ok}
  POST /video/session/propagate : propagate tracking through video -> masks per frame
  POST /video/session/close     : free session memory

Notes:
- Video tracking is implemented with Sam3TrackerVideoModel/Processor if available.
- The client already has the video, so the server does NOT provide frames back.
- Server still needs to decode the uploaded video internally; install PyAV (conda-forge av) if needed.

Run:
  pip install fastapi uvicorn pillow numpy torch transformers python-multipart
  (optional) pip install pycocotools
  (video)    conda install -c conda-forge av    # recommended, provides pyav + ffmpeg

  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import traceback
from pydantic import BaseModel, Field
from PIL import Image

from transformers import Sam3Model, Sam3Processor

# Optional: COCO RLE encoding
try:
    from pycocotools import mask as mask_utils

    HAS_RLE = True
except Exception:
    HAS_RLE = False

# Optional: Tracker (image)
try:
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    HAS_TRACKER = True
except Exception:
    HAS_TRACKER = False

# Optional: Tracker (video)
try:
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    from transformers.video_utils import load_video

    HAS_VIDEO_TRACKER = True
except Exception:
    HAS_VIDEO_TRACKER = False


# ----------------------------
# Utilities
# ----------------------------
def decode_base64_image(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


def mask_to_base64_png(mask: np.ndarray) -> str:
    """mask: HxW bool array"""
    m = (mask.astype(np.uint8) * 255)
    im = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """COCO-style RLE, Fortran order. Requires pycocotools."""
    if not HAS_RLE:
        raise HTTPException(status_code=500, detail="pycocotools not installed; cannot return RLE.")
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # pycocotools returns bytes
    return rle


def frame_to_pil(frame: Any) -> Image.Image:
    """
    load_video() can return frames as PIL Images OR numpy arrays depending on
    transformers version/backend. Normalize to PIL RGB.
    """
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3:
            raise TypeError(f"Unsupported ndarray frame shape: {arr.shape}")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _to_device(batch_encoding, device: str):
    # BatchEncoding implements .to(device) in recent Transformers; keep fallback
    try:
        return batch_encoding.to(device)
    except Exception:
        for k, v in batch_encoding.items():
            if torch.is_tensor(v):
                batch_encoding[k] = v.to(device)
        return batch_encoding


def cast_batchencoding_floats(batch, dtype: torch.dtype):
    """
    Cast all floating-point tensors in a BatchEncoding to dtype.
    Leave integer tensors (input_ids, attention masks, etc.) alone.
    """
    for k, v in batch.items():
        if torch.is_tensor(v) and v.is_floating_point():
            batch[k] = v.to(dtype=dtype)
    return batch


# ----------------------------
# Schemas
# ----------------------------
class Box(BaseModel):
    # xyxy in pixel coordinates
    x1: int
    y1: int
    x2: int
    y2: int
    # For PCS: label supports neg/pos via input_boxes_labels. For Tracker: label may be ignored.
    label: Literal[0, 1] = 1  # 1=positive, 0=negative


class Point(BaseModel):
    x: int
    y: int
    label: Literal[0, 1] = 1  # 1=positive, 0=negative


class TextSegmentRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded RGB image bytes (jpg/png).")
    text: str = Field(..., description="Concept prompt, e.g., 'person', 'yellow school bus'.")
    boxes: Optional[List[Box]] = Field(
        default=None,
        description="Optional XYXY boxes to guide segmentation (label 1=pos, 0=neg) (PCS supports labels).",
    )
    output: Literal["png", "rle"] = "png"
    threshold: float = 0.5
    mask_threshold: float = 0.5


class VisualSegmentRequest(BaseModel):
    image_b64: str
    text: Optional[str] = None
    boxes: Optional[List[Box]] = None
    points: Optional[List[Point]] = None
    output: Literal["png", "rle"] = "png"
    threshold: float = 0.5
    multimask_output: bool = True


class MaskResult(BaseModel):
    score: Optional[float] = None
    png_b64: Optional[str] = None
    rle: Optional[Dict[str, Any]] = None
    box_xyxy: Optional[List[float]] = None


class SegmentResponse(BaseModel):
    model: str
    device: str
    results: List[MaskResult]


# Video
class VideoSessionInitRequest(BaseModel):
    video_id: str
    mode: Literal["tracker"] = "tracker"
    inference_device: Optional[str] = None  # override; defaults to server DEVICE


class VideoPromptRequest(BaseModel):
    session_id: str
    frame_idx: int = 0
    obj_id: int = 1
    points: Optional[List[Point]] = None
    boxes: Optional[List[Box]] = None
    clear_old_inputs: bool = True

    # coordinate system info from client (recommended)
    client_width: Optional[int] = None
    client_height: Optional[int] = None

    # optional: send the clicked frame (not used by default)
    frame_b64: Optional[str] = None

    # NEW: return mask for this frame
    output: Literal["png", "rle"] = "png"
    mask_threshold: float = 0.0
    binarize: bool = True
    score_threshold: Optional[float] = None


class VideoPropagateRequest(BaseModel):
    session_id: str
    max_frames: Optional[int] = None
    only_frames: Optional[List[int]] = None
    output: Literal["png", "rle"] = "png"
    mask_threshold: float = 0.0
    binarize: bool = True
    score_threshold: Optional[float] = None


class VideoCloseRequest(BaseModel):
    session_id: str


# ----------------------------
# App + CORS
# ----------------------------
app = FastAPI(title="SAM3 FastAPI Server", version="2.1.0")

# In development/debug scenarios it can be useful to return full tracebacks for POST endpoints.
# Enable verbose error responses by setting environment variable SAM3_VERBOSE_ERRORS=1.
VERBOSE_ERRORS = os.getenv("SAM3_VERBOSE_ERRORS", "0") == "1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers that return full tracebacks when VERBOSE_ERRORS=True.
async def _http_exception_handler(request: Request, exc: HTTPException):
    if VERBOSE_ERRORS and getattr(exc, "status_code", 500) >= 500:
        traces = []
        cur = exc
        while cur:
            traces.append("".join(traceback.format_exception(type(cur), cur, cur.__traceback__)))
            cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        return PlainTextResponse("\n----\n".join(traces), status_code=exc.status_code)
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


async def _generic_exception_handler(request: Request, exc: Exception):
    if VERBOSE_ERRORS:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        cur = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        while cur:
            tb += "\nCaused by:\n" + "".join(traceback.format_exception(type(cur), cur, cur.__traceback__))
            cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        return PlainTextResponse(tb, status_code=500)
    return PlainTextResponse("Internal server error", status_code=500)


app.add_exception_handler(HTTPException, _http_exception_handler)
app.add_exception_handler(Exception, _generic_exception_handler)

# ----------------------------
# Model Loading
# ----------------------------
DEVICE = os.getenv("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if (DEVICE.startswith("cuda") and os.getenv("SAM3_FP16", "1") == "1") else torch.float32


def _get_hf_token_from_env_or_file() -> Optional[str]:
    """Look for a HuggingFace token in common env vars or a mounted token file.

    Returns the token string or None if not found.
    """
    env_vars = ["HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"]
    for v in env_vars:
        t = os.environ.get(v)
        if t:
            return t.strip()

    # Common mounted token file location
    possible_paths = ["/root/.huggingface/token", os.path.expanduser("~/.huggingface/token")]
    for p in possible_paths:
        try:
            if p and os.path.exists(p):
                with open(p, "r") as f:
                    t = f.read().strip()
                    if t:
                        return t
        except Exception:
            pass

    return None


def _check_hf_access(repo_id: str = "facebook/sam3") -> None:
    """Verify that the running container has access to a gated HF repo.

    If a token is available it will be used to query model_info. On failure this
    function raises a RuntimeError with guidance for the operator.
    """
    try:
        from huggingface_hub import HfApi
    except Exception:
        # huggingface_hub should be available via transformers; if not, skip check
        return

    token = _get_hf_token_from_env_or_file()
    if not token:
        # No token found — warn and raise so the server fails fast when the repo is gated
        raise RuntimeError(
            "Hugging Face token not found. Access to gated model 'facebook/sam3' requires a token.\n"
            "Provide a token via one of: HUGGINGFACE_HUB_TOKEN env var, HF_HUB_TOKEN env var, or a mounted file at /root/.huggingface/token.\n"
            "See README or pass -e HUGGINGFACE_HUB_TOKEN=hf_xxx when running the container."
        )

    try:
        api = HfApi()
        # model_info will raise if the token is invalid or lacks access
        info = api.model_info(repo_id, token=token)
        # success — optionally log minimal info
        print(f"Hugging Face access check: model '{repo_id}' reachable (gated={getattr(info, 'gated', 'unknown')}).")
    except Exception as e:
        raise RuntimeError(
            f"Hugging Face access check failed for repo '{repo_id}': {e}\n"
            "Ensure your token has 'read' scope and that you've been granted access to the gated model."
        )


# Perform HF gated-repo access check early so failures are clearer than deep import errors.
# If no token is present, wait until one appears (mounted file or env var). To skip waiting
# set SAM3_SKIP_HF_CHECK=1 in the environment.
if os.getenv("SAM3_SKIP_HF_CHECK", "0") == "1":
    print("Skipping Hugging Face access check (SAM3_SKIP_HF_CHECK=1).")
else:
    # Wait for a token or mounted token file to appear. This allows starting the container
    # and mounting a token file afterward (or having an orchestration system provide it).
    token = _get_hf_token_from_env_or_file()
    while not token:
        print(
            "Waiting for Hugging Face token. Provide one via env HUGGINGFACE_HUB_TOKEN/HF_TOKEN or mount /root/.huggingface/token."
        )
        time.sleep(5)
        token = _get_hf_token_from_env_or_file()

    try:
        _check_hf_access("facebook/sam3")
    except Exception:
        # Fail fast — better to stop container startup than have uvicorn crash later with an opaque trace
        raise

# PCS (text / boxes)
HF_TOKEN = _get_hf_token_from_env_or_file()

# Pass the HF token explicitly to from_pretrained so downloads use the authenticated client
# Ensure the token is available in common env vars so huggingface_hub/transformers internals
# that read from environment will pick it up as well. This covers versions that ignore the
# use_auth_token argument in some code paths.
if HF_TOKEN:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HF_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_TOKEN", HF_TOKEN)

# Rely on environment variables for authentication (HF token propagated above).
# Some transformers builds pass unknown kwargs through to the model constructor,
# which causes errors (e.g., TypeError: __init__() got an unexpected keyword 'use_auth_token').
# To be robust across versions, do not pass use_auth_token here; the HF token is already
# available via env vars (HUGGINGFACE_HUB_TOKEN / HF_HUB_TOKEN / HUGGINGFACE_TOKEN).
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE, dtype=DTYPE)
model.eval()

# Tracker (image)
if HAS_TRACKER:
    tracker_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    tracker_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(DEVICE, dtype=DTYPE)
    tracker_model.eval()

# Tracker (video)
if HAS_VIDEO_TRACKER:
    video_tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    video_tracker_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(DEVICE, dtype=DTYPE)
    video_tracker_model.eval()

# ----------------------------
# Video storage + session registry
# ----------------------------
VIDEO_DIR = os.getenv("SAM3_VIDEO_DIR", "/tmp/sam3_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# video_id -> {"path": str, "frames": Optional[List[PIL.Image.Image]], "ts": float, "width": int, "height": int, "num_frames": int}
VIDEO_FILES: Dict[str, Dict[str, Any]] = {}
# session_id -> {"mode": "tracker", "session": obj, "video_id": str, "ts": float, "width": int, "height": int, "num_frames": int}
VIDEO_SESSIONS: Dict[str, Dict[str, Any]] = {}

MAX_VIDEOS = int(os.getenv("SAM3_MAX_VIDEOS", "10"))
MAX_SESSIONS = int(os.getenv("SAM3_MAX_SESSIONS", "10"))
SESSION_TTL_SEC = int(os.getenv("SAM3_SESSION_TTL_SEC", "1800"))  # 30 min


def _cleanup_registry():
    now = time.time()

    expired = [sid for sid, v in VIDEO_SESSIONS.items() if (now - float(v.get("ts", now))) > SESSION_TTL_SEC]
    for sid in expired:
        VIDEO_SESSIONS.pop(sid, None)

    if len(VIDEO_SESSIONS) > MAX_SESSIONS:
        oldest = sorted(VIDEO_SESSIONS.items(), key=lambda kv: kv[1].get("ts", 0.0))
        for sid, _ in oldest[: max(0, len(VIDEO_SESSIONS) - MAX_SESSIONS)]:
            VIDEO_SESSIONS.pop(sid, None)

    if len(VIDEO_FILES) > MAX_VIDEOS:
        oldest = sorted(VIDEO_FILES.items(), key=lambda kv: kv[1].get("ts", 0.0))
        for vid, meta in oldest[: max(0, len(VIDEO_FILES) - MAX_VIDEOS)]:
            path = meta.get("path")
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            VIDEO_FILES.pop(vid, None)


def _require_video(video_id: str) -> Dict[str, Any]:
    meta = VIDEO_FILES.get(video_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown video_id.")
    return meta


def _ensure_video_frames_loaded(video_id: str) -> Dict[str, Any]:
    meta = _require_video(video_id)
    if meta.get("frames") is None:
        if not HAS_VIDEO_TRACKER:
            raise HTTPException(status_code=501, detail="Video utilities not available in this transformers install.")
        try:
            frames, _ = load_video(meta["path"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load video frames: {e}")

        # Normalize to PIL RGB for consistent downstream behavior
        try:
            frames_pil = [frame_to_pil(f) for f in frames]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to normalize frames to PIL: {e}")

        meta["frames"] = frames_pil
        first = frames_pil[0]
        w, h = first.size
        meta["width"] = w
        meta["height"] = h
        meta["num_frames"] = len(frames_pil)

    meta["ts"] = time.time()
    return meta


# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    _cleanup_registry()
    return {
        "ok": True,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "tracker_enabled": HAS_TRACKER,
        "video_tracker_enabled": HAS_VIDEO_TRACKER,
        "rle_enabled": HAS_RLE,
        "videos_cached": len(VIDEO_FILES),
        "sessions_active": len(VIDEO_SESSIONS),
        "session_ttl_sec": SESSION_TTL_SEC,
    }


# ----------------------------
# /segment/text  (SAM3 PCS)
# ----------------------------
@app.post("/segment/text", response_model=SegmentResponse)
@torch.inference_mode()
def segment_text(req: TextSegmentRequest):
    _cleanup_registry()

    img = decode_base64_image(req.image_b64)

    input_boxes = None
    input_boxes_labels = None
    if req.boxes:
        input_boxes = [[[b.x1, b.y1, b.x2, b.y2] for b in req.boxes]]
        input_boxes_labels = [[int(b.label) for b in req.boxes]]

    try:
        inputs = processor(
            images=img,
            text=req.text,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        )
        inputs = _to_device(inputs, DEVICE)
        inputs = cast_batchencoding_floats(inputs, DTYPE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processor call failed: {e}")

    outputs = model(**inputs)

    try:
        post = processor.post_process_instance_segmentation(
            outputs,
            threshold=req.threshold,
            mask_threshold=req.mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"post_process_instance_segmentation failed: {e}")

    masks = post.get("masks", None)
    boxes = post.get("boxes", None)
    scores = post.get("scores", None)

    if masks is None:
        return SegmentResponse(model="facebook/sam3", device=DEVICE, results=[])

    masks_np = masks.detach().cpu().numpy().astype(bool)
    boxes_np = boxes.detach().cpu().numpy().tolist() if boxes is not None else [None] * masks_np.shape[0]
    scores_np = scores.detach().cpu().numpy().tolist() if scores is not None else [None] * masks_np.shape[0]

    results: List[MaskResult] = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        score_val = float(scores_np[i]) if scores_np[i] is not None else None
        box_val = boxes_np[i] if boxes_np[i] is not None else None

        if req.output == "png":
            results.append(MaskResult(score=score_val, box_xyxy=box_val, png_b64=mask_to_base64_png(m)))
        else:
            results.append(MaskResult(score=score_val, box_xyxy=box_val, rle=encode_rle(m)))

    return SegmentResponse(model="facebook/sam3", device=DEVICE, results=results)


# ----------------------------
# /segment/visual  (SAM3 Tracker image PVS)
# ----------------------------
@app.post("/segment/visual", response_model=SegmentResponse)
@torch.inference_mode()
def segment_visual(req: VisualSegmentRequest):
    _cleanup_registry()

    if not HAS_TRACKER:
        raise HTTPException(
            status_code=501,
            detail="Sam3TrackerProcessor/Sam3TrackerModel not available in this transformers install.",
        )

    if (not req.points) and (not req.boxes) and (not req.text):
        raise HTTPException(status_code=400, detail="Provide at least one of: points, boxes, text.")

    img = decode_base64_image(req.image_b64)

    input_points = None
    input_labels = None
    input_boxes = None

    if req.points:
        input_points = [[[[p.x, p.y] for p in req.points]]]
        input_labels = [[[int(p.label) for p in req.points]]]

    if req.boxes:
        input_boxes = [[[b.x1, b.y1, b.x2, b.y2] for b in req.boxes]]

    processor_kwargs: Dict[str, Any] = dict(images=img, return_tensors="pt")
    if req.text:
        processor_kwargs["text"] = req.text
    if input_points is not None:
        processor_kwargs["input_points"] = input_points
        processor_kwargs["input_labels"] = input_labels
    if input_boxes is not None:
        processor_kwargs["input_boxes"] = input_boxes

    try:
        inputs = tracker_processor(**processor_kwargs)
        inputs = _to_device(inputs, DEVICE)
        inputs = cast_batchencoding_floats(inputs, DTYPE)
    except TypeError:
        if "input_points" in processor_kwargs:
            processor_kwargs["points"] = processor_kwargs.pop("input_points")
            processor_kwargs["labels"] = processor_kwargs.pop("input_labels")
        inputs = tracker_processor(**processor_kwargs)
        inputs = _to_device(inputs, DEVICE)
        inputs = cast_batchencoding_floats(inputs, DTYPE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tracker processor call failed: {e}")

    try:
        outputs = tracker_model(**inputs, multimask_output=req.multimask_output)
    except TypeError:
        outputs = tracker_model(**inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tracker model call failed: {e}")

    masks = None
    post_errs: List[str] = []

    try:
        masks = tracker_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs.get("reshaped_input_sizes", None),
        )[0]
    except Exception as e:
        post_errs.append(str(e))

    if masks is None:
        try:
            masks = tracker_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"],
            )[0]
        except Exception as e:
            post_errs.append(str(e))

    if masks is None:
        try:
            masks = tracker_processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]
        except Exception as e:
            post_errs.append(str(e))

    if masks is None:
        raise HTTPException(status_code=500, detail=f"Tracker post_process_masks failed. Errors: {post_errs}")

    masks_np = masks.detach().float().cpu().numpy()
    if masks_np.ndim == 4:
        masks_np = masks_np.reshape(-1, masks_np.shape[-2], masks_np.shape[-1])

    scores = getattr(outputs, "iou_scores", None)
    scores_np = None
    if scores is not None:
        try:
            scores_np = scores.detach().float().cpu().numpy().reshape(-1).tolist()
        except Exception:
            scores_np = None

    results: List[MaskResult] = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i] >= float(req.threshold)
        score_val = float(scores_np[i]) if (scores_np and i < len(scores_np)) else None
        if req.output == "png":
            results.append(MaskResult(score=score_val, png_b64=mask_to_base64_png(m)))
        else:
            results.append(MaskResult(score=score_val, rle=encode_rle(m)))

    return SegmentResponse(model="facebook/sam3-tracker", device=DEVICE, results=results)


# ----------------------------
# Video endpoints (tracker video)
# ----------------------------
@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    _cleanup_registry()

    if not HAS_VIDEO_TRACKER:
        raise HTTPException(
            status_code=501,
            detail="Sam3TrackerVideoModel/Processor not available in this transformers install.",
        )

    filename = (file.filename or "").lower()
    if not filename.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    video_id = str(uuid4())
    path = os.path.join(VIDEO_DIR, f"{video_id}_{os.path.basename(file.filename)}")
    try:
        with open(path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {e}")

    VIDEO_FILES[video_id] = {"path": path, "frames": None, "ts": time.time()}
    return {"video_id": video_id}


@app.post("/video/session/init")
@torch.inference_mode()
def video_session_init(req: VideoSessionInitRequest):
    _cleanup_registry()

    if req.mode != "tracker":
        raise HTTPException(status_code=400, detail="Only mode='tracker' is implemented in this server.")
    if not HAS_VIDEO_TRACKER:
        raise HTTPException(status_code=501, detail="Sam3TrackerVideoModel/Processor not available.")

    meta = _ensure_video_frames_loaded(req.video_id)
    frames: List[Image.Image] = meta["frames"]

    inference_device = req.inference_device or DEVICE
    try:
        inference_session = video_tracker_processor.init_video_session(
            video=frames,
            inference_device=inference_device,
            dtype=DTYPE,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"init_video_session failed: {e}")

    w = int(meta["width"])
    h = int(meta["height"])
    n = int(meta["num_frames"])

    session_id = str(uuid4())
    VIDEO_SESSIONS[session_id] = {
        "mode": "tracker",
        "session": inference_session,
        "video_id": req.video_id,
        "ts": time.time(),
        "width": w,
        "height": h,
        "num_frames": n,
    }
    return {"session_id": session_id, "mode": "tracker", "video_id": req.video_id, "num_frames": n, "width": w, "height": h}


@app.post("/video/session/prompt")
@torch.inference_mode()
def video_session_prompt(req: VideoPromptRequest):
    """
    Add a visual prompt (points and/or boxes) at a given frame for a given object id,
    then RUN inference on that frame and return the mask(s) so the client can review.

    This also "primes" the session so /video/session/propagate can start correctly.
    """
    _cleanup_registry()

    entry = VIDEO_SESSIONS.get(req.session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    if entry.get("mode") != "tracker":
        raise HTTPException(status_code=400, detail="This endpoint is only for tracker sessions.")
    if not HAS_VIDEO_TRACKER:
        raise HTTPException(status_code=501, detail="Sam3TrackerVideoModel/Processor not available.")
    if (not req.points) and (not req.boxes):
        raise HTTPException(status_code=400, detail="Provide points and/or boxes for tracker-video prompting.")

    sess = entry["session"]
    num_frames = int(entry.get("num_frames", 0))
    if req.frame_idx < 0 or (num_frames and req.frame_idx >= num_frames):
        raise HTTPException(status_code=400, detail=f"frame_idx out of range (0..{num_frames-1})")

    server_w = int(entry.get("width", 0))
    server_h = int(entry.get("height", 0))
    if server_w <= 0 or server_h <= 0:
        raise HTTPException(status_code=500, detail="Server video size metadata missing (width/height).")

    # ---- coordinate rescaling (client -> server decoded video space) ----
    sx = sy = 1.0
    if req.client_width and req.client_height:
        if req.client_width <= 0 or req.client_height <= 0:
            raise HTTPException(status_code=400, detail="client_width/client_height must be positive if provided.")
        sx = server_w / float(req.client_width)
        sy = server_h / float(req.client_height)

    def _clip(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def _scale_xy(x: int, y: int) -> tuple[int, int]:
        xs = int(round(x * sx))
        ys = int(round(y * sy))
        xs = _clip(xs, 0, server_w - 1)
        ys = _clip(ys, 0, server_h - 1)
        return xs, ys

    def _scale_box(b: Box) -> Box:
        x1, y1 = _scale_xy(b.x1, b.y1)
        x2, y2 = _scale_xy(b.x2, b.y2)
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        if server_w > 1:
            x2 = max(x2, x1 + 1)
        if server_h > 1:
            y2 = max(y2, y1 + 1)
        x2 = _clip(x2, 0, server_w - 1)
        y2 = _clip(y2, 0, server_h - 1)
        return Box(x1=x1, y1=y1, x2=x2, y2=y2, label=b.label)

    points = req.points
    boxes = req.boxes

    if (sx != 1.0) or (sy != 1.0):
        if points:
            scaled_points: List[Point] = []
            for p in points:
                x, y = _scale_xy(p.x, p.y)
                scaled_points.append(Point(x=x, y=y, label=p.label))
            points = scaled_points
        if boxes:
            boxes = [_scale_box(b) for b in boxes]

    # ---- Build SAM3 tracker-video inputs ----
    input_points = None
    input_labels = None
    input_boxes = None

    if points:
        input_points = [[[[p.x, p.y] for p in points]]]
        input_labels = [[[int(p.label) for p in points]]]

    if boxes:
        # simplest: first box as object prompt
        b = boxes[0]
        input_boxes = [[[b.x1, b.y1, b.x2, b.y2]]]

    # 1) Add inputs
    try:
        video_tracker_processor.add_inputs_to_inference_session(
            inference_session=sess,
            frame_idx=req.frame_idx,
            obj_ids=req.obj_id,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            clear_old_inputs=req.clear_old_inputs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"add_inputs_to_inference_session failed: {e}")

    # 2) RUN inference on that frame (critical!)
    try:
        out = video_tracker_model(inference_session=sess, frame_idx=req.frame_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video_tracker_model forward failed: {e}")

    # 3) Post-process masks to original video size
    post_errs: List[str] = []
    masks = None
    try:
        masks = video_tracker_processor.post_process_masks(
            [out.pred_masks],
            original_sizes=[[server_h, server_w]],
            mask_threshold=req.mask_threshold,
            binarize=req.binarize,
        )[0]
    except Exception as e:
        post_errs.append(str(e))

    if masks is None:
        try:
            masks = video_tracker_processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[server_h, server_w]],
            )[0]
        except Exception as e:
            post_errs.append(str(e))

    if masks is None:
        raise HTTPException(status_code=500, detail=f"video post_process_masks failed. Errors: {post_errs}")

    masks_np = masks.detach().float().cpu().numpy()
    if masks_np.ndim == 4:
        masks_np = masks_np[0]
    if masks_np.ndim == 4:
        masks_np = masks_np.reshape(-1, masks_np.shape[-2], masks_np.shape[-1])

    # Scores (best-effort: different builds expose different fields)
    scores = getattr(out, "iou_scores", None)
    if scores is None:
        scores = getattr(out, "object_score_logits", None)

    scores_np = None
    if scores is not None:
        try:
            scores_np = scores.detach().float().cpu().numpy().reshape(-1).tolist()
        except Exception:
            scores_np = None

    frame_results: List[Dict[str, Any]] = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        mb = (m >= 0.5) if not req.binarize else (m.astype(np.float32) >= 0.5)

        score_val = float(scores_np[i]) if (scores_np and i < len(scores_np)) else None
        if req.score_threshold is not None and score_val is not None and score_val < float(req.score_threshold):
            continue

        if req.output == "png":
            frame_results.append({"score": score_val, "png_b64": mask_to_base64_png(mb)})
        else:
            frame_results.append({"score": score_val, "rle": encode_rle(mb)})

    # record last prompted frame (used as propagate start)
    entry["last_prompt_frame_idx"] = int(req.frame_idx)
    entry["ts"] = time.time()

    return {
        "ok": True,
        "session_id": req.session_id,
        "frame_idx": req.frame_idx,
        "obj_id": req.obj_id,
        "server_width": server_w,
        "server_height": server_h,
        "scale_x": sx,
        "scale_y": sy,
        "frame_masks": frame_results,
    }


@app.post("/video/session/propagate")
@torch.inference_mode()
def video_session_propagate(req: VideoPropagateRequest):
    _cleanup_registry()

    entry = VIDEO_SESSIONS.get(req.session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    if entry["mode"] != "tracker":
        raise HTTPException(status_code=400, detail="This endpoint is only for tracker sessions.")
    if not HAS_VIDEO_TRACKER:
        raise HTTPException(status_code=501, detail="Sam3TrackerVideoModel/Processor not available.")

    sess = entry["session"]
    width = int(entry["width"])
    height = int(entry["height"])
    num_frames = int(entry.get("num_frames", 0))

    start_frame_idx = entry.get("last_prompt_frame_idx", None)
    if start_frame_idx is None:
        raise HTTPException(
            status_code=400,
            detail="No prompted frame found. Call /video/session/prompt first (it runs inference and sets the start frame).",
        )
    start_frame_idx = int(start_frame_idx)

    wanted = set(req.only_frames) if req.only_frames else None
    if wanted is not None:
        for f in wanted:
            if f < 0 or (num_frames and f >= num_frames):
                raise HTTPException(status_code=400, detail=f"only_frames contains out-of-range index: {f}")

    outputs_per_frame: Dict[int, List[Dict[str, Any]]] = {}

    try:
        iterator = video_tracker_model.propagate_in_video_iterator(
            sess,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=req.max_frames,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"propagate_in_video_iterator failed: {e}")

    for out in iterator:
        frame_idx = int(getattr(out, "frame_idx", -1))
        if frame_idx < 0:
            # if missing, fall back to monotonic
            frame_idx = len(outputs_per_frame)

        if wanted is not None and frame_idx not in wanted:
            continue

        post_errs: List[str] = []
        masks = None
        try:
            masks = video_tracker_processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[height, width]],
                mask_threshold=req.mask_threshold,
                binarize=req.binarize,
            )[0]
        except Exception as e:
            post_errs.append(str(e))

        if masks is None:
            try:
                masks = video_tracker_processor.post_process_masks(
                    [out.pred_masks],
                    original_sizes=[[height, width]],
                )[0]
            except Exception as e:
                post_errs.append(str(e))

        if masks is None:
            raise HTTPException(status_code=500, detail=f"video post_process_masks failed. Errors: {post_errs}")

        masks_np = masks.detach().float().cpu().numpy()
        if masks_np.ndim == 4:
            masks_np = masks_np[0]
        if masks_np.ndim == 4:
            masks_np = masks_np.reshape(-1, masks_np.shape[-2], masks_np.shape[-1])

        scores = getattr(out, "iou_scores", None)
        if scores is None:
            scores = getattr(out, "object_score_logits", None)

        scores_np = None
        if scores is not None:
            try:
                scores_np = scores.detach().float().cpu().numpy().reshape(-1).tolist()
            except Exception:
                scores_np = None

        frame_results: List[Dict[str, Any]] = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            mb = (m >= 0.5) if not req.binarize else (m.astype(np.float32) >= 0.5)

            score_val = float(scores_np[i]) if (scores_np and i < len(scores_np)) else None
            if req.score_threshold is not None and score_val is not None and score_val < float(req.score_threshold):
                continue

            if req.output == "png":
                frame_results.append({"score": score_val, "png_b64": mask_to_base64_png(mb)})
            else:
                frame_results.append({"score": score_val, "rle": encode_rle(mb)})

        outputs_per_frame[frame_idx] = frame_results

        if wanted is not None and len(outputs_per_frame) >= len(wanted):
            break

    entry["ts"] = time.time()
    return {
        "session_id": req.session_id,
        "video_id": entry["video_id"],
        "start_frame_idx": start_frame_idx,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frames": outputs_per_frame,
    }


@app.post("/video/session/close")
def video_session_close(req: VideoCloseRequest):
    _cleanup_registry()
    if req.session_id in VIDEO_SESSIONS:
        del VIDEO_SESSIONS[req.session_id]
        return {"ok": True}
    return {"ok": False, "detail": "Unknown session_id."}