SAM3 FastAPI Server
====================

Overview
--------
This small FastAPI server exposes a lightweight HTTP API around the SAM3 models (PCS, Tracker and Tracker-Video) from the `transformers` library. It supports:

- Text-conditioned segmentation (PCS): `/segment/text`
- Visual segmentation using points/boxes (Tracker image): `/segment/visual`
- Video tracking workflow (upload/init/prompt/propagate/close): `/video/*`
- Health/status endpoint: `/health`

The server is designed for clients that host the UI (images/videos) and call the server to run segmentation/tracking inference.

Quick start
-----------

1. Install runtime dependencies (example):

```bash
pip install fastapi uvicorn pillow numpy torch transformers python-multipart
# optional for RLE or COCO utilities
pip install pycocotools
# optional (recommended) for video decoding: install PyAV via conda
conda install -c conda-forge av
```

2. Ensure a Hugging Face token is available if access to gated model weights is required. Provide via one of:

- `HUGGINGFACE_HUB_TOKEN` or `HF_HUB_TOKEN` environment variable
- Or mount a token file at `/root/.huggingface/token` or `~/.huggingface/token`

3. Run the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Environment variables
---------------------

- `SAM3_DEVICE`: override device (default `cuda` if available else `cpu`)
- `SAM3_FP16`: set to `1` to prefer float16 on CUDA devices (default `1`)
- `SAM3_VIDEO_DIR`: directory to store uploaded videos (default `/tmp/sam3_videos`)
- `SAM3_MAX_VIDEOS`: max cached uploaded videos (default `10`)
- `SAM3_MAX_SESSIONS`: max active sessions (default `10`)
- `SAM3_SESSION_TTL_SEC`: session TTL in seconds (default `1800`)
- `SAM3_SKIP_HF_CHECK`: set to `1` to skip the initial HF gated-repo access check
- `SAM3_VERBOSE_ERRORS`: set to `1` to return full tracebacks in HTTP responses (dev only)

Hugging Face token env vars supported: `HUGGINGFACE_HUB_TOKEN`, `HF_HUB_TOKEN`, `HUGGINGFACE_TOKEN`, `HF_TOKEN`.

Endpoints
---------

- `GET /health`
	- Returns server health, device and optional feature flags (tracker/video/rle enabled).

- `POST /segment/text`  (Text-conditioned segmentation — PCS)
	- Request schema: `image_b64` (base64-encoded RGB image), `text` (concept), optional `boxes` (xyxy + label), `output`=`png|rle`.
	- Returns: `model`, `device`, and `results` array where each item contains `score`, `png_b64` (mask), or `rle`.

- `POST /segment/visual`  (Visual segmentation — PVS / tracker image)
	- Request accepts `image_b64`, optional `text`, `points`, `boxes`, `output`, thresholds and multimask flag.
	- Requires tracker support in the installed `transformers` build; returns masks similar to `/segment/text`.

- `POST /video/upload` (multipart file)
	- Upload a video file (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`). Returns `{ "video_id": "..." }`.

- `POST /video/session/init`
	- Body: `{ "video_id": "...", "mode": "tracker", "inference_device": optional }`.
	- Initializes a tracker session and returns `{ "session_id": "...", "num_frames": N, "width": W, "height": H }`.

- `POST /video/session/prompt`
	- Add a visual prompt (points/boxes) at a frame for an object id, runs inference for that frame and returns masks.
	- Body includes `session_id`, `frame_idx`, `obj_id`, `points`/`boxes`, optional `client_width`/`client_height` for coordinate scaling.

- `POST /video/session/propagate`
	- Propagate the last prompted object through the video. Returns masks per requested frames.

- `POST /video/session/close`
	- Free server-side session memory for `session_id`.

JSON examples
-------------

Minimal `segment/text` example (curl):

```bash
curl -X POST http://127.0.0.1:8000/segment/text \
	-H 'Content-Type: application/json' \
	-d '{"image_b64":"<BASE64_IMAGE>", "text":"person", "output":"png"}'
```

`image_b64` can be produced locally with Python:

```python
import base64
from pathlib import Path
img_b64 = base64.b64encode(Path('image.jpg').read_bytes()).decode('utf-8')
```

Notes on outputs
----------------

- `png` output: masks are returned as base64-encoded PNG images (`png_b64`).
- `rle` output: COCO-style RLE dictionaries are returned (requires `pycocotools`).
- Scores and bounding boxes are included where available.

Video workflow notes
--------------------

- Video utilities depend on `transformers` video tracker classes and `pyav` for decoding. If video tracker features are missing, video endpoints will return `501`.
- The server caches uploaded videos and maintains lightweight session state. Use `SAM3_MAX_VIDEOS`, `SAM3_MAX_SESSIONS` and `SAM3_SESSION_TTL_SEC` to tune memory.

Authentication and gated HF models
---------------------------------

This server attempts to load `facebook/sam3` via `transformers`. If the model repo is gated, you must provide a valid Hugging Face token (see above). Use `SAM3_SKIP_HF_CHECK=1` to skip the initial access check (not recommended unless you know the model is public or token is supplied later).

Troubleshooting
---------------

- If you get `501` responses for tracker/video endpoints, install a `transformers` build with `Sam3Tracker` and `Sam3TrackerVideo` support and ensure `pyav` is available for video decoding.
- If HF downloads fail, verify your token and that it's exported in the environment the server runs in.
- For detailed tracebacks during development, set `SAM3_VERBOSE_ERRORS=1` (do not enable in production).

Contributing
------------

The server is intentionally small — contributions that add tests, sample clients, or improved error messages are welcome.

Files
-----

- `server.py`: the FastAPI application and endpoint implementations.

License
-------
See the project `LICENSE` file at the repository root.

----
Generated README for the server: contact repository maintainers for further customizations.

Node.js example
---------------

Minimal Node.js examples showing how to call the server and save a PNG mask.

1) Simple `segment/text` POST using `axios` (install with `npm i axios`):

```javascript
const fs = require('fs');
const axios = require('axios');

async function run() {
	const imgB64 = fs.readFileSync('image.jpg', { encoding: 'base64' });

	const payload = {
		image_b64: imgB64,
		text: 'person',
		output: 'png'
	};

	const res = await axios.post('http://127.0.0.1:8000/segment/text', payload);
	console.log('Response:', res.data);

	if (res.data.results && res.data.results[0] && res.data.results[0].png_b64) {
		const maskB64 = res.data.results[0].png_b64;
		fs.writeFileSync('mask.png', Buffer.from(maskB64, 'base64'));
		console.log('Wrote mask.png');
	}
}

run().catch(err => console.error(err.response?.data || err.message));
```

2) Upload a video (multipart) using `form-data` + `axios` (install with `npm i axios form-data`):

```javascript
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

async function uploadVideo() {
	const form = new FormData();
	form.append('file', fs.createReadStream('video.mp4'));

	const res = await axios.post('http://127.0.0.1:8000/video/upload', form, {
		headers: form.getHeaders(),
	});
	console.log('Uploaded video_id:', res.data.video_id);
}

uploadVideo().catch(err => console.error(err.response?.data || err.message));
```

Notes:

- The examples assume the server runs on `http://127.0.0.1:8000`.
- Replace filenames (`image.jpg`, `video.mp4`) with your local files. For browser clients, send the same JSON payloads from the frontend.
- For more advanced flows (video session init / prompt / propagate) send the JSON bodies described above to the corresponding `/video/session/*` endpoints.
