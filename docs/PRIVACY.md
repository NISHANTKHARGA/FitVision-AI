# Privacy & Data Handling

FitVision AI is a **local-first** application. No account, no cloud, no telemetry.

| Data | Default behaviour |
|---|---|
| Your photo | Kept only for the current session inside `backend/data/uploads/<job>` (deleted right after the render), and copied next to the result in `backend/data/results/<job>` for the before/after view. Unsaved results **auto-expire after 24 h** (cleanup on server start). |
| Garment photo | Same lifecycle as your photo. |
| Generated try-ons | Stored in `backend/data/results/<job>`; persisted to history **only** when you tick "Save to history" or click "Save to history" on a result. |
| Measurement profile | Stored in local SQLite (`backend/data/fitvision.db`) on this machine. Never transmitted. |
| History metadata | Stored locally in SQLite; deleted with "Delete" in the UI (file + record). |

**Explicitly opting in to history** keeps the uploads and the result on disk
indefinitely (until you delete them). Unsaved results are NOT kept beyond the 24 h TTL.

Nothing in this codebase sends images or measurements to any external service.
The only network traffic the app generates itself is:
- one-time model/weight downloads from HuggingFace (FASHN weights ~2 GB, CLIP ViT-B/32 ~600 MB, EasyOCR ~100 MB, rembg u²net ~170 MB);
- none thereafter — inference is fully offline.
