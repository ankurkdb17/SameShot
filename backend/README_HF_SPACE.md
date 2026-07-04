---
title: SameShot
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# SameShot backend

Near-duplicate image detection API (DINOv2 + FAISS + SIFT). See the main
repo README for architecture details.

**Note on storage:** this Space's free tier uses ephemeral disk. Uploaded
pool images and the FAISS index are lost when the Space restarts or sleeps
from inactivity — re-upload your reference images after a cold start.

Set `CORS_ORIGINS` in this Space's "Variables and secrets" settings to your
deployed frontend URL (e.g. `https://your-app.vercel.app`).
