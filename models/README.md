# Upstream model assets

Run the repository-local downloader from the project root:

```bash
python scripts/download_nesso_checkpoint.py
```

It writes the pinned Nesso-1 v1.0.0 checkpoint to
`models/nesso-1/v1.0.0/model.safetensors` and verifies its SHA-256 checksum.
Downloaded model files are ignored by Git.
