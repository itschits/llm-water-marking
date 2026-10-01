# Watermark Lab web app

Run watermark experiments from the browser instead of reading terminal output.
The **Run experiment** page generates a plain continuation and a watermarked
continuation, then shows the text, statistical signal, z-score, p-value, and
detection verdict side by side.

## Start the app

From the repository root, open two PowerShell terminals.

**Terminal 1 — API**

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn api.server:app --reload --port 8000
```

**Terminal 2 — web interface**

```powershell
Set-Location webapp
npm.cmd run dev
```

Open the URL printed by Vite (normally `http://localhost:5173`), then choose
**Run experiment** in the sidebar.

## Models

- **Qwen 2.5 1.5B** is the default Qwen model and should run faster than 3B.
- **GPT-2** is a smaller fast baseline and is suitable for CPU runs.
- **Qwen 2.5 3B** is public and needs no gated-model approval, but requires
  considerably more resources (about 12 GB RAM on CPU or 8 GB+ VRAM on GPU).

Only one experiment can run at a time. The first run of a model downloads its
Hugging Face weights; subsequent runs use the local cache.
