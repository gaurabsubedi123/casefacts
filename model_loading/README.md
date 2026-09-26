# Model Portal

Download local AI models, plug one in, and ask questions about your OCR'd
documents, all in the browser and all on your own computer.

It's built to be handed to someone as **one .exe**. They double-click it, the
browser opens, and a *Getting started* checklist walks them through the rest.
Nobody has to open a terminal.

---

## What it does

**Models tab**

* **Recommended for this computer.** It shows the best size of Qwen 2.5,
  DeepSeek-R1, Qwen 3, Llama 3.1 and Gemma 3 that runs *entirely on this
  computer's GPU*, each one click to download.
* **Find a model.** Search the whole Ollama library, or Hugging Face GGUF
  repositories. Every version shows its download size and a verdict for this
  machine:

  | Badge | Meaning |
  |---|---|
  | **Fits on GPU** | Runs entirely in GPU memory. Fast. |
  | **Slow — partly on CPU** | Too big for the GPU. It works, but several times slower. |
  | **CPU only** | No usable GPU; runs from RAM. Slow. |
  | **Too big** | Bigger than GPU and RAM together. You're asked before downloading. |

* Downloads show a progress bar and can be stopped. Pressing Download again
  resumes where it stopped.
* **Use** plugs a model in. The portal picks the largest context window that
  keeps the model wholly on the GPU, loads it, checks where Ollama actually put
  it, and shrinks the context and reloads if part of it spilled to the CPU. The
  first model you download is plugged in automatically.
* **Cloud models are refused.** Some Ollama entries run on Ollama's servers
  rather than your computer, so a question would send your document there.
  They're marked *cloud only* and can't be downloaded or used.

**Documents tab**: drag in PDFs, text files, or a whole folder. Files are
copied into the portal's own data folder, and the originals are never touched.

* **PDFs must already be OCR'd** (have a text layer). A raw scan is **refused
  with an error** telling you to run it through ocrtool first. A file where only
  a few pages lack text is accepted, and you're told which pages.
* ocrtool's `.txt` files keep their real page numbers.

**Ask tab**: tick documents, type a question, press Ask (or Ctrl+Enter).

* Asking with no document selected, no model plugged in, or no question gives
  an error straight away instead of an answer.
* The model must back every finding with a word-for-word quote. Each quote is
  checked against the page before you see it:
  **verified** / **close** (OCR noise) / **joined** (read across columns) /
  **wrong page** (citation corrected) / **unverified** (not found; shown last).
* Click a citation and the page opens on the right with the quote highlighted.
  For a PDF you can switch to the original.
* Small documents are read whole. Larger ones are searched: by keyword, and by
  meaning too if the small `nomic-embed-text` model is downloaded (it's in the
  recommended list).
* Reasoning models (DeepSeek-R1, Qwen 3) think first, and their reasoning can
  be expanded under the answer.

---

## What the other computer needs

1. **The .exe.** That's the whole program.
2. **Ollama.** The exe can't contain it, because Ollama ships gigabytes of GPU
   runtime. On Windows the portal shows an **Install Ollama** button that
   downloads and opens Ollama's official installer. On macOS and Linux it shows
   the download link.
3. **Internet access, for downloading models only.** Documents and questions
   never leave the computer.

The GPU is detected on each start. NVIDIA is read through `nvidia-smi` (it
ships with the driver). On Windows, AMD cards are read from the display-adapter
registry, and on Linux AMD cards are read from the system. Apple Silicon uses
its shared memory. Intel integrated graphics are shown but not counted, because
Ollama doesn't run models on them.

The portal's data lives in:

| OS | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\ModelPortal` |
| macOS | `~/Library/Application Support/ModelPortal` |
| Linux | `~/.modelportal` |

Set `MODELPORTAL_HOME` to put it somewhere else.

---

## Building the Windows .exe, step by step

PyInstaller builds only for the system it runs on, so **build the Windows .exe
on Windows**. It takes one installer and one double-click.

### Step 1 – Install Python (once per computer)

1. Go to <https://www.python.org/downloads/> and click the yellow
   **Download Python 3.x** button.
2. Run the installer. On the first screen, **tick "Add python.exe to PATH"** at
   the bottom, then click **Install Now**.
3. When it says "Setup was successful", click **Close**.

Many computers already have a `python.exe` inside `WindowsApps`. That is only a
Microsoft Store shortcut, not Python. `build_exe.bat` avoids it by using the
`py` launcher that the python.org installer adds.

### Step 2 – Build

1. In File Explorer, open this `model_loading` folder.
2. Double-click **`build_exe.bat`**. A console window:
   * creates a build environment (`.venv-win`),
   * downloads Flask, pypdf and PyInstaller,
   * builds the exe.

   The first build takes 2–5 minutes; later ones are faster.
3. It ends with **`Built: …\dist\ModelPortal.exe`**. Press any key to close.

If it ends with **"The build failed"**, the red text above that line says why.
The usual cause is Python not being on PATH: re-run the Python installer,
choose **Modify**, and tick **Add Python to environment variables**.

### Step 3 – Test it

Double-click `dist\ModelPortal.exe`. The browser should open the portal. Ask one
question before sending the exe to anyone.

If Ollama runs inside WSL on the build computer, the portal usually still finds
it. If it reports Ollama as not installed, try the **Install Ollama** button:
that is exactly what a fresh computer will go through.

### Rebuilding

After any change to the code, double-click `build_exe.bat` again. It replaces
the old exe. On Linux or macOS, `make exe` builds `dist/ModelPortal` for that
system instead.

---

## Running the exe (for the person who receives it)

Send only `dist\ModelPortal.exe`, one file of about 20 MB. Email often blocks
`.exe` attachments, so share it through Google Drive, OneDrive or a USB stick.

1. **Double-click `ModelPortal.exe`.** If Windows shows "Windows protected your
   PC", click **More info**, then **Run anyway**. This appears once because the
   exe is not code-signed.
2. **Keep the black window open.** It shows the address and is the off switch:
   closing it stops the portal. The browser opens by itself; if it does not,
   type the address from the black window (usually `http://127.0.0.1:5050`).
3. **Install Ollama if the yellow bar asks.** Click **Install Ollama** and click
   through the installer that opens. The bar clears by itself when Ollama runs.
4. **Download a model.** On the **Models** tab, under *Recommended for this
   computer*, click **Download** next to `qwen2.5`. The first model downloaded
   is plugged in automatically.
5. **Add documents.** On the **Documents** tab, drag in OCR'd PDFs or text files.
6. **Ask.** On the **Ask** tab, tick the documents, type a question, and click
   **Ask**.

A *Getting started* checklist on the Ask tab tracks these steps.

---

## Running from source

```bash
make install
make run          # opens http://127.0.0.1:5050 (or the next free port)
make test         # 37 tests, no model or network needed
```

Port 5050 was chosen so it doesn't collide with ocrtool or the wiki on 5000, or
casefacts on 5001.

---

## How it's put together

```
modelportal/
  __main__.py   start: detect hardware, start Ollama, free port, open browser
  hardware.py   GPU/RAM/disk detection, fit verdicts, context sizing
  catalog.py    Ollama library + Hugging Face search, recommended models
  ollama.py     Ollama HTTP client: pull, load, chat (streamed), embed
  documents.py  the document library; PDF/text extraction; the OCR check
  retrieve.py   whole-document or keyword+meaning search, page-bounded chunks
  answer.py     prompt, JSON parsing, quote verification
  jobs.py       background tasks with progress and stop
  web.py        Flask app, localhost-only, with cross-site request guards
packaging/      PyInstaller entry point and spec
```

Only two dependencies, Flask and pypdf, to keep the .exe small and predictable.
