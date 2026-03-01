# Pie – AI Agent (Gemini + Chess Bot)

Unified GUI: type a task → screenshot is sent to **Gemini** → Gemini either starts the **chess bot** (if it sees a board) or controls the screen / comments in the log.

- **Chess agent:** YOLO piece detection + Stockfish; auto-plays on sites like Lichess/Chess.com.
- **Vision:** Gemini only. Get an API key at [Google AI Studio](https://aistudio.google.com/apikey).

## Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/AumkarMali/pie.git
   cd pie
   pip install -r requirements.txt
   ```

2. **Stockfish (required for chess)**  
   Download a Windows build from [Stockfish](https://stockfishchess.org/download/) and place `stockfish.exe` in the project root.

3. **Chess YOLO model (required for chess)**  
   Place the chess piece detection model as `chess_model.pt` in the project root.  
   Example: [NAKSTStudio/yolov8m-chess-piece-detection](https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection) – download `best.pt` and rename to `chess_model.pt`.

4. **Gemini API key**  
   In the app open **Settings** and enter your **Gemini API Key**. Get a key at [Google AI Studio](https://aistudio.google.com/apikey). Models: `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`.

## Run

The app uses the **Electron** UI (floating dot, task panel, settings). From the project root:

```bash
cd electron-gui
npm install
npm start
```

Then open the task window from the floating dot, enter a task (e.g. *play chess* or *open Notepad*), and start. The **Python backend** (`agent_backend.py`) runs headless: it takes the prompt, hides the app windows, takes screenshots, and runs the same flow as the reference implementation. When the task finishes (or you stop it), the windows reappear. Settings (Gemini API key, model, YOLO confidence, Stockfish depth, scan interval, click delay) match the reference `gui.py` and are stored in `config.json`. `gui.py` is reference only; the app does not use it to run tasks.
