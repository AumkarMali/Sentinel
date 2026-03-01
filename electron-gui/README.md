# Vibe Gamer — Desktop app

Electron UI for the AI agent (task input, settings, floating dot). The backend still runs the Python agent when you start a task.

## How to run

From this folder (`electron-gui`):

```bash
npm install
npm start
```

**First time:** ensure the project root has the Python venv set up and `requirements.txt` installed (see root README). Stockfish and the chess model go in the project root if you use the chess agent.

The app is plain **JavaScript** (no TypeScript). Entry point is `main.js`; renderer pages are in `renderer/`. Press **Escape** to quit.
