"""Coach's console (interface layer).

Planned:
  server.py     FastAPI: POST /simulate, /step (duel mode), /whatif
  static/       index.html + canvas pitch renderer + Chart.js panels

The console consumes ONLY the versioned replay JSON from `pitchsim.replay`; it
never imports the engine. That is what lets it be built against the stub and what
makes the offline demo (Gate 4) possible from cached replays alone.
"""
