# AudioCAPTCHA-DSP

Exploiting AI Blindspots via Psychoacoustic Multirate Distortions.

Research framework for experimentally studying differences between human
intelligibility and automatic speech recognition robustness under controlled
psychoacoustic transformations.

## Status

Phase 1 — Scaffold. No DSP logic implemented yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run Tests

```bash
make test
```

## Run CLI

```bash
audiocaptcha --help
```

## License

Research Use Only. Not for production deployment.
