# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Image compression using discrete cosine transform (DCT) and a quantization matrix for grayscale images (from README).

Repository state (detected)
- Source code, tests, packaging, and linting configs are not yet present; only `README.md` exists.

Commands
- Python environment (Windows, PowerShell):
  - Create venv: `py -3 -m venv .venv`
  - Activate: `.\\.venv\\Scripts\\Activate.ps1`
- Dependencies (when `requirements.txt` is added):
  - `pip install -r requirements.txt`
- Linting (when configured/installed):
  - Ruff: `ruff check .`  (auto-fix: `ruff check . --fix`)
  - Black (format): `black .`
- Tests (when `pytest` is added):
  - All tests: `pytest -q`
  - Single test: `pytest tests/test_file.py::TestClass::test_name -q`
  - By pattern: `pytest -q -k "pattern"`

Architecture and structure
- No code files are present yet; once modules are added, prefer clear separation of concerns (core compression logic, I/O, CLI/entrypoint, and tests).

Important project files
- README.md: "Uses discrete cosine transformations and a quantization matrix to compress black and white images while maintaining quality"
- No CLAUDE, Cursor, or Copilot rules detected.
