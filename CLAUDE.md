# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`model_eval` is a Python package for evaluating model outputs (e.g., comparing WRF weather model runs). It uses UV for environment management, hachling for the build system, and targets Python >= 3.11.

## Code Style

- **Line length**: 120 characters (both black and ruff)
- **Formatter**: black with `skip-string-normalization`
- **Linter**: ruff targeting py311, with relative imports banned (`ban-relative-imports = "all"`)
- **Imports**: Use absolute imports (e.g., `from model_eval.module import X`, not relative)
- **Indent**: 4 spaces, UTF-8, LF line endings

## Architecture

- `model_eval/` — Main package. Version defined in `__init__.py`.
- `model_eval/tests/` — Test directory (pytest). Tests are excluded from sdist builds.
- `docs/` — MkDocs documentation with Material theme and mkdocstrings for API reference.
- `conda/meta.yaml` — Conda package recipe.

The project is in early development; core implementation modules are still being built out.
