# Repository Instructions for Codex

## Role
You are helping me rebuild this repository into a professional AI/Computer Vision portfolio project for GitHub.

## Main Goal
Improve repository quality, reproducibility, readability, and presentation for an AI Engineer / Computer Vision Intern portfolio.

## Project Direction
Prioritize:
- Clean and reproducible code
- Clear README
- Proper project structure
- Minimal runnable examples
- Model results and visual outputs
- Good GitHub presentation
- No unnecessary or risky changes to model logic

## Coding Rules
- Do not rewrite the whole project unless necessary.
- Do not change the core model logic unless there is a bug or reproducibility issue.
- Prefer small, safe, reviewable changes.
- Preserve existing notebooks if they are useful, but clean them if needed.
- Refactor repeated code into `src/` only when it improves clarity.
- Add comments only where they improve understanding.
- Do not add fake results, fake metrics, or unsupported claims.

## Files That Should Not Be Committed
Make sure `.gitignore` excludes:
- datasets
- model weights
- checkpoints
- runs/
- wandb/
- cache files
- __pycache__/
- .ipynb_checkpoints/
- large generated outputs
- environment-specific files

## Required Repository Quality
The final repository should contain, when appropriate:
- README.md
- requirements.txt or environment.yml
- .gitignore
- src/
- notebooks/
- configs/
- assets/ or sample_outputs/
- results/ or reports/
- clear usage commands

## README Requirements
The README should include:
1. Project overview
2. Motivation
3. Dataset description
4. Method or model architecture
5. Training/evaluation protocol
6. Results table
7. Demo images or sample outputs
8. How to install
9. How to run training/inference/evaluation
10. Project structure
11. Limitations
12. Future work

## Output Style
After making changes, provide:
- Summary of what was changed
- Files modified/created
- Commands tested
- Remaining issues
- Suggested next steps

## Important
Do not fabricate missing information. If results, metrics, dataset details, or commands are unclear, mark them as TODO instead of inventing them.
