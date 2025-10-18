# SmartRecipeApp (Clean Reset)
Fresh start for DTSC691 project (image classification + Streamlit).

## Structure
- `model/` — training script & saved model
- `Data/Training`, `Data/Test` — image folders (not tracked)
- `streamlit_app.py` — Streamlit UI
- `requirements.txt` — pinned deps for Streamlit Cloud

## Quick Start
1. Put images in `Data/Training` and `Data/Test` (class subfolders).
2. Train: `python model/train_model.py`
3. Run app: `streamlit run streamlit_app.py`
