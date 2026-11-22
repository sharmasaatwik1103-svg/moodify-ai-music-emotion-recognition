# Moodify AI — Music Emotion Recognition & Recommendation

A student-friendly, research-grounded project using the DEAM music emotion dataset.

## What it does
- Uses 1,802 DEAM song-level records with human-annotated valence/arousal targets.
- Uses 520 aggregated openSMILE acoustic descriptors.
- Trains and compares a mean baseline, Random Forest, Extra Trees, and Histogram Gradient Boosting.
- Predicts continuous valence and arousal values (DEAM's 1–9 annotation scale).
- Maps predictions into four easy-to-understand mood regions.
- Explores the dataset, visualizes emotion space, recommends emotionally similar song IDs, and stores local history in SQLite.

## Important limitations (please read)
- The CSV is a derived feature table from DEAM, not the original audio files.
- The app's prediction page accepts a CSV with the same 520-feature schema. It does NOT claim to infer the trained model's output from an arbitrary MP3.
- Recommendations are based on DEAM song IDs and emotion similarity. The feature table does not contain a licensed playable music catalogue.
- The four mood names are an interpretation created by this project using thresholds at 5 on the 1–9 scales. DEAM supplies continuous valence/arousal labels, not these four categories.
- DEAM use is subject to its non-commercial terms. Read the official manual before redistribution or commercial use.

## Setup (Windows / VS Code)
1. Extract the ZIP.
2. Open the `Moodify_AI_Final` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate an environment (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
5. Install packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Run the app:
   ```powershell
   python -m streamlit run app.py
   ```
7. Open the local URL shown in the terminal, usually http://localhost:8501.
8. On Overview, click **Download dataset and train models**. Internet is required for the first download.

The first download fetches the 18.7 MB CSV from the GitHub release and verifies its SHA-256 checksum before use.

## Project structure
- `app.py` — Streamlit user interface
- `data_loader.py` — dataset download, checksum validation and loading
- `train_model.py` — preprocessing, model training and evaluation
- `mood_logic.py` — valence/arousal to mood interpretation
- `history.py` — local SQLite history
- `config.py` — paths and dataset configuration
- `data/` — downloaded dataset is stored here on first run
- `models/` — trained model bundle is stored here on first run

## Dataset and citation
Dataset: DEAM song-level acoustic feature table (1,802 songs; 520 openSMILE descriptors; mean valence/arousal and annotator spread).

Derived feature-table release:
https://github.com/asher0913/deam-song-level-features

Original dataset:
https://cvml.unige.ch/databases/DEAM/

Please cite:
Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. PLOS ONE, 12(3), e0173392.

Feature extraction representation:
Eyben et al., “Recent developments in openSMILE,” ACM Multimedia 2013.

## Suggested future work
- Add the original DEAM audio and build a matching openSMILE inference pipeline for MP3/WAV uploads.
- Validate model selection using cross-validation on training data, then report one final untouched test score.
- Add a licensed song catalogue and streaming-service integration where API terms allow.
- Consider MERT/AST embeddings after establishing a reproducible baseline.
