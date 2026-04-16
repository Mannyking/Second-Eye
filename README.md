# Second Eye

Second Eye is a multilabel inventory checker built with PyTorch and Streamlit.  
It detects everyday objects in an uploaded image, then can generate contextual feedback using Gemini.

## What it does

- Runs multilabel image classification with a fine-tuned ResNet-18 artifact.
- Applies per-class thresholds to convert probabilities into final labels.
- Provides contextual feedback presets (for example, school commute or work session).
- Includes a Streamlit reports page for dataset and training run visualization.

## Project structure

```text
second-eye/
├── app.py                        # Main Streamlit app (upload + inference + feedback)
├── model/
│   └── inference.py              # Model loading and prediction logic
├── llm/
│   └── gemini.py                 # Gemini feedback generation
├── prompts/
│   └── presets.py                # Prompt presets and expected item sets
├── pages/
│   └── 2_Reports.py              # Reports dashboard page
└── best_model_with_thresholds.pt # Trained model artifact with class metadata
```

## Requirements

- Python 3.10+ (3.11 recommended)
- `pip`
- A valid `GEMINI_API_KEY` if you want AI feedback enabled

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (or use `.env`):

```env
GEMINI_API_KEY=your_api_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

## Using the app

- Open the main page to upload an image and run inference.
- Choose a preset to get contextual feedback on detected items.
- Open the **Reports** page from the Streamlit sidebar to view:
  - dataset profile
  - training run history
  - config impact analysis
  - per-run drilldown (loss, per-class metrics, thresholds)

## Notes

- Inference is configured for CPU by default in this repo.
- If `GEMINI_API_KEY` is missing, detection still works; AI feedback is skipped with a warning message.
