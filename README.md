# Egyptian Arabic Sentiment Analysis

Real-time sentiment analysis for Egyptian Arabic dialect customer reviews, powered by fine-tuned **MARBERTv2**.

---

##  Overview

This project fine-tunes [UBC-NLP/MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) — a BERT model pre-trained on Arabic dialects — to classify Egyptian customer reviews as **positive** or **negative**. A Gradio web interface is included for real-time inference.

---

## Project Structure

```
├── Arabic_Egypt_Customer_Reviews.ipynb   # Training & inference notebook
├── Final_Data.csv                        # Dataset (review_description, rating, company)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Model Weights

Download `marbert_sentiment.pt` from the link below and place it in the same folder as `app.py`:

🔗 [Download marbert_sentiment.pt](https://drive.google.com/file/d/1E_ruU5VVAJllhIlHijDiTj_XOis6AEgi/view?usp=sharing)

### 3. Run the App

```bash
python app.py
```

---

## Dataset

The dataset (`Final_Data.csv`) contains Egyptian Arabic customer reviews with the following columns:

| Column               | Description                        |
|----------------------|------------------------------------|
| `review_description` | Raw review text (Egyptian Arabic)  |
| `rating`             | Sentiment label (`positive` / `negative`) |
| `company`            | Company name (dropped during preprocessing) |

Duplicates and null values are removed before training.

---

## Text Preprocessing

Applied before tokenization:

- Remove HTML tags
- Remove URLs
- Remove mentions (`@user`) and hashtags (`#tag`)
- Normalize repeated characters (e.g., `ههههه` → `ههه`)
- Strip extra whitespace

---

## Model

| Detail         | Value                        |
|----------------|------------------------------|
| Base model     | `UBC-NLP/MARBERTv2`          |
| Task           | Binary sequence classification |
| Labels         | `positive (1)` / `negative (0)` |
| Max token length | 128                         |
| Epochs         | 5                            |
| Learning rate  | 3e-5                         |
| Train batch size | 16                          |
| Eval batch size  | 32                          |
| Best model metric | F1 (weighted)             |
| Mixed precision | FP16 (GPU only)             |

Data split: **64% train / 16% validation / 20% test**

---

## Run the App

After training, run the Gradio interface directly:

```bash
python app.py
```

> Make sure `marbert_sentiment.pt` and `marbert_tokenizer/` are in the same folder.

---

## Dependencies

| Library        | Purpose                        |
|----------------|--------------------------------|
| `torch`        | Deep learning backend          |
| `transformers` | MARBERTv2 model & tokenizer    |
| `datasets`     | HuggingFace dataset utilities  |
| `gradio`       | Web UI for inference           |
| `pandas`       | Data loading & preprocessing   |
| `scikit-learn` | Metrics & train/test split     |

---

## Evaluation Metrics

- **Accuracy**
- **F1 Score** (weighted) — used as the primary metric for best model selection

---

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab or local Jupyter environment
