# Amazon Musical Instruments Review Sentiment Analysis

This is an end-to-end machine-learning project based on real Amazon musical-instrument reviews from [Kaggle](https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews/data).

The project starts with text cleaning and exploratory analysis, compares several classical machine-learning models, and then uses the best TF-IDF + Logistic Regression pipeline in a small deployment simulation.

## What is included?

```text
Amazon_Music_Sentiment_Project/
├── Amazon_Reviews_Sentiment_Analysis.ipynb
├── app.py
├── api.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── best_sentiment_system.joblib
```

- `Amazon_Reviews_Sentiment_Analysis.ipynb` contains the full analysis, model comparison, Colab-friendly monitoring simulation, SQLite demo, and human-review workflow.
- `app.py` is the main Streamlit portfolio app.
- `api.py` is an optional FastAPI prediction service.
- `models/best_sentiment_system.joblib` contains the trained TF-IDF + Logistic Regression pipeline.

## Main model result

The selected model is TF-IDF + Logistic Regression.

- Holdout accuracy: about 86.2%
- Holdout macro F1: about 0.605
- Positive F1: about 0.933
- Neutral is the hardest class, with an F1 around 0.38

The dataset is strongly imbalanced, so balanced class weights and macro F1 are used. This helps, but it does not create more real Neutral or Negative training examples.

## Pure monitoring simulation

This repository intentionally uses a small local SQLite database.

The Streamlit app automatically creates fake prediction logs the first time it runs. New predictions and human corrections are added to the same temporary database.

On free Streamlit Community Cloud, the database may reset when the app restarts or redeploys. That is expected for this demo. The app does not promise permanent user-data storage.

This keeps the project:

- free;
- easy to understand;
- easy to reset;
- safe for a public portfolio.

## Run the notebook in Google Colab

1. Upload `Amazon_Reviews_Sentiment_Analysis.ipynb` to Colab.
2. Download `Musical_instruments_reviews.csv` from the Kaggle link.
3. Upload the CSV to the Colab session.
4. Run the notebook from top to bottom.

The notebook checks these common locations:

```text
/content/Musical_instruments_reviews.csv
Musical_instruments_reviews.csv
data/Musical_instruments_reviews.csv
../data/Musical_instruments_reviews.csv
```

The final notebook section creates `demo_monitoring.db`. This database belongs to the temporary Colab runtime and can disappear after the runtime disconnects. That is okay for the pure simulation option.

## Run the Streamlit app locally

First install the packages:

```bash
python -m pip install -r requirements.txt
```

Then start the app:

```bash
streamlit run app.py
```

The app contains three tabs:

1. **Predict** — enter a new review and see probabilities.
2. **Monitoring** — view simulated volume, sentiment mix, and review-queue metrics.
3. **Human Review** — confirm or correct uncertain predictions.

Use the **Reset demo data** button in the sidebar whenever you want a clean demo.

## Deploy the Streamlit app for free

1. Create a new GitHub repository.
2. Upload all files in this project folder, including the `models` folder.
3. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Sign in and connect your GitHub account.
5. Click **Create app**.
6. Select your repository and branch.
7. Set the entrypoint file to:

```text
app.py
```

8. Click **Deploy**.

Streamlit will read `requirements.txt`, install the packages, load the model, and give the app a shareable URL.

No GPU is needed. No paid database is needed for this simulated version.

## Optional: run the FastAPI service

The Streamlit app does not need FastAPI. This file is included to show how the same model could be exposed as a small API.

Start it locally with:

```bash
uvicorn api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The two endpoints are:

```text
GET  /health
POST /predict
```

Example request body:

```json
{
  "review_text": "The guitar sounds great, but the case feels cheap."
}
```

## Human-review rule

This student demo sends a prediction to the queue when:

- the highest class probability is below 60%; or
- the Negative probability is at least 35%.

These are easy-to-understand demo thresholds. They were not optimized for a real customer-support cost function.

## Important limitations

- Sentiment labels come from star ratings, not manual text annotation.
- About 87.9% of the reviews are Positive.
- The data covers musical instruments and may not generalize to other product categories.
- Neutral and Negative performance is much weaker than Positive performance.
- Monitoring logs are simulated.
- The local SQLite database is not permanent cloud storage.
- This is a portfolio project, not a production-ready Amazon service.
