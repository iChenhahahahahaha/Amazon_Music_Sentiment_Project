"""A small Streamlit demo for the Amazon Music review sentiment project."""

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
import streamlit as st


# Keeping the main paths here makes the rest of the app easier to read.
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "best_sentiment_system.joblib"
DATABASE_PATH = PROJECT_DIR / "demo_predictions.db"

# These are simple demo thresholds, not production-optimized values.
LOW_CONFIDENCE_THRESHOLD = 0.60
# This is the best Negative-F1 threshold selected in the notebook.
POSSIBLE_NEGATIVE_THRESHOLD = 0.45


st.set_page_config(
    page_title="Amazon Review Sentiment Demo",
    page_icon="🎸",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the saved TF-IDF and Logistic Regression pipeline once."""
    artifact = joblib.load(MODEL_PATH)
    return artifact["system"]


def predict_review(review_text):
    """Predict one review and decide whether a person should check it."""
    model = load_model()
    probabilities = model.predict_proba([review_text])[0]

    # I use model.classes_ so the probability labels never get mixed up.
    probability_by_class = {
        label: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    predicted_label = max(
        probability_by_class,
        key=probability_by_class.get,
    )
    confidence = probability_by_class[predicted_label]
    negative_probability = probability_by_class["Negative"]

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        review_status = "Needs Review: Low Confidence"
    elif negative_probability >= POSSIBLE_NEGATIVE_THRESHOLD:
        review_status = "Needs Review: Possible Negative"
    else:
        review_status = "Auto Accepted"

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "negative_probability": negative_probability,
        "review_status": review_status,
        "probability_by_class": probability_by_class,
    }


def initialize_database():
    """Create the small local demo database if it does not exist."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                negative_probability REAL NOT NULL,
                review_status TEXT NOT NULL,
                corrected_label TEXT,
                reviewed INTEGER DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                reviewed_at TIMESTAMP
            )
            """
        )
        connection.commit()


def save_prediction(review_text, prediction, created_at=None):
    """Save one prediction so it can appear in the demo dashboard."""
    created_at = created_at or datetime.now()

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                review_text,
                predicted_label,
                confidence,
                negative_probability,
                review_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                review_text,
                prediction["predicted_label"],
                prediction["confidence"],
                prediction["negative_probability"],
                prediction["review_status"],
                created_at.isoformat(),
            ),
        )
        connection.commit()
        return cursor.lastrowid


def load_prediction_logs():
    """Read all demo predictions into a dataframe."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY created_at",
            connection,
            parse_dates=["created_at", "reviewed_at"],
        )


def load_review_queue():
    """Read predictions that still need a human decision."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM predictions
            WHERE review_status LIKE 'Needs Review:%'
              AND reviewed = 0
            ORDER BY negative_probability DESC, created_at
            """,
            connection,
        )


def save_human_feedback(review_id, corrected_label):
    """Save the label selected by the demo reviewer."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            UPDATE predictions
            SET corrected_label = ?,
                reviewed = 1,
                reviewed_at = ?
            WHERE id = ?
            """,
            (
                corrected_label,
                datetime.now().isoformat(),
                int(review_id),
            ),
        )
        connection.commit()


def seed_demo_data():
    """Add a few made-up logs when the demo database is empty."""
    demo_reviews = [
        "Great sound and very easy to use.",
        "The strings feel nice and stay in tune.",
        "It works, but the build quality feels average.",
        "The product is fine, nothing really special.",
        "The cable stopped working during the first week.",
        "Really disappointed with the sound quality.",
        "Good value for the price.",
        "I like the tone, but the case feels cheap.",
        "The tuner is accurate and simple.",
        "Not bad, but I would probably buy another brand.",
        "Excellent quality and fast setup.",
        "The stand feels unstable and unsafe.",
    ]

    with sqlite3.connect(DATABASE_PATH) as connection:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

    if row_count > 0:
        return

    start_time = datetime.now().replace(
        hour=10,
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(days=5)

    for index, review_text in enumerate(demo_reviews):
        prediction = predict_review(review_text)
        save_prediction(
            review_text,
            prediction,
            created_at=start_time + timedelta(hours=index * 10),
        )


def reset_demo():
    """Delete the temporary demo database and create fresh sample rows."""
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
    initialize_database()
    seed_demo_data()


# The app creates its sample data automatically on the first run.
initialize_database()
seed_demo_data()


st.title("🎸 Amazon Musical Instruments Sentiment Demo")
st.caption(
    "A demo using TF-IDF + Logistic Regression"
)

st.info(
    "This app uses simulated monitoring data. The local database may reset "
    "when the free Streamlit app restarts, which is expected for this demo."
)

with st.sidebar:
    st.header("About this demo")
    st.write(
        "Low-confidence and possibly negative predictions are sent to a "
        "small human review queue."
    )
    st.write(
        f"Low-confidence threshold: {LOW_CONFIDENCE_THRESHOLD:.0%}"
    )
    st.write(
        "Possible-negative threshold: "
        f"{POSSIBLE_NEGATIVE_THRESHOLD:.0%}"
    )

    if st.button("Reset demo data"):
        reset_demo()
        st.success("The demo data was reset.")
        st.rerun()


predict_tab, monitoring_tab, review_tab = st.tabs(
    ["Predict", "Monitoring", "Human Review"]
)


with predict_tab:
    st.subheader("Try a new review")
    review_text = st.text_area(
        "Enter a musical instrument review:",
        placeholder="Example: The sound is great, but the stand feels cheap.",
        height=130,
    )

    if st.button("Predict sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            prediction = predict_review(review_text.strip())
            prediction_id = save_prediction(
                review_text.strip(),
                prediction,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Prediction",
                prediction["predicted_label"],
            )
            col2.metric(
                "Confidence",
                f"{prediction['confidence']:.1%}",
            )
            col3.metric(
                "Negative probability",
                f"{prediction['negative_probability']:.1%}",
            )

            if prediction["review_status"] == "Auto Accepted":
                st.success(prediction["review_status"])
            else:
                st.warning(prediction["review_status"])

            probability_table = pd.DataFrame(
                {
                    "sentiment": prediction[
                        "probability_by_class"
                    ].keys(),
                    "probability": prediction[
                        "probability_by_class"
                    ].values(),
                }
            ).set_index("sentiment")

            st.bar_chart(probability_table)
            st.caption(
                f"Saved as demo prediction #{prediction_id}."
            )


with monitoring_tab:
    st.subheader("Simulated online monitoring")
    prediction_logs = load_prediction_logs()

    total_predictions = len(prediction_logs)
    negative_rate = (
        prediction_logs["predicted_label"]
        .eq("Negative")
        .mean()
    )
    low_confidence_rate = (
        prediction_logs["confidence"]
        .lt(LOW_CONFIDENCE_THRESHOLD)
        .mean()
    )
    waiting_count = (
        prediction_logs["review_status"]
        .str.startswith("Needs Review")
        & prediction_logs["reviewed"].eq(0)
    ).sum()

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total predictions", total_predictions)
    metric2.metric("Predicted Negative", f"{negative_rate:.1%}")
    metric3.metric(
        "Low confidence",
        f"{low_confidence_rate:.1%}",
    )
    metric4.metric("Waiting for review", int(waiting_count))

    chart1, chart2 = st.columns(2)

    with chart1:
        st.write("Predicted sentiment mix")
        label_counts = (
            prediction_logs["predicted_label"]
            .value_counts()
            .reindex(
                ["Negative", "Neutral", "Positive"],
                fill_value=0,
            )
        )
        st.bar_chart(label_counts)

    with chart2:
        st.write("Prediction volume by day")
        prediction_logs["date"] = (
            prediction_logs["created_at"].dt.date
        )
        daily_volume = prediction_logs.groupby("date").size()
        st.line_chart(daily_volume)

    reviewed_logs = prediction_logs[
        prediction_logs["reviewed"].eq(1)
    ].dropna(subset=["corrected_label"])

    if reviewed_logs.empty:
        st.caption(
            "No human feedback yet. Review a queued prediction "
            "to create this metric."
        )
    else:
        correction_rate = (
            reviewed_logs["predicted_label"]
            .ne(reviewed_logs["corrected_label"])
            .mean()
        )
        st.metric(
            "Human correction rate",
            f"{correction_rate:.1%}",
        )

    with st.expander("Show prediction log"):
        st.dataframe(
            prediction_logs[
                [
                    "id",
                    "created_at",
                    "review_text",
                    "predicted_label",
                    "confidence",
                    "review_status",
                    "corrected_label",
                ]
            ].sort_values("created_at", ascending=False),
            width="stretch",
            hide_index=True,
        )


with review_tab:
    st.subheader("Human review queue")
    review_queue = load_review_queue()

    if review_queue.empty:
        st.success("Nothing is waiting for review.")
    else:
        st.write(
            f"{len(review_queue)} prediction(s) are waiting for a person."
        )

        selected_id = st.selectbox(
            "Choose a prediction:",
            review_queue["id"].tolist(),
            format_func=lambda review_id: (
                f"Review #{review_id}"
            ),
        )

        selected_row = review_queue[
            review_queue["id"].eq(selected_id)
        ].iloc[0]

        st.write("**Customer review**")
        st.write(selected_row["review_text"])

        info1, info2, info3 = st.columns(3)
        info1.metric(
            "Model prediction",
            selected_row["predicted_label"],
        )
        info2.metric(
            "Confidence",
            f"{selected_row['confidence']:.1%}",
        )
        info3.metric(
            "Negative probability",
            f"{selected_row['negative_probability']:.1%}",
        )

        labels = ["Negative", "Neutral", "Positive"]
        default_index = labels.index(
            selected_row["predicted_label"]
        )
        corrected_label = st.selectbox(
            "Human-reviewed label:",
            labels,
            index=default_index,
        )

        if st.button("Save human feedback", type="primary"):
            save_human_feedback(
                selected_id,
                corrected_label,
            )
            st.success("Feedback saved.")
            st.rerun()
