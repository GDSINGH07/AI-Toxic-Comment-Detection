
import gradio as gr
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import html
import os

# ==========================================
# 1. LOAD TRAINED MODEL AND DATA
# ==========================================

MODEL_PATH = "models/toxic_comment_model.pkl"

bundle = joblib.load(MODEL_PATH)

tfidf = bundle["tfidf"]
model = bundle["model"]
categories = bundle["categories"]
threshold = bundle.get("threshold", 0.5)

with open("models/category_counts.json", "r") as f:
    category_counts = json.load(f)

with open("models/classification_report.json", "r") as f:
    report = json.load(f)


# ==========================================
# 2. LIVE TOXIC COMMENT DETECTION
# ==========================================

def analyze_comment(comment):

    if not comment or not comment.strip():
        return "Please enter some text first.", pd.DataFrame(), None

    # Transform input text
    text_vector = tfidf.transform([comment])

    # Predict probabilities
    probabilities = model.predict_proba(text_vector)[0]

    # Convert probabilities into percentages
    scores = probabilities * 100

    # Apply threshold to each category
    predictions = probabilities >= threshold

    detected = [
        categories[i]
        for i in range(len(categories))
        if predictions[i]
    ]

    # Prepare result
    if len(detected) == 0:
        result = """
## GOOD / NON-TOXIC

No toxicity category was detected above the configured threshold.
"""
    else:
        result = """
## TOXIC COMMENT DETECTED

**Detected categories:**

"""
        for category in detected:
            result += f"- {category.replace('_', ' ').title()}\n"

    result += f"\n**Detection threshold:** {threshold * 100:.0f}%"

    # Prepare score table
    score_df = pd.DataFrame({
        "Category": [
            category.replace("_", " ").title()
            for category in categories
        ],
        "Model Score (%)": [round(float(s), 2) for s in scores],
        "Detected": [
            "Yes" if predictions[i] else "No"
            for i in range(len(categories))
        ]
    })

    # Plot individual category scores
    fig, ax = plt.subplots(figsize=(9, 4))

    colors = [
        "tomato" if predictions[i] else "steelblue"
        for i in range(len(categories))
    ]

    labels = [
        category.replace("_", " ").title()
        for category in categories
    ]

    ax.barh(labels, scores, color=colors)
    ax.axvline(
        threshold * 100,
        linestyle="--",
        color="black",
        label="Detection threshold"
    )

    ax.set_xlabel("Model score (%)")
    ax.set_title("Toxicity Category Scores")
    ax.set_xlim(0, 100)
    ax.legend()
    plt.tight_layout()

    return result, score_df, fig


# ==========================================
# 3. DATASET DISTRIBUTION GRAPH
# ==========================================

def create_distribution_chart():

    labels = [
        category.replace("_", " ").title()
        for category in categories
    ]

    values = [
        category_counts.get(category, 0)
        for category in categories
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(labels, values)

    ax.set_title("Toxic Comment Dataset Distribution")
    ax.set_ylabel("Number of labelled comments")
    ax.set_xlabel("Toxicity Categories")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return fig


# ==========================================
# 4. MODEL PERFORMANCE GRAPH
# ==========================================

def create_performance_chart():

    labels = []
    f1_values = []

    for category in categories:

        if category in report:
            labels.append(category.replace("_", " ").title())
            f1_values.append(
                report[category].get("f1-score", 0) * 100
            )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(labels, f1_values)

    ax.set_title("Model F1 Score by Toxicity Category")
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(0, 100)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return fig


# ==========================================
# 5. OVERALL MODEL METRICS
# ==========================================

overall = report.get("micro avg", {})
macro = report.get("macro avg", {})

micro_f1 = overall.get("f1-score", 0) * 100
macro_f1 = macro.get("f1-score", 0) * 100


# ==========================================
# 6. WEBSITE INTERFACE
# ==========================================

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    css=custom_css,
    title="AI Toxic Comment Detection"
) as demo:

    gr.Markdown("""
    # 🛡️ AI TOXIC COMMENT DETECTION SYSTEM

    ### Machine Learning Based Text Classification

    Detect harmful language and identify six toxicity categories
    using an AI-powered machine learning model.
    """)

    with gr.Tabs():

        # ----------------------------------
        # HOME AND LIVE DETECTION
        # ----------------------------------

        with gr.Tab("🏠 Home / Live Detection"):

            gr.Markdown("""
            ## Test Your Comment

            Enter a word, sentence, or paragraph below.
            The AI system will analyze it and display the
            detected categories and model scores.
            """)

            comment_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type your comment here...",
                lines=5
            )

            with gr.Row():
                analyze_button = gr.Button(
                    "🔍 Analyze Comment",
                    variant="primary"
                )

                clear_button = gr.ClearButton(
                    components=[comment_input]
                )

            result_output = gr.Markdown(
                label="Detection Result"
            )

            score_output = gr.Dataframe(
                label="Category-wise Model Scores",
                interactive=False
            )

            graph_output = gr.Plot(
                label="Toxicity Score Visualization"
            )

            analyze_button.click(
                fn=analyze_comment,
                inputs=comment_input,
                outputs=[
                    result_output,
                    score_output,
                    graph_output
                ]
            )

            gr.Markdown("""
            **Note:** GOOD / NON-TOXIC means that none of the
            six toxicity categories crossed the configured
            detection threshold. It does not guarantee that
            the comment is positive or harmless.
            """)

        # ----------------------------------
        # GRAPHS AND ANALYTICS
        # ----------------------------------

        with gr.Tab("📊 Graphs & Analytics"):

            gr.Markdown("""
            ## Dataset and Model Analytics

            Explore the dataset distribution and the
            performance of the trained classification model.
            """)

            with gr.Row():

                gr.Column()

                gr.Markdown(
                    f"### Micro F1 Score: {micro_f1:.2f}%"
                )

                gr.Markdown(
                    f"### Macro F1 Score: {macro_f1:.2f}%"
                )

            gr.Markdown("### 1. Dataset Category Distribution")

            distribution_plot = gr.Plot(
                value=create_distribution_chart()
            )

            gr.Markdown("### 2. Model F1 Score by Category")

            performance_plot = gr.Plot(
                value=create_performance_chart()
            )

            gr.Markdown("""
            **F1 Score:** A measure that combines precision
            and recall. Higher values generally indicate
            better classification performance.
            """)

        # ----------------------------------
        # ABOUT PROJECT
        # ----------------------------------

        with gr.Tab("ℹ️ About Project"):

            gr.Markdown("""
            ## About the Project

            **AI Toxic Comment Detection System** is a
            machine learning project designed to identify
            potentially harmful comments in text.

            ### Six Detection Categories

            1. Toxic
            2. Severe Toxic
            3. Obscene
            4. Threat
            5. Insult
            6. Identity Hate

            ### Technology Used

            - Python
            - Scikit-learn
            - TF-IDF Vectorization
            - One-vs-Rest Logistic Regression
            - Gradio
            - Matplotlib
            - Pandas

            ### Working Method

            1. User enters a comment.
            2. TF-IDF converts text into numerical features.
            3. The trained model analyzes the text.
            4. Six category scores are calculated.
            5. Categories crossing the threshold are displayed.

            ### Dataset

            Jigsaw Toxic Comment Classification Dataset.

            ### Project Purpose

            To help identify potentially harmful online
            language and encourage safer digital communication.

            **Disclaimer:** This is an automated classification
            system. Predictions may be incorrect and should
            not be treated as a final judgment about a person
            or their intentions.
            """)

    gr.Markdown("""
    ---
    <center>AI Toxic Comment Detection System | Machine Learning Project</center>
    """)


# ==========================================
# 7. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
