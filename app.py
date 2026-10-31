
import gradio as gr
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
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


def category_name(category):
    return category.replace("_", " ").title()


all_category_choices = [
    (category_name(category), category)
    for category in categories
]


# ==========================================
# 2. LIVE COMMENT DETECTION
# ==========================================

def analyze_comment(comment, threshold_value, selected_categories):

    if not comment or not comment.strip():
        return "Please enter some text first.", pd.DataFrame(), None

    if not selected_categories:
        return (
            "Please select at least one toxicity category.",
            pd.DataFrame(),
            None
        )

    text_vector = tfidf.transform([comment])
    probabilities = model.predict_proba(text_vector)[0]

    selected_indices = [
        i for i, category in enumerate(categories)
        if category in selected_categories
    ]

    detected = [
        categories[i]
        for i in selected_indices
        if probabilities[i] >= threshold_value / 100
    ]

    if detected:
        result = "## TOXIC CATEGORY DETECTED\n\n"
        result += "**Detected categories:**\n\n"

        for category in detected:
            result += f"- {category_name(category)}\n"
    else:
        result = """
## GOOD / NON-TOXIC

No selected toxicity category crossed the chosen threshold.
"""

    result += f"\n**Detection threshold:** {threshold_value:.0f}%"
    result += f"\n**Selected categories:** {len(selected_categories)}"

    score_df = pd.DataFrame({
        "Category": [
            category_name(categories[i])
            for i in selected_indices
        ],
        "Model Score (%)": [
            round(float(probabilities[i] * 100), 2)
            for i in selected_indices
        ],
        "Detected": [
            "Yes" if probabilities[i] >= threshold_value / 100 else "No"
            for i in selected_indices
        ]
    })

    fig, ax = plt.subplots(figsize=(9, 4))

    scores = [
        probabilities[i] * 100
        for i in selected_indices
    ]

    labels = [
        category_name(categories[i])
        for i in selected_indices
    ]

    colors = [
        "tomato" if probabilities[i] >= threshold_value / 100
        else "steelblue"
        for i in selected_indices
    ]

    ax.barh(labels, scores, color=colors)
    ax.axvline(
        threshold_value,
        linestyle="--",
        color="black",
        label="Detection threshold"
    )

    ax.set_xlabel("Model score (%)")
    ax.set_title("Selected Toxicity Category Scores")
    ax.set_xlim(0, 100)
    ax.legend()
    plt.tight_layout()

    return result, score_df, fig


# ==========================================
# 3. INTERACTIVE ANALYTICS
# ==========================================

def create_analytics(selected_categories):

    if not selected_categories:
        return None, None, pd.DataFrame()

    selected = [
        category for category in categories
        if category in selected_categories
    ]

    labels = [category_name(c) for c in selected]

    counts = [
        category_counts.get(c, 0)
        for c in selected
    ]

    f1_values = [
        report.get(c, {}).get("f1-score", 0) * 100
        for c in selected
    ]

    # Dataset distribution chart
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(labels, counts)
    ax1.set_title("Dataset Distribution - Selected Categories")
    ax1.set_ylabel("Number of labelled comments")
    ax1.set_xlabel("Toxicity Categories")
    ax1.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    # Model F1 score chart
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(labels, f1_values)
    ax2.set_title("Model F1 Score - Selected Categories")
    ax2.set_ylabel("F1 Score (%)")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    # Updated values table
    values_df = pd.DataFrame({
        "Category": labels,
        "Dataset Label Count": counts,
        "F1 Score (%)": [round(v, 2) for v in f1_values]
    })

    return fig1, fig2, values_df


# ==========================================
# 4. OVERALL MODEL METRICS
# ==========================================

overall = report.get("micro avg", {})
macro = report.get("macro avg", {})

micro_f1 = overall.get("f1-score", 0) * 100
macro_f1 = macro.get("f1-score", 0) * 100


# ==========================================
# 5. WEBSITE INTERFACE
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

    Detect potentially harmful language using an AI-powered
    machine learning model.
    """)

    with gr.Tabs():

        # ----------------------------------
        # HOME / LIVE DETECTION
        # ----------------------------------

        with gr.Tab("🏠 Home / Live Detection"):

            gr.Markdown("""
            ## Test Your Comment

            Enter a word, sentence, or paragraph.
            Adjust the filters to explore the prediction results.
            """)

            comment_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type your comment here...",
                lines=5
            )

            threshold_slider = gr.Slider(
                minimum=10,
                maximum=90,
                value=threshold * 100,
                step=5,
                label="Detection Threshold (%)"
            )

            live_category_filter = gr.CheckboxGroup(
                choices=all_category_choices,
                value=categories,
                label="Select Categories to Analyze"
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
                label="Updated Category-wise Model Scores",
                interactive=False
            )

            graph_output = gr.Plot(
                label="Updated Toxicity Score Visualization"
            )

            live_inputs = [
                comment_input,
                threshold_slider,
                live_category_filter
            ]

            live_outputs = [
                result_output,
                score_output,
                graph_output
            ]

            analyze_button.click(
                fn=analyze_comment,
                inputs=live_inputs,
                outputs=live_outputs
            )

            threshold_slider.change(
                fn=analyze_comment,
                inputs=live_inputs,
                outputs=live_outputs
            )

            live_category_filter.change(
                fn=analyze_comment,
                inputs=live_inputs,
                outputs=live_outputs
            )

            gr.Markdown("""
            **Note:** GOOD / NON-TOXIC means that none of the
            selected categories crossed the chosen threshold.
            It does not guarantee that the comment is positive
            or harmless.

            Model scores are estimates, not guarantees.
            """)

        # ----------------------------------
        # GRAPHS AND ANALYTICS
        # ----------------------------------

        with gr.Tab("📊 Graphs & Analytics"):

            gr.Markdown("""
            ## Interactive Dataset and Model Analytics

            Select or deselect categories below.
            The charts and values will update automatically.
            """)

            gr.Markdown("### Overall Model Performance")

            with gr.Row():
                gr.Markdown(
                    f"### Micro F1 Score: {micro_f1:.2f}%"
                )
                gr.Markdown(
                    f"### Macro F1 Score: {macro_f1:.2f}%"
                )

            analytics_filter = gr.CheckboxGroup(
                choices=all_category_choices,
                value=categories,
                label="Filter Analytics by Toxicity Category"
            )

            analytics_values = gr.Dataframe(
                value=create_analytics(categories)[2],
                label="Selected Category Values",
                interactive=False
            )

            gr.Markdown("### 1. Dataset Category Distribution")

            distribution_plot = gr.Plot(
                value=create_analytics(categories)[0]
            )

            gr.Markdown("### 2. Model F1 Score by Category")

            performance_plot = gr.Plot(
                value=create_analytics(categories)[1]
            )

            analytics_filter.change(
                fn=create_analytics,
                inputs=analytics_filter,
                outputs=[
                    distribution_plot,
                    performance_plot,
                    analytics_values
                ]
            )

            gr.Markdown("""
            **F1 Score:** A measure combining precision and
            recall. Higher values generally indicate better
            classification performance.

            **Note:** Filtering changes which categories are
            displayed. It does not retrain the model or change
            the original dataset and performance measurements.
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
            5. Selected categories crossing the threshold are displayed.

            ### Dataset

            Jigsaw Toxic Comment Classification Dataset.

            ### Project Purpose

            To help identify potentially harmful online
            language and encourage safer digital communication.

            **Disclaimer:** This is an automated classification
            system. Predictions may be incorrect and should not
            be treated as a final judgment about a person
            or their intentions.
            """)

    gr.Markdown("""
    ---
    <center>AI Toxic Comment Detection System | Machine Learning Project</center>
    """)


# ==========================================
# 6. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
