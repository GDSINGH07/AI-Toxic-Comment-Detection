
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

body {
    background:
        radial-gradient(circle at 15% 15%, #12366b 0%, transparent 35%),
        radial-gradient(circle at 85% 80%, #30205e 0%, transparent 35%),
        #050b18 !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    background: transparent !important;
    color: #eaf2ff !important;
    font-family: 'Inter', sans-serif !important;
}

.block, .form, .panel, .gr-panel {
    background: rgba(17, 30, 55, 0.58) !important;
    border: 1px solid rgba(120, 170, 255, 0.20) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.22) !important;
}

h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #8ec5ff !important;
    font-weight: 700 !important;
}

p, label, span {
    color: #d6e5ff;
}

input, textarea {
    background: rgba(5, 15, 32, 0.75) !important;
    color: white !important;
    border: 1px solid rgba(96, 165, 250, 0.35) !important;
    border-radius: 12px !important;
}

button {
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
}

button.primary {
    background: linear-gradient(110deg, #2563eb, #06b6d4) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 0 18px rgba(37, 99, 235, 0.35) !important;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 26px rgba(6, 182, 212, 0.5) !important;
}

input[type="range"] {
    accent-color: #38bdf8 !important;
}

footer {
    display: none !important;
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

        # PREMIUM SIDEBAR DASHBOARD

    def navigate(page):
        return (
            gr.update(visible=page == "Home / Live Detection"),
            gr.update(visible=page == "Graphs & Analytics"),
            gr.update(visible=page == "About Project")
        )

    with gr.Row():

        # LEFT SIDEBAR
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("""
            # 🛡️ TOXIC AI
            ### Detection System
            ---
            **NAVIGATION**
            """)

            page_navigation = gr.Radio(
                choices=[
                    "Home / Live Detection",
                    "Graphs & Analytics",
                    "About Project"
                ],
                value="Home / Live Detection",
                label="Select Page"
            )

            gr.Markdown("""
            ---
            <small>AI-Powered Content Safety</small>
            """)

        # MAIN CONTENT AREA
        with gr.Column(scale=5, min_width=500):

            # HOME PAGE
            with gr.Column(visible=True) as home_page:
                gr.Markdown("""
                # 🛡️ AI Toxic Comment Detection
                ### Analyze comments and identify harmful content.
                """)

                comment_input = gr.Textbox(
                    label="Enter Your Comment",
                    placeholder="Type or paste a comment here...",
                    lines=4
                )

                threshold_slider = gr.Slider(
                    minimum=10,
                    maximum=90,
                    value=50,
                    step=5,
                    label="Detection Threshold (%)"
                )

                category_filter = gr.CheckboxGroup(
                    choices=all_category_choices,
                    value=categories,
                    label="Select Toxicity Categories"
                )

                analyze_button = gr.Button(
                    "🔍 Analyze Comment",
                    variant="primary"
                )

                result_output = gr.Markdown(label="Detection Result")

                score_table = gr.Dataframe(
                    label="Category Scores",
                    interactive=False
                )

                score_chart = gr.Plot(label="Detection Score Chart")

                analyze_button.click(
                    fn=analyze_comment,
                    inputs=[
                        comment_input,
                        threshold_slider,
                        category_filter
                    ],
                    outputs=[
                        result_output,
                        score_table,
                        score_chart
                    ]
                )

            # ANALYTICS PAGE
            with gr.Column(visible=False) as analytics_page:
                gr.Markdown("""
                # 📊 Graphs & Analytics
                ### Explore dataset distribution and model performance.
                """)

                gr.Markdown(
                    f"**Micro F1 Score:** {micro_f1:.2f}%"
                    f" &nbsp;&nbsp; | &nbsp;&nbsp; "
                    f"**Macro F1 Score:** {macro_f1:.2f}%"
                )

                analytics_filter = gr.CheckboxGroup(
                    choices=all_category_choices,
                    value=categories,
                    label="Filter Categories"
                )

                refresh_analytics = gr.Button(
                    "Update Analytics",
                    variant="primary"
                )

                distribution_chart = gr.Plot(
                    label="Dataset Distribution"
                )

                performance_chart = gr.Plot(
                    label="Model Performance"
                )

                analytics_table = gr.Dataframe(
                    label="Analytics Data",
                    interactive=False
                )

                refresh_analytics.click(
                    fn=create_analytics,
                    inputs=analytics_filter,
                    outputs=[
                        distribution_chart,
                        performance_chart,
                        analytics_table
                    ]
                )

            # ABOUT PAGE
            with gr.Column(visible=False) as about_page:
                gr.Markdown("""
                # ℹ️ About This Project

                **AI Toxic Comment Detection System**

                This project uses Machine Learning and Natural Language
                Processing to identify harmful comments.

                ### Detection Categories
                - Toxic
                - Severe Toxic
                - Obscene
                - Threat
                - Insult
                - Identity Hate

                ### Technologies Used
                - Python
                - Scikit-learn
                - TF-IDF
                - Logistic Regression
                - Gradio

                ### Project Purpose
                To help identify harmful online comments and support
                safer digital communication.
                """)

    page_navigation.change(
        fn=navigate,
        inputs=page_navigation,
        outputs=[
            home_page,
            analytics_page,
            about_page
        ]
    )
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
