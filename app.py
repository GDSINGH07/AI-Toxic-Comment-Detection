
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

    pie_fig, pie_ax = plt.subplots(figsize=(5, 5))

pie_ax.pie(
    scores,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

pie_ax.set_title("Toxicity Category Scores")
pie_ax.axis("equal")

return result, score_df, fig, pie_fig


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
    max-width: 1500px !important;
    margin: auto !important;
    background: transparent !important;
    color: #eaf2ff !important;
    font-family: 'Inter', sans-serif !important;
}

/* SIDEBAR */
#sidebar {
    background: rgba(17, 30, 55, 0.92) !important;
    border: 1px solid rgba(120, 170, 255, 0.25) !important;
    border-radius: 20px !important;
    padding: 22px !important;
    min-width: 250px !important;
    height: fit-content !important;
    box-shadow: 0 10px 35px rgba(0,0,0,0.3) !important;
}

/* Remove unnecessary boxes inside sidebar */
#sidebar .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}

/* VERTICAL NAVIGATION */
#page-navigation fieldset {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 12px !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

#page-navigation .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#page-navigation label {
    display: flex !important;
    width: 100% !important;
    box-sizing: border-box !important;
    padding: 13px !important;
    background: rgba(50, 75, 115, 0.45) !important;
    border: 1px solid rgba(120, 170, 255, 0.2) !important;
    border-radius: 12px !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
}

/* MAIN CONTENT */
.gradio-container h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.gradio-container h2,
.gradio-container h3 {
    color: #8ec5ff !important;
    font-weight: 700 !important;
}

.gradio-container p,
.gradio-container label {
    color: #d6e5ff;
}

/* INPUTS */
input, textarea {
    background: rgba(5, 15, 32, 0.85) !important;
    color: white !important;
    border: 1px solid rgba(96, 165, 250, 0.35) !important;
    border-radius: 12px !important;
    max-width: 100% !important;
}

button {
    border-radius: 12px !important;
    white-space: normal !important;
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
}

input[type="range"] {
    accent-color: #38bdf8 !important;
}

/* CHARTS AND TABLES */
.gradio-container img,
.gradio-container canvas,
.gradio-container table {
    max-width: 100% !important;
}

footer {
    display: none !important;
}

/* SMALL SCREENS */
@media (max-width: 768px) {
    #sidebar {
        min-width: 0 !important;
        width: 100% !important;
    }

    .gradio-container {
        padding: 10px !important;
    }
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
         with gr.Column(scale=1, min_width=250, elem_id="sidebar"):
                gr.Image(
                    value="Gemini_Generated_Image_rvtbb0rvtbb0rvtbu-removebg-preview.png",
                    show_label=False,
                    container=False,
                    interactive=False,
                    height=180
                )

                gr.Markdown("""
                ### Detection System
                """)

                page_navigation = gr.Radio(
                    elem_id="page-navigation",
                    choices=[
                        "Home / Live Detection",
                        "Graphs & Analytics",
                        "About Project"
                    ],
                    value="Home / Live Detection",
                    label="NAVIGATION"
                )

                
        # MAIN CONTENT AREA
         with gr.Column(scale=4, min_width=0):

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
                pie_chart = gr.Plot(label="Toxicity Category Pie Chart")
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
                        score_chart,
                        pie_chart
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
