import gradio as gr
from app.resume_parser import extract_text_from_pdf
from app.matcher import JobMatcher
from app.utils import clean_text, highlight_matches, highlight_resume_text

# Initialize the matcher (default: hybrid mode using BERT and TF-IDF)
matcher = JobMatcher("data/job.json", method="hybrid")

def process_resume(file, method_label, alpha):
    """
    Process the uploaded resume and return top job matches + highlighted resume.
    """
    # Map UI label to internal method
    method_map = {
        "TF-IDF (Basic)": "tfidf",
        "Semantic (BERT)": "berth",
        "Hybrid (TF-IDF + BERT)": "hybrid"
    }
    method = method_map.get(method_label, "hybrid")

    # Update matcher settings
    matcher.method = method
    matcher.alpha = alpha if method == "hybrid" else 0.5  # Default alpha if not hybrid

    # Extract and clean resume text
    try:
        text = extract_text_from_pdf(file.name)
    except Exception as e:
        return f"<p><strong>Error reading PDF:</strong> {str(e)}</p>", ""

    cleaned = clean_text(text)

    # Perform matching
    matched_jobs = matcher.match(cleaned, top_k=5)

    if matched_jobs.empty or matched_jobs['match_score'].max() < 0.1:
        return "<p><strong>No strong matches found.</strong></p>", ""

    # Prepare tokens from job descriptions for highlighting resume
    job_tokens = set()
    for _, row in matched_jobs.iterrows():
        job_text = ' '.join(row['description'] + row['requirements'])
        job_tokens.update(clean_text(job_text).split())

    resume_tokens = set(cleaned.split())
    highlighted_resume = highlight_resume_text(text, job_tokens)

    # Build HTML output for matched jobs
    result = ""
    for _, row in matched_jobs.iterrows():
        result += f"<h1>📌 {row['title']}</h1>"
        result += f"<h2>Match Score: {row.get('match_score', 0):.2f}</h2><br>"
        result += f"<strong>Company:</strong> {row['company']}<br>"
        result += f"<strong>Location:</strong> {row.get('location', 'N/A')}<br>"
        result += f"<strong>Type:</strong> {row.get('type', 'N/A')}<br>"
        result += f"<strong>Deadline:</strong> {row.get('deadline', 'N/A')}<br><br>"

        result += "<strong>🔧 Description:</strong><ul>"
        for item in row['description']:
            result += f"<li>{highlight_matches(item, resume_tokens)}</li>"
        result += "</ul>"

        result += "<strong>✅ Requirements:</strong><ul>"
        for item in row['requirements']:
            result += f"<li>{highlight_matches(item, resume_tokens)}</li>"
        result += "</ul>"

        result += f"<strong>📞 Phone:</strong> {row.get('phone', 'N/A')}<br>"
        result += f"<strong>🔗 <a href='{row['url']}' target='_blank'>More Info</a></strong><br>"
        result += "<hr><br>"

    return result, f"<h1>📄 Highlighted Resume</h1><div style='white-space: pre-wrap;'>{highlighted_resume}</div>"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎯 Text Similarity for Resume Job Matcher (UM Student Internship)")
    gr.Markdown("Upload your resume to find the best-matching internships based on TF-IDF and semantic similarity (BERT).")

    with gr.Row():
        with gr.Column(scale=1):
            resume_file = gr.File(label="📄 Upload Resume (PDF)")

            method = gr.Radio(
                ["TF-IDF (Basic)", "Semantic (BERT)", "Hybrid (TF-IDF + BERT)"],
                label="Matching Method",
                value="Hybrid (TF-IDF + BERT)"
            )

            alpha = gr.Slider(
                0.0, 1.0, value=0.5, step=0.1,
                label="TF-IDF vs BERT Weight (0=BERT, 1=TF-IDF)",
                visible=True
            )

            with gr.Row():
                clear_btn = gr.ClearButton()
                submit_btn = gr.Button("Submit")

            highlighted_resume = gr.HTML(label="🧠 Highlighted Resume")

        with gr.Column(scale=2):
            job_matches = gr.HTML(label="🔍 Top Matching Jobs")

    def toggle_slider(method_label):
        return gr.update(visible=(method_label == "Hybrid (TF-IDF + BERT)"))

    method.change(
        fn=toggle_slider,
        inputs=method,
        outputs=alpha
    )

    def wrapped_process_resume(file, method_label, alpha):
        result, highlighted = process_resume(file, method_label, alpha)
        return highlighted, result

    submit_btn.click(
        wrapped_process_resume,
        inputs=[resume_file, method, alpha],
        outputs=[highlighted_resume, job_matches]
    )

    clear_btn.add([resume_file, method, alpha, highlighted_resume, job_matches])

if __name__ == "__main__":
    demo.launch(theme="default")
