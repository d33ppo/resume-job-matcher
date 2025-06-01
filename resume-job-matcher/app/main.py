import gradio as gr
from app.resume_parser import extract_text_from_pdf
from app.matcher import JobMatcher
from app.utils import clean_text, highlight_matches, highlight_resume_text

# Initialize the matcher (default: hybrid mode using BERT and TF-IDF)
matcher = JobMatcher("data/job.json", use_bert=True, alpha=0.5)

def process_resume(file, method, alpha):
    """
    Process the uploaded resume and return top job matches + highlighted resume.
    - file: uploaded resume PDF
    - method: selected matching method (TF-IDF, BERT, or Hybrid)
    - alpha: weighting between TF-IDF and BERT (0=BERT only, 1=TF-IDF only)
    """
    # Update matcher settings based on user input
    matcher.use_bert = (method != "TF-IDF (Basic)")
    matcher.alpha = alpha

    # Extract and clean resume text
    try:
        text = extract_text_from_pdf(file.name)
    except Exception as e:
        return "<p><strong>Error reading PDF:</strong> {}</p>".format(str(e)), "" # Return error message if PDF extraction fails

    cleaned = clean_text(text)

    # Perform matching
    matched_jobs = matcher.match(cleaned, top_k=5)

    if matched_jobs.empty or matched_jobs['match_score'].max() < 0.1:
        return "<p><strong>No strong matches found.</strong></p>", highlighted_resume # Return empty results if no matches found

    # Prepare tokens from job descriptions for highlighting resume
    job_tokens = set()
    for _, row in matched_jobs.iterrows():
        job_text = ' '.join(row['description'] + row['requirements'])
        job_tokens.update(clean_text(job_text).split())

    # Prepare tokens from resume for highlighting job descriptions
    resume_tokens = set(cleaned.split())

    # Highlight matched tokens in the resume
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

    # Return job results and highlighted resume
    return result, f"<h1>📄 Highlighted Resume</h1><div style='white-space: pre-wrap;'>{highlighted_resume}</div>"

# Gradio UI with Blocks layout
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
                label="TF-IDF vs BERT Weight (0=BERT, 1=TF-IDF)"
            )

            with gr.Row():
                clear_btn = gr.ClearButton()
                submit_btn = gr.Button("Submit")

            highlighted_resume = gr.HTML(label="🧠 Highlighted Resume")

        with gr.Column(scale=2):
            job_matches = gr.HTML(label="🔍 Top Matching Jobs")

    # Define submit behavior
    def wrapped_process_resume(file, method, alpha):
        result, highlighted = process_resume(file, method, alpha)
        return highlighted, result

    submit_btn.click(
        wrapped_process_resume,
        inputs=[resume_file, method, alpha],
        outputs=[highlighted_resume, job_matches]
    )

    # Clear all fields
    clear_btn.add([resume_file, method, alpha, highlighted_resume, job_matches])

# Launch app
if __name__ == "__main__":
    demo.launch(theme="default")
