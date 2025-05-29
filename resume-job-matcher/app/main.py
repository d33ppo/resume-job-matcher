import gradio as gr
from app.resume_parser import extract_text_from_pdf
from app.matcher import JobMatcher
from app.utils import clean_text, highlight_matches, highlight_resume_text

# Default matcher (hybrid mode with SBERT and TF-IDF)
matcher = JobMatcher("data/job.json", use_bert=True, alpha=0.5)

def process_resume(file, method, alpha):
    # Update matcher config based on user selection
    matcher.use_bert = (method != "TF-IDF (Basic)")
    matcher.alpha = alpha

    # Extract and clean text
    text = extract_text_from_pdf(file.name)
    cleaned = clean_text(text)

    # Match jobs
    matched_jobs = matcher.match(cleaned, top_k=5)

    # Collect job tokens for highlighting resume
    job_tokens = set()
    for _, row in matched_jobs.iterrows():
        job_text = ' '.join(row['description'] + row['requirements'])
        job_tokens.update(clean_text(job_text).split())

    # Resume token set for highlighting job descriptions
    resume_tokens = set(cleaned.split())

    # Highlight resume based on job tokens
    highlighted_resume = highlight_resume_text(text, job_tokens)

    # Build result HTML
    result = ""
    for _, row in matched_jobs.iterrows():
        result += f"<h2>📌 {row['title']}</h2>"
        result += f"<h3>Match Score: {row.get('match_score', 0):.2f}</h3><br>"
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
demo = gr.Interface(
    fn=process_resume,
    inputs=[
        gr.File(label="📄 Upload Resume (PDF)"),
        gr.Radio(
            ["TF-IDF (Basic)", "Semantic (BERT)", "Hybrid (TF-IDF + BERT)"],
            label="Matching Method",
            value="Hybrid (TF-IDF + BERT)"
        ),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="TF-IDF vs BERT Weight (0=BERT, 1=TF-IDF)")
    ],
    outputs=[
        gr.HTML(label="🔍 Top Matching Jobs"),
        gr.HTML(label="🧠 Highlighted Resume")
    ],
    title="🎯 Resume-Internship Matcher",
    description="Upload your resume to find the best-matching internships based on TF-IDF and semantic similarity (BERT).",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(theme="default")

