# app/main.py
import gradio as gr
from app.resume_parser import extract_text_from_pdf
from app.matcher import JobMatcher
from app.utils import clean_text, highlight_matches, highlight_resume_text

matcher = JobMatcher("data/job.json")

def process_resume(file):
    text = extract_text_from_pdf(file.name)
    cleaned = clean_text(text)

    # Match top jobs using cleaned resume
    matched_jobs = matcher.match(cleaned)

    # Combine all descriptions + requirements from top jobs
    job_tokens = set()
    for _, row in matched_jobs.iterrows():
        job_text = ' '.join(row['description']) + ' ' + ' '.join(row['requirements'])
        tokens = matcher.vectorizer.build_tokenizer()(job_text.lower())
        job_tokens.update(tokens)

    # Highlight words in the resume that appear in job descriptions
    highlighted_resume = highlight_resume_text(text, job_tokens)

    # Tokenize resume for highlighting job descriptions
    resume_tokens = set(matcher.vectorizer.build_tokenizer()(cleaned.lower()))

    # Build job display
    result = ""
    for _, row in matched_jobs.iterrows():
        result += f"<h3>📌 {row['title']}</h3>"
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

    return result, f"<h3>📄 Highlighted Resume</h3><div style='white-space: pre-wrap;'>{highlighted_resume}</div>"


demo = gr.Interface(
    fn=process_resume,
    inputs=gr.File(label="Upload Resume (PDF)"),
    outputs=[
        gr.HTML(label="Top Matching Jobs"),
        gr.HTML(label="Highlighted Resume")
    ],
    title="Resume Job Matcher",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(theme="default")
