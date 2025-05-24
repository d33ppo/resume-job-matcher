# app/main.py
import gradio as gr
from app.resume_parser import extract_text_from_pdf
from app.matcher import JobMatcher
from app.utils import clean_text

matcher = JobMatcher("data/job.csv")

def process_resume(file):
    text = extract_text_from_pdf(file.name)
    cleaned = clean_text(text)
    matched_jobs = matcher.match(cleaned)
    result = ""
    for i, row in matched_jobs.iterrows():
        result += f"**{row['job_title']}**\n{row['description']}\n\n"
    return result

demo = gr.Interface(
    fn=process_resume,
    inputs=gr.File(label="Upload Resume (PDF)"),
    outputs=gr.Markdown(label="Top Matching Jobs"),
    title="Resume Job Matcher",
    allow_flagging="never"  # Add this line to disable flagging
)

if __name__ == "__main__":
    demo.launch()
