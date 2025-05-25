# app/main.py
import gradio as gr
from app.resume_parser import extract_text_from_pdf
from app.matcher import JobMatcher
from app.utils import clean_text

matcher = JobMatcher("data/job.json")

def process_resume(file):
    text = extract_text_from_pdf(file.name)
    cleaned = clean_text(text)
    matched_jobs = matcher.match(cleaned)

    result = ""
    for _, row in matched_jobs.iterrows():
        result += f"### 📌 {row['title']}\n"
        result += f"**Company:** {row['company']}\n"
        result += f"**Location:** {row['location']}\n"
        result += f"**Type:** {row['type']}\n"
        result += f"**Deadline:** {row['deadline']}\n\n"

        result += "**🔧 Description:**\n"
        for item in row['description']:  # Refer to original list
            result += f"- {item}\n"
        
        result += "\n**✅ Requirements:**\n"
        for item in row['requirements']:  # Refer to original list
            result += f"- {item}\n"

        result += f"\n📞 **Phone:** {row.get('phone', 'N/A')}\n"
        result += f"🔗 [More Info]({row['url']})\n"
        result += "---\n\n"

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
