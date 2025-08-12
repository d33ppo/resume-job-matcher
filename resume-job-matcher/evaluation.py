import os
from app.matcher import JobMatcher
from app.resume_parser import extract_text_from_pdf

# 1. Map each resume filename to a list of relevant job_ids (as strings)
ground_truth = {
    "resume1.pdf": ["1", "3"],
    "resume2.pdf": ["2"],
    "resume3.pdf": ["1", "2", "4"],
    "resume4.pdf": ["3", "5"],
    "resume5.pdf": ["1", "2", "3", "4", "5"],
    "resume6.pdf": ["2", "4"],
    "resume7.pdf": ["1", "3", "5"],
    "resume8.pdf": ["2", "3"],
    "resume9.pdf": ["1", "4"],
    "resume10.pdf": ["2", "5"],
}

matcher = JobMatcher("data/job.json", method="hybrid")

def precision_at_k(predicted, relevant, k):
    predicted_k = predicted[:k]
    hits = sum(1 for job_id in predicted_k if job_id in relevant)
    return hits / k

def recall_at_k(predicted, relevant, k):
    predicted_k = predicted[:k]
    hits = sum(1 for job_id in predicted_k if job_id in relevant)
    return hits / len(relevant) if relevant else 0

def reciprocal_rank(predicted, relevant):
    for idx, job_id in enumerate(predicted, 1):
        if job_id in relevant:
            return 1 / idx
    return 0

def average_precision(predicted, relevant, k):
    ap = 0.0
    hits = 0
    for i, job_id in enumerate(predicted[:k], 1):
        if job_id in relevant:
            hits += 1
            ap += hits / i
    return ap / len(relevant) if relevant else 0

def evaluate(k=5):
    p_at_k_list = []
    r_at_k_list = []
    rr_list = []
    ap_list = []

    for fname, relevant in ground_truth.items():
        resume_path = os.path.join("data/sample_resumes", fname)
        if not os.path.exists(resume_path):
            print(f"Resume not found: {resume_path}")
            continue
        resume_text = extract_text_from_pdf(resume_path)
        matched = matcher.match(resume_text, top_k=k)
        # Use the correct job ID column (job_id as string)
        predicted = [str(row["job_id"]) for _, row in matched.iterrows()]

        p_at_k = precision_at_k(predicted, relevant, k)
        r_at_k = recall_at_k(predicted, relevant, k)
        rr = reciprocal_rank(predicted, relevant)
        ap = average_precision(predicted, relevant, k)

        print(f"{fname}: Precision@{k}={p_at_k:.2f}, Recall@{k}={r_at_k:.2f}, RR={rr:.2f}, AP={ap:.2f}")

        p_at_k_list.append(p_at_k)
        r_at_k_list.append(r_at_k)
        rr_list.append(rr)
        ap_list.append(ap)

    if p_at_k_list:
        print(f"\nAverage Precision@{k}: {sum(p_at_k_list)/len(p_at_k_list):.2f}")
        print(f"Average Recall@{k}: {sum(r_at_k_list)/len(r_at_k_list):.2f}")
        print(f"Mean Reciprocal Rank (MRR): {sum(rr_list)/len(rr_list):.2f}")
        print(f"Mean Average Precision (MAP): {sum(ap_list)/len(ap_list):.2f}")
    else:
        print("No results to evaluate.")

if __name__ == "__main__":
    evaluate(k=5)