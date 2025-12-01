🧠 Intelligent Resume-Based Job Matching & Skill-Gap Recommendation System
Using Streamlit, AWS Lambda, DynamoDB, Sentence Transformers & JSearch API
🚀 Project Overview

This project is an end-to-end AI-driven resume analysis and job-matching system.
It allows a user to upload a resume, extracts the content, generates embeddings using Sentence Transformers, fetches jobs using JSearch API, and compares both datasets to:

Identify matching jobs

Highlight missing skills

Provide ranking scores

Suggest improvements

This system uses AWS Lambda + S3 + DynamoDB as the backend and Streamlit as the UI layer.

🔧 Tech Stack (Actual Components Used)
🖥 Frontend

Streamlit

Resume upload

"Fetch Jobs" button

"Compare" (Agentic AI comparison)

Display job matches + scores

🧩 Backend

AWS S3 – Stores uploaded resumes

AWS Lambda – Triggered by S3 → parses resume

PyPDF – Extracts text from PDF resumes

Sentence Transformers – Generates semantic embeddings

Model used: all-MiniLM-L6-v2

AWS DynamoDB

resumes_meta table → parsed resume + embedding

jobs_meta table → JSearch API jobs + embedding

🌐 External APIs

JSearch Free API → Fetch job listings based on resume skills

🤖 AI/ML Components

Sentence Transformers for semantic similarity

Custom Agentic AI snippet (Claude/GPT) to compare:

Resume embedding

Job embedding

Missing skills

Final ranking score



📌 System Workflow
1️⃣ Resume Upload (Streamlit)

User uploads PDF

File pushed to S3

Triggers Lambda

2️⃣ Lambda Processing

Extract text using PyPDF

Generate embedding using SentenceTransformer

Extract:

Skills

Experience

Summary

Store all metadata in resumes_meta (DynamoDB)

3️⃣ Fetch Jobs (Streamlit → JSearch API)

Use resume skills to call JSearch

Clean job descriptions

Generate embeddings using SentenceTransformer

Store in jobs_meta (DynamoDB)

4️⃣ Agentic AI Comparison (Streamlit Button)

A Python function compares resume vs job using:

final_score = 0.55 * semantic_similarity
            + 0.25 * keyword_overlap
            + 0.10 * recency_weight
            + 0.10 * popularity_score


Outputs:

Match score

Missing skills

Why this job matches

Recommendations

5️⃣ Streamlit Visualization

Table of jobs + ranking score

Skill-gap insights

Suggestions for improvement

📊 Example Data Stored in DynamoDB
resumes_meta
Field	Description
user_id	Unique ID
extracted_text	Full resume text
skills	Parsed skills
embedding	384-dim vector from SentenceTransformer
timestamp	Upload time
jobs_meta
Field	Description
job_id	API job ID
title	Job role
summary	Job description
skills_required	Extracted from description
embedding	Semantic vector
posted_on	Recency score
💡 Features Implemented
✔ Resume parsing
✔ Embedding generation (sentence-transformers)
✔ Job retrieval via JSearch
✔ Resume vs Job comparison
✔ Final ranking score
✔ Skill-gap detection
✔ Streamlit dashboard
📝 Installation
pip install streamlit sentence-transformers boto3 pypdf requests


Run Streamlit:

streamlit run app.py

🔐 Environment Variables (.env)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
JSEARCH_API_KEY=
DYNAMODB_TABLE_RESUMES=
DYNAMODB_TABLE_JOBS=

📦 Project Structure
/lambda
   handler.py
/streamlit
   app.py
utils/
   embedding.py
   parser.py
   job_fetcher.py
   agent_compare.py
README.md

🧪 Final Output Example

“Match Score: 84%”

“Missing Skills: SQL, FastAPI”

“This job matches because your resume shows experience in ML pipelines...”

Ranked job list




BY PRIYADHARSHINI M

🏁 Conclusion

This project demonstrates a complete AI + Cloud + API pipeline using practical tools such as Streamlit, AWS Lambda, DynamoDB, and Sentence Transformers.
