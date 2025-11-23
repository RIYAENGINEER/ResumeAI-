
import os
import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import boto3
from botocore.exceptions import ClientError

# Optional LLM backend
LLAMA_PATH = os.environ.get("LLAMA_MODEL_PATH")  # path to GGUF model
LLAMA_AVAILABLE = False
MODEL = None

# Try to import llama-cpp-python if available
try:
    from llama_cpp import Llama
    if LLAMA_PATH and os.path.exists(LLAMA_PATH):
        # instantiate model once (small models only). You can tune n_ctx, n_gpu_layers etc.
        MODEL = Llama(model_path=LLAMA_PATH, n_ctx=2048)
        LLAMA_AVAILABLE = True
    else:
        LLAMA_AVAILABLE = False
except Exception:
    LLAMA_AVAILABLE = False

# Optional: sentence-transformers for embeddings fallback (not required)
try:
    from sentence_transformers import SentenceTransformer
    EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMB_AVAILABLE = True
except Exception:
    EMB_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS config (env)
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
RESUME_TABLE = os.environ.get("RESUME_TABLE", "resumes_meta")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "jobs_meta")

s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
resume_table = dynamodb.Table(RESUME_TABLE)
jobs_table = dynamodb.Table(JOBS_TABLE)

# ---------- basic helpers (upload, DB) ----------
def upload_file_to_s3(file_bytes: bytes, bucket: str, key: str, content_type: str = None) -> bool:
    try:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        s3.put_object(Bucket=bucket, Key=key, Body=file_bytes, **extra_args)
        return True
    except ClientError:
        logger.exception("S3 upload failed")
        return False

def write_resume_metadata(user_id: str, resume_id: str, file_name: str, s3_key: str) -> bool:
    item = {
        "user_id": user_id,
        "resume_id": resume_id,
        "file_name": file_name,
        "s3_key": s3_key,
        "created_at": datetime.utcnow().isoformat(),
        "summary": "",
        "skills": []
    }
    try:
        resume_table.put_item(Item=item)
        return True
    except ClientError:
        logger.exception("Failed to write resume metadata")
        return False

def fetch_resume_record(user_id: str, resume_id: Optional[str]) -> Dict[str, Any]:
    if resume_id:
        try:
            resp = resume_table.get_item(Key={"user_id": user_id, "resume_id": resume_id})
            return resp.get("Item", {})
        except ClientError:
            logger.exception("Failed to read resume item")
            return {}
    # fallback: return latest for user
    try:
        resp = resume_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("user_id").eq(user_id),
            Limit=1
        )
        items = resp.get("Items", [])
        return items[0] if items else {}
    except ClientError:
        logger.exception("Scan failed")
        return {}

def fetch_jobs_for_resume(user_id: Optional[str], resume_id: Optional[str]) -> List[Dict[str, Any]]:
    results = []
    try:
        if resume_id:
            resp = jobs_table.scan(FilterExpression=boto3.dynamodb.conditions.Attr("resume_id").eq(resume_id), Limit=50)
            results = resp.get("Items", [])
        if not results and user_id:
            resp = jobs_table.scan(FilterExpression=boto3.dynamodb.conditions.Attr("user_id").eq(user_id), Limit=50)
            results = resp.get("Items", [])
        if not results:
            resp = jobs_table.scan(Limit=20)
            results = resp.get("Items", [])
    except ClientError:
        logger.exception("Failed to fetch jobs")
    return results

# ---------- text processing helpers ----------
def extract_resume_text(resume_item: Dict[str, Any]) -> str:
    # prefer summary, else concat fields if present
    text_pieces = []
    if resume_item.get("summary"):
        text_pieces.append(resume_item.get("summary"))
    # include skills list
    skills = resume_item.get("skills") or []
    if isinstance(skills, list) and skills:
        text_pieces.append("Skills: " + ", ".join(skills))
    # optionally include other fields
    for k in ["objective", "experience", "projects", "education"]:
        if resume_item.get(k):
            text_pieces.append(str(resume_item.get(k)))
    return "\n\n".join(text_pieces) if text_pieces else "(no summary available)"

def aggregate_job_text(jobs: List[Dict[str, Any]], n_sample: int = 6) -> str:
    # take top N job titles + snippets
    pieces = []
    for j in jobs[:n_sample]:
        title = j.get("job_title") or ""
        snippet = j.get("job_snippet") or (j.get("job_description") or "")[:1000]
        pieces.append(f"{title}\n{snippet}")
    return "\n\n---\n\n".join(pieces) if pieces else "(no jobs found)"

# ---------- local LLM / agent helpers ----------
def craft_agent_prompt(resume_text: str, job_text: str, max_missing: int = 12) -> str:
    prompt = (
        "You are a helpful career agent. You will:\n"
        "1) Read the resume summary and job descriptions.\n"
        "2) Extract top skills and experience from the resume.\n"
        "3) Extract required skills from the job descriptions.\n"
        "4) Compare and list the top missing skills the candidate should add.\n"
        "5) Produce 6 short, actionable resume bullet points the candidate can copy-paste to show those skills (each <= 150 chars).\n"
        "6) Provide 3 quick learning resources (names or titles) for the top missing skills.\n\n"
        "Respond as JSON with keys: resume_skills (list), job_skills (list), missing_skills (list), suggestions (list of bullets), resources (list).\n\n"
        f"Resume:\n{resume_text}\n\nJob postings (several):\n{job_text}\n\n"
        "Now produce the JSON output only (no extra commentary)."
    )
    return prompt

def call_local_llm(prompt: str, max_tokens: int = 800) -> Optional[str]:
    if not LLAMA_AVAILABLE or MODEL is None:
        return None
    # Using llama-cpp-python chat completion / text generation
    try:
        # create a single completion
        out = MODEL.create(prompt=prompt, max_tokens=max_tokens, temperature=0.2, top_p=0.95)
        # MODEL.create returns dict with 'choices' or 'text' depending on version; handle both
        text = None
        if isinstance(out, dict):
            # older/newer versions vary
            if out.get("choices"):
                c = out["choices"][0]
                text = c.get("text") or c.get("message", {}).get("content") or ""
            else:
                text = out.get("content") or out.get("text") or ""
        elif hasattr(out, "choices"):
            # fallback
            text = out.choices[0].text
        else:
            text = str(out)
        return text
    except Exception:
        logger.exception("Local LLM call failed")
        return None

# ---------- safe JSON parser for LLM output ----------
def safe_parse_json_from_text(s: str) -> Optional[Dict[str, Any]]:
    import re, json
    if not s:
        return None
    # attempt to find first { ... } block
    m = re.search(r'(\{[\s\S]*\})', s)
    candidate = m.group(1) if m else s
    try:
        return json.loads(candidate)
    except Exception:
        # try some sanitization: replace single quotes with double (risky)
        try:
            cand2 = candidate.replace("'", '"')
            return json.loads(cand2)
        except Exception:
            logger.exception("Failed to parse JSON from LLM output")
            return None

# ---------- deterministic fallback (rule-based) ----------
def deterministic_suggestions(resume_text: str, jobs_text: str) -> Dict[str, Any]:
    # naive token extraction and set-difference
    import re
    def tokens(text):
        toks = re.split(r'[\s,;()/:.-]+', text.lower())
        toks = [t for t in toks if len(t) > 2]
        return toks
    resume_toks = tokens(resume_text)
    job_toks = tokens(jobs_text)
    # simple frequency
    from collections import Counter
    job_cnt = Counter(job_toks)
    top_job = [t for t,_ in job_cnt.most_common(30)]
    missing = [t for t in top_job if t not in resume_toks][:12]
    # create simple bullets
    suggestions = []
    for s in missing[:6]:
        suggestions.append(f"Worked on {s}: built a small project using {s} to solve [problem], outcome: [result].")
    resources = [f"Search 'Intro to {s}' on Coursera/YouTube" for s in missing[:3]]
    return {"resume_skills": resume_toks[:40], "job_skills": top_job, "missing_skills": missing, "suggestions": suggestions, "resources": resources}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Local LLM Agent — Resume → Jobs", layout="wide")
st.title("Local LLM Agent — Resume vs Jobs (Local model)")

st.sidebar.header("Environment & model")
st.sidebar.write(f"AWS region: `{AWS_REGION}`  \nS3 bucket: `{S3_BUCKET}`  \nResume table: `{RESUME_TABLE}`  \nJobs table: `{JOBS_TABLE}`")
if LLAMA_AVAILABLE:
    st.sidebar.success(f"LLM model loaded from {LLAMA_PATH}")
else:
    st.sidebar.warning("Local LLM not available — falling back to deterministic suggestions")

# Upload resume
st.header("1) Upload resume (S3 + DB)")
with st.form("upload_form"):
    user_id = st.text_input("User ID", value="anonymous")
    file = st.file_uploader("Choose resume (pdf/docx)", type=["pdf","docx"])
    submit = st.form_submit_button("Upload and register")
    if submit:
        if not file:
            st.error("Select a file first")
        elif not S3_BUCKET:
            st.error("S3_BUCKET not configured")
        else:
            resume_id = str(uuid.uuid4())
            ext = file.name.split(".")[-1]
            s3_key = f"resumes/{user_id}/{resume_id}.{ext}"
            ok = upload_file_to_s3(file.read(), S3_BUCKET, s3_key, content_type=file.type or "application/octet-stream")
            if ok:
                wrote = write_resume_metadata(user_id, resume_id, file.name, s3_key)
                if wrote:
                    st.success(f"Uploaded & registered resume_id={resume_id}")
                else:
                    st.error("Upload succeeded but metadata write failed")

# Fetch resume & jobs
st.header("2) Fetch resume record & related jobs")
with st.form("fetch_form"):
    view_user = st.text_input("User ID", value="anonymous")
    view_resume = st.text_input("Resume ID (optional)", value="")
    run = st.form_submit_button("Fetch resume and related jobs")
    if run:
        res_item = fetch_resume_record(view_user.strip(), view_resume.strip() or None)
        if not res_item:
            st.warning("No resume record found")
        else:
            st.subheader("Resume record")
            st.json(res_item)
            # prepare texts
            resume_text = extract_resume_text(res_item)
            jobs = fetch_jobs_for_resume(res_item.get("user_id"), res_item.get("resume_id"))
            st.subheader(f"Found {len(jobs)} related job postings (sample)")
            for j in jobs[:6]:
                st.markdown(f"**{j.get('job_title')}** — { (j.get('job_snippet') or (j.get('job_description') or ''))[:350] }")

            st.session_state["resume_item"] = res_item
            st.session_state["jobs_items"] = jobs

# Run agent
st.header("3) Run local LLM agent (analyze & suggest)")
if st.button("Run agent now"):
    resume_item = st.session_state.get("resume_item")
    jobs_items = st.session_state.get("jobs_items", [])
    if not resume_item:
        st.error("No resume selected. Fetch resume first.")
    else:
        rtext = extract_resume_text(resume_item)
        jtext = aggregate_job_text(jobs_items)
        st.subheader("Inputs (truncated)")
        st.write("Resume (summary):")
        st.write(rtext[:2000])
        st.write("Jobs aggregate (sample):")
        st.write(jtext[:2000])

        if LLAMA_AVAILABLE:
            st.info("Calling local LLM...")
            prompt = craft_agent_prompt(rtext, jtext)
            llm_out = call_local_llm(prompt)
            if llm_out:
                st.subheader("LLM raw output")
                st.code(llm_out[:10000])
                parsed = safe_parse_json_from_text(llm_out)
                if parsed:
                    st.subheader("Parsed agent result")
                    st.json(parsed)
                else:
                    st.error("LLM returned text but JSON parse failed — showing raw output above. You can re-run or use fallback.")
                    # optionally run deterministic fallback too
            else:
                st.error("Local LLM failed to produce output. Falling back to deterministic suggestions.")
                parsed = deterministic_suggestions(rtext, jtext)
                st.json(parsed)
        else:
            st.info("Local LLM unavailable — using deterministic fallback")
            parsed = deterministic_suggestions(rtext, jtext)
            st.json(parsed)

        # Store suggestions into session for user actions
        st.session_state["last_agent_result"] = parsed

# Display suggestions and allow copy
st.header("4) Suggestions & Actions")
result = st.session_state.get("last_agent_result")
if result:
    st.subheader("Missing skills")
    st.write(result.get("missing_skills", []))
    st.subheader("Suggested resume bullets")
    for b in result.get("suggestions", []):
        st.write("- " + b)
    st.subheader("Resources")
    for r in result.get("resources", []):
        st.write("- " + r)
    if st.button("Mark as applied / Save suggestions to DB"):
        # example: write suggestions into resume record for future display
        try:
            resume_item = st.session_state.get("resume_item", {})
            key = {"user_id": resume_item["user_id"], "resume_id": resume_item["resume_id"]}
            update = {"suggestions": result.get("suggestions", []), "missing_skills": result.get("missing_skills", [])}
            resume_table.update_item(Key=key, UpdateExpression="SET suggestions=:s, missing_skills=:m",
                                     ExpressionAttributeValues={":s": update["suggestions"], ":m": update["missing_skills"]})
            st.success("Saved suggestions into resume record")
        except Exception:
            st.exception("Save failed")

st.caption("Notes: local LLM must be installed & model path set in LLAMA_MODEL_PATH env var. If unavailable the app uses a deterministic fallback.")
