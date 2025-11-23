# handler.py
# Lambda handler: extract text from S3, extract contact & skills, summarize & embed via HuggingFace Router,
# store metadata in DynamoDB.
# Requirements in Lambda Layer: requests, PyPDF2
# Env vars required: HF_API_KEY, MY_AWS_REGION, DYNAMO_TABLE

import os
import io
import re
import json
import uuid
import zipfile
import traceback
from datetime import datetime
from urllib.parse import unquote_plus

import boto3
import requests
from PyPDF2 import PdfReader
import xml.etree.ElementTree as ET

# ----------------- Config -----------------
REGION = os.environ.get("MY_AWS_REGION", "ap-south-1")
DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE", "resumes_meta")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

if not HF_API_KEY:
    print("ERROR: HF_API_KEY environment variable not set. Summarization/embedding calls will fail.")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
ROUTER_BASE = "https://router.huggingface.co/hf-inference/models/"

# Models
SUMMARIZER_MODELS = [
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
    "google/pegasus-xsum"
]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------- AWS Clients -----------------
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(DYNAMO_TABLE)

# ----------------- Local extraction helpers -----------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9.\-+_]+@[a-zA-Z0-9.\-+_]+\.[a-zA-Z]+")
# phone regex that may capture groups -- we'll flatten later
PHONE_RE = re.compile(r"((?:\+?\d{1,3}[\s\-\.])?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{6,10})")

COMMON_SKILLS = {
    "python","java","c++","sql","aws","docker","kubernetes","spark","hadoop",
    "nlp","tensorflow","pytorch","scikit-learn","pandas","numpy","react","nodejs",
    "javascript","html","css","git","linux","rest","api","flask","django","excel",
    "powerbi","tableau","matlab","swift","r","bash","terraform"
}

def extract_contact_details(text: str):
    emails = list(dict.fromkeys(EMAIL_RE.findall(text)))  # unique preserve order
    phones_raw = PHONE_RE.findall(text)
    # flatten phone tuples / groups (PHONE_RE returns full match)
    phones = []
    for p in phones_raw:
        if isinstance(p, tuple):
            # sometimes findall returns tuples, join non-empty pieces
            phones.append("".join([x for x in p if x]))
        else:
            phones.append(p)
    name = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        # If email present, try line above email
        if emails:
            idx = next((i for i,l in enumerate(lines) if emails[0] in l), None)
            if idx and idx-1 >= 0:
                cand = lines[idx-1]
                if 1 < len(cand.split()) <= 4:
                    name = cand
        if not name:
            first = lines[0]
            if len(first.split()) <= 4 and any(c.isalpha() for c in first) and first[0].isupper():
                name = first
    return name, emails, phones

def extract_skills_from_text(text: str, extra_skill_list=None):
    skills = set()
    if extra_skill_list:
        skillset = set(s.lower() for s in extra_skill_list) | COMMON_SKILLS
    else:
        skillset = COMMON_SKILLS
    txt = text.lower()
    for skill in skillset:
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", txt):
            skills.add(skill)
    return sorted(skills)

# ----------------- PDF / DOCX extraction -----------------
def extract_text_from_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception as e:
        print("WARN: PDF extraction failed:", e)
        return ""

def extract_text_from_docx(data: bytes) -> str:
    try:
        texts = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                for node in root.iter():
                    if node.tag.endswith("}t") and node.text:
                        texts.append(node.text)
        return "\n".join(texts).strip()
    except Exception as e:
        print("WARN: DOCX extraction failed:", e)
        return ""

def extract_text_from_s3(bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    txt = extract_text_from_pdf(data)
    if txt:
        return txt
    txt = extract_text_from_docx(data)
    if txt:
        return txt
    return "[UNABLE_TO_EXTRACT_TEXT]"

# ----------------- HuggingFace Router helpers -----------------
def call_router_model(model_id: str, payload: dict, timeout: int = 60):
    url = ROUTER_BASE + model_id
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
        return r
    except requests.RequestException as e:
        print(f"WARN: request to model {model_id} failed: {e}")
        return None

def hf_summarize(text: str, max_length: int = 150) -> str:
    if not text:
        return ""
    payload = {"inputs": text, "parameters": {"min_length": 20, "max_length": max_length}}
    for model in SUMMARIZER_MODELS:
        print(f"[HF] Trying summarizer: {model}")
        r = call_router_model(model, payload)
        if r is None:
            continue
        print(f"[HF] {model} status: {r.status_code}")
        try:
            j = r.json()
        except Exception:
            print("[HF] non-json response:", r.text[:300])
            j = None
        if r.status_code == 200 and isinstance(j, list) and j:
            summary = j[0].get("summary_text") or j[0].get("generated_text") or j[0].get("text")
            if summary:
                return summary.strip()
            else:
                return json.dumps(j)[:1000]
        if r.status_code == 410:
            print(f"[HF] Model {model} returned 410 (Gone). Trying fallback.")
            continue
        if r.status_code in (502, 503, 504):
            print(f"[HF] transient error {r.status_code} for {model}.")
            continue
        print(f"[HF] model {model} returned {r.status_code}. Body: {r.text[:400]}")
    return "[SUMMARIZATION_FAILED]"

def hf_embed(text: str) -> list:
    if not text:
        return []
    trim = text if len(text) <= 4000 else text[:4000]
    payload = {"inputs": trim}
    r = call_router_model(EMBEDDING_MODEL, payload)
    if not r:
        print("[HF] embedding request failed (no response).")
        return []
    print(f"[HF] embedding status: {r.status_code}")
    try:
        out = r.json()
    except Exception:
        print("[HF] embedding non-json response:", r.text[:400])
        return []
    if isinstance(out, dict) and "embedding" in out:
        return out["embedding"]
    if isinstance(out, list) and len(out) > 0:
        if isinstance(out[0], dict) and "embedding" in out[0]:
            return out[0]["embedding"]
        if isinstance(out[0], (int, float)):
            return out
    print("[HF] embedding response shape unexpected:", type(out))
    return []

# ----------------- DynamoDB write helper -----------------
def save_resume_metadata(item: dict):
    try:
        resp = table.put_item(Item=item)
        print("DEBUG: DynamoDB put_item response:", resp)
        return resp
    except Exception as e:
        print("ERROR: DynamoDB put_item failed:", str(e))
        traceback.print_exc()
        raise

# ----------------- Lambda handler -----------------
def lambda_handler(event, context):
    print("Event (truncated):", json.dumps(event)[:2000])
    results = []
    records = event.get("Records", [])
    if not records:
        print("No records found in event.")
        return {"status": "no_records"}

    for rec in records:
        try:
            s3_info = rec.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name")
            raw_key = s3_info.get("object", {}).get("key")
            if not bucket or not raw_key:
                print("Skipping record with missing s3 info:", rec)
                continue

            key = unquote_plus(raw_key)
            print(f"Processing S3 object: bucket={bucket}, key={key}")

            text = extract_text_from_s3(bucket, key)
            print(f"Extracted text length: {len(text)}")

            # build ids & file metadata
            resume_id = str(uuid.uuid4())
            file_name = key.split("/")[-1]
            parts = key.split("/")
            user_id = "anonymous"
            if len(parts) >= 2 and parts[0].lower().startswith("resumes"):
                user_id = parts[1] or "anonymous"

            # extract contact & skills
            name, emails, phones = extract_contact_details(text)
            skills = extract_skills_from_text(text, extra_skill_list=None)

            # summarization + embedding
            if text.startswith("[UNABLE_TO_EXTRACT_TEXT]"):
                summary = text
                embedding = []
            else:
                try:
                    summary = hf_summarize(text)
                except Exception as e:
                    print("WARN: summarization failed:", e)
                    traceback.print_exc()
                    summary = "[SUMMARIZATION_EXCEPTION]"

                try:
                    embedding = hf_embed(text)
                except Exception as e:
                    print("WARN: embedding failed:", e)
                    traceback.print_exc()
                    embedding = []

            item = {
                "resume_id": resume_id,   # Partition key - must match your table
                "user_id": user_id,
                "name": name or "",
                "emails": emails,
                "phones": phones,
                "skills": skills,
                "file_name": file_name,
                "s3_bucket": bucket,
                "s3_key": key,
                "summary": summary,
                "embedding": json.dumps(embedding),
                "created_at": datetime.utcnow().isoformat()
            }

            print("DEBUG: Prepared item for DynamoDB write:", {
                "resume_id": item.get("resume_id"),
                "file_name": item.get("file_name"),
                "s3_key": item.get("s3_key"),
                "s3_bucket": item.get("s3_bucket"),
                "summary_len": len(item.get("summary") or ""),
                "skills_count": len(item.get("skills") or [])
            })

            resp = save_resume_metadata(item)
            results.append({"resume_id": resume_id, "dynamo_response": resp})

        except Exception as e:
            print("ERROR: processing record failed:", e)
            traceback.print_exc()
            # continue to next record

    print("Finished processing records. Count:", len(results))
    return {"status": "ok", "processed": len(results)}
