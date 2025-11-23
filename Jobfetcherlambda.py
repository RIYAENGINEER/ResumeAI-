# handler.py
"""
Lambda handler: fetch jobs from JSearch (RapidAPI) for a resume and store to jobs_meta DynamoDB.
- Uses only requests + boto3
- No heavy libraries (no numpy / sentence-transformers)
- Saves job_id as partition key in jobs_meta (keeps user_id/resume_id as attributes)
- Adds original_resume_local_path for traceability: /mnt/data/MDTM47 final project.pdf
"""

import os
import time
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

import requests
import boto3
from botocore.exceptions import ClientError

# ---------- config ----------
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

RESUME_TABLE = os.environ.get("RESUME_TABLE", "resumes_meta")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "jobs_meta")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "jsearch.p.rapidapi.com")
JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"

JSEARCH_RESULTS = int(os.environ.get("JSEARCH_RESULTS", "20"))
MAX_VARIANTS_TO_TRY = int(os.environ.get("MAX_VARIANTS_TO_TRY", "6"))

# traceable local file path you uploaded
ORIGINAL_LOCAL_PATH = "/mnt/data/MDTM47 final project.pdf"

# boto3 tables
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
resume_table = dynamodb.Table(RESUME_TABLE)
jobs_table = dynamodb.Table(JOBS_TABLE)

# minimal skill keywords for fast heuristic extraction
COMMON_SKILLS = {
    "python","pandas","numpy","sql","excel","tableau","spark","aws","s3","redshift","hadoop",
    "tensorflow","pytorch","ml","machine learning","deep learning","nlp","scikit-learn",
    "javascript","react","node","java","spring","docker","kubernetes","rest","api",
    "c++","c#","go","bash","git","linux","html","css","agile","scrum","etl","powerbi",
    "statistics","scala","keras","matlab","nosql","mongodb","redis","airflow","fastapi","flask"
}

# ---------- helpers ----------
def mask_secret(s: Optional[str]) -> str:
    if not s:
        return "(not set)"
    if len(s) <= 8:
        return s[0:1] + "..." + s[-1:]
    return s[:4] + "..." + s[-4:]

def extract_keys_from_stream(record: Dict[str, Any]) -> Dict[str, str]:
    keys = record.get("dynamodb", {}).get("Keys", {})
    out = {}
    for k, v in keys.items():
        if isinstance(v, dict):
            # Dynamo stream shape: {"S": "value"} or {"N": "123"}
            for typ, val in v.items():
                out[k] = str(val)
                break
        else:
            out[k] = str(v)
    return out

def get_resume_item(user_id: str, resume_id: str) -> Dict[str, Any]:
    try:
        resp = resume_table.get_item(Key={"user_id": user_id, "resume_id": resume_id})
        return resp.get("Item", {}) or {}
    except ClientError:
        LOG.exception("DynamoDB get_item failed for resume %s/%s", user_id, resume_id)
        return {}

def build_query_variants(resume_item: Dict[str, Any]) -> List[str]:
    """
    Build focused query variants from resume fields:
      - explicit title/target_role/name
      - combos of top skills
      - short summary phrases
    """
    variants = []
    # title or target role
    title = resume_item.get("title") or resume_item.get("target_role") or resume_item.get("name") or ""
    if isinstance(title, dict):
        title = title.get("S", "") if "S" in title else str(title)
    title = str(title).strip()
    if title:
        variants.append(title)

    # skills
    raw_skills = resume_item.get("skills") or []
    skills_list = []
    if isinstance(raw_skills, str):
        skills_list = [s.strip() for s in raw_skills.split(",") if s.strip()]
    elif isinstance(raw_skills, list):
        for s in raw_skills:
            if isinstance(s, dict):
                if "S" in s:
                    skills_list.append(str(s["S"]))
                else:
                    # generic dict element
                    skills_list.append(str(next(iter(s.values()))))
            else:
                skills_list.append(str(s))
    # add skill combos
    if skills_list:
        # e.g., "python nlp pandas"
        variants.append(" ".join(skills_list[:3]))
        for s in skills_list[:3]:
            variants.append(f"{s} developer")
            variants.append(f"{s} engineer")

    # summary snippet
    summary = resume_item.get("summary") or ""
    if isinstance(summary, dict):
        summary = summary.get("S", "")
    summary = str(summary).strip()
    if summary:
        short = " ".join(summary.split()[:12])
        variants.append(short)

    # fallback generic
    if not variants:
        variants = ["data scientist", "software engineer", "backend developer", "machine learning engineer"]

    # dedupe & limit
    out = []
    seen = set()
    for v in variants:
        vv = " ".join(v.split()).strip()
        lv = vv.lower()
        if lv and lv not in seen:
            out.append(vv)
            seen.add(lv)
        if len(out) >= MAX_VARIANTS_TO_TRY:
            break
    LOG.info("Built query variants: %s", out)
    return out

def jsearch_call(query: str, results: int = JSEARCH_RESULTS, country: str = "in", timeout: int = 12) -> List[Dict[str, Any]]:
    """
    Call RapidAPI JSearch and return list of job dicts (body['data']) or [].
    """
    if not RAPIDAPI_KEY:
        LOG.error("RAPIDAPI_KEY env var is not set")
        return []
    url = JSEARCH_URL
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    params = {
        "query": query,
        "page": 1,
        "num_pages": 1,
        "limit": results,
        "country": country
    }
    try:
        LOG.info("Calling JSearch: %s ? %s", url, urlencode(params))
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        LOG.info("JSearch HTTP %s for query=%s", r.status_code, query)
        body_text = r.text or ""
        LOG.debug("JSearch raw body (truncated): %s", body_text[:1500])
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict):
            data = body.get("data") or body.get("results") or []
            if not isinstance(data, list):
                LOG.warning("JSearch returned non-list in data; returning empty")
                return []
            return data
        else:
            LOG.warning("Unexpected JSearch response type: %s", type(body))
            return []
    except Exception:
        LOG.exception("JSearch request failed for query=%s", query)
        return []

def extract_skills_from_text(text: str) -> List[str]:
    if not text:
        return []
    text_low = text.lower()
    found = set()
    for skill in COMMON_SKILLS:
        if skill in text_low:
            found.add(skill)
    # additional heuristic for common frameworks
    extra_tokens = ["react","django","flask","kubernetes","docker","spark"]
    for t in extra_tokens:
        if t in text_low:
            found.add(t)
    return sorted(found)

def sanitize_job_item(raw_job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize JSearch job dict to a flattened schema for DynamoDB.
    """
    title = raw_job.get("job_title") or raw_job.get("title") or raw_job.get("position") or ""
    snippet = raw_job.get("snippet") or raw_job.get("description") or raw_job.get("job_description") or ""
    description = raw_job.get("description") or raw_job.get("job_description") or snippet or ""
    company = raw_job.get("company_name") or raw_job.get("company") or ""
    job_url = raw_job.get("job_link") or raw_job.get("apply_link") or raw_job.get("url") or raw_job.get("job_url") or ""
    job_id_raw = raw_job.get("job_id") or raw_job.get("id") or job_url or (title[:120] if title else str(time.time()))
    created = raw_job.get("date") or raw_job.get("created") or raw_job.get("created_at") or ""

    skills = extract_skills_from_text(" ".join([title, description, company]))
    item = {
        "job_id": str(job_id_raw),
        "job_title": title,
        "job_description": description,
        "job_snippet": snippet[:600] if snippet else "",
        "company": company,
        "location": raw_job.get("location") or "",
        "url": job_url,
        "created_at": created,
        "skills": skills,
        "original_resume_local_path": ORIGINAL_LOCAL_PATH,
        "_raw": raw_job
    }
    return item

def save_job_item_to_dynamo(user_id: str, resume_id: str, job_item: Dict[str, Any]) -> bool:
    """
    Write job_item into jobs_meta. Table partition key expected to be 'job_id'.
    We include user_id and resume_id attributes for later filtering.
    """
    put_item = {
        "job_id": job_item.get("job_id"),
        "user_id": user_id,
        "resume_id": resume_id,
        "job_title": job_item.get("job_title"),
        "job_description": job_item.get("job_description"),
        "job_snippet": job_item.get("job_snippet"),
        "company": job_item.get("company"),
        "location": job_item.get("location"),
        "url": job_item.get("url"),
        "created_at": job_item.get("created_at"),
        "skills": job_item.get("skills"),
        "original_resume_local_path": job_item.get("original_resume_local_path"),
        "_raw": job_item.get("_raw"),
    }
    # remove None
    put_item = {k: v for k, v in put_item.items() if v is not None}
    try:
        jobs_table.put_item(Item=put_item)
        return True
    except ClientError:
        LOG.exception("Failed to put job item into DynamoDB for job_id=%s", job_item.get("job_id"))
        return False

# ---------- main --------------
def lambda_handler(event, context):
    """
    Accepts either DynamoDB stream record(s) or direct invocation with:
    { "user_id": "...", "resume_id": "..." }
    """
    LOG.info("Handler invoked. Event preview: %s", json.dumps(event)[:1000])

    # handle direct invocation convenience
    records = event.get("Records")
    if not records:
        user_id = event.get("user_id")
        resume_id = event.get("resume_id")
        if not user_id or not resume_id:
            LOG.error("Direct invocation requires user_id & resume_id")
            return {"statusCode": 400, "body": "user_id & resume_id required for direct invocation"}
        # emulate stream record
        records = [{"dynamodb": {"Keys": {"user_id": {"S": user_id}, "resume_id": {"S": resume_id}}}}]

    total_saved = 0
    try:
        for rec in records:
            keys = extract_keys_from_stream(rec)
            user_id = keys.get("user_id")
            resume_id = keys.get("resume_id")
            if not user_id or not resume_id:
                LOG.warning("Skipping record with missing keys: %s", keys)
                continue

            # fetch resume item
            resume_item = get_resume_item(user_id, resume_id)
            if not resume_item:
                LOG.warning("Resume not found for %s/%s - skipping", user_id, resume_id)
                continue

            # Build query variants
            query_variants = build_query_variants(resume_item)
            collected = []
            for variant in query_variants:
                try:
                    LOG.info("Trying JSearch variant: %s", variant)
                    results = jsearch_call(variant, results=JSEARCH_RESULTS, country="in")
                    LOG.info("Variant '%s' returned %d results", variant, len(results))
                    if results:
                        collected.extend(results)
                    time.sleep(0.2)
                except Exception:
                    LOG.exception("Error calling JSearch for variant %s", variant)

                # stop early if many collected
                if len(collected) >= (JSEARCH_RESULTS * 3):
                    break

            if not collected:
                LOG.info("No jobs collected for resume %s/%s", user_id, resume_id)
                continue

            # dedupe by job_id or url or title
            unique = {}
            for raw in collected:
                jid = raw.get("job_id") or raw.get("id") or raw.get("job_link") or raw.get("url") or raw.get("apply_link") or raw.get("title")
                key = str(jid) if jid else None
                if not key:
                    # fallback to hash of title+company
                    key = (raw.get("title") or raw.get("job_title") or "") + "|" + str(raw.get("company_name") or raw.get("company") or "")
                if key and key not in unique:
                    unique[key] = raw

            saved_count = 0
            for raw_job in unique.values():
                job_item = sanitize_job_item(raw_job)
                ok = save_job_item_to_dynamo(user_id, resume_id, job_item)
                if ok:
                    saved_count += 1

            LOG.info("For resume %s/%s saved %d jobs (unique candidates %d)", user_id, resume_id, saved_count, len(unique))
            total_saved += saved_count

        return {"statusCode": 200, "body": json.dumps({"saved": total_saved})}
    except Exception:
        LOG.exception("Unhandled exception in lambda_handler")
        return {"statusCode": 500, "body": json.dumps({"error": "internal"})}
