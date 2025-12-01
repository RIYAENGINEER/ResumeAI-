import streamlit as st
import requests
import os
import json
import time
import boto3
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from math import exp
import numpy as np
import pandas as pd
import streamlit as st

# try sentence-transformers, else use sklearn TF-IDF
USE_SENTENCE_TRANSFORMER = False
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast
    USE_SENTENCE_TRANSFORMER = True
except Exception:
    USE_SENTENCE_TRANSFORMER = False

# sklearn fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DynamoDB client
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
RESUMES_TABLE = os.environ.get("RESUME_TABLE", "resumes_meta")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "jobs_meta")

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
resumes_table = dynamodb.Table(RESUMES_TABLE)
jobs_table = dynamodb.Table(JOBS_TABLE)

# --- Utilities ---
def _get_latest_resume_for_user(user_id):
    """Fetch latest resume item for user_id from resumes_meta table.
       Assumes resume entries have 'updated_ts' or numeric resume_id timestamp.
    """
    # Simple scan + filter (ok for small dev table). For production use query with GSIs.
    resp = resumes_table.scan(FilterExpression=None) if False else resumes_table.scan()
    items = resp.get("Items", [])
    # filter by user_id
    user_items = [it for it in items if it.get("user_id") == user_id]
    if not user_items:
        return None
    # try sort by updated_ts ISO otherwise resume_id numeric
    def _ts(item):
        t = item.get("updated_ts") or item.get("created_ts") or item.get("resume_id")
        try:
            return datetime.fromisoformat(t)
        except Exception:
            try:
                return datetime.fromtimestamp(float(t)/1000)
            except Exception:
                return datetime.fromtimestamp(0)
    user_items_sorted = sorted(user_items, key=_ts, reverse=True)
    return user_items_sorted[0]

def _scan_jobs(filter_location=None):
    """Scan jobs_meta table; optional filter by location substring."""
    resp = jobs_table.scan()
    items = resp.get("Items", [])
    if filter_location:
        fl = filter_location.lower()
        items = [it for it in items if fl in str(it.get("location","")).lower() or fl in str(it.get("job_city","")).lower() or fl in str(it.get("job_country","")).lower()]
    return items

def _ensure_text(x):
    return x if isinstance(x, str) and x.strip() else ""

def _extract_keywords_top_n(corpus_texts, top_n=20):
    """Return top_n keywords per document using TF-IDF feature ranking.
       Returns list-of-lists of keywords (strings).
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords_per_doc = []
    for row in X:
        row_arr = row.toarray().ravel()
        if row_arr.sum() == 0:
            keywords_per_doc.append([])
            continue
        top_idx = row_arr.argsort()[::-1][:top_n]
        kws = [feature_names[i] for i in top_idx if row_arr[i] > 0]
        keywords_per_doc.append(kws)
    return keywords_per_doc

def compute_semantic_similarity_pairs(resume_text, job_texts):
    """Return list of semantic similarities in [0,1] between resume and each job text.
       Uses sentence-transformers if available, else TF-IDF cosine fallback.
    """
    resume_text = _ensure_text(resume_text)
    job_texts = [_ensure_text(t) for t in job_texts]
    if USE_SENTENCE_TRANSFORMER:
        try:
            emb_resume = model.encode(resume_text, convert_to_tensor=True)
            emb_jobs = model.encode(job_texts, convert_to_tensor=True)
            sims = st_util.cos_sim(emb_resume, emb_jobs).cpu().numpy().ravel()
            # clamp to 0..1 (though cosine might be -1..1)
            sims = ((sims + 1) / 2).clip(0,1)
            return sims.tolist()
        except Exception:
            pass
    # fallback TF-IDF cosine
    texts = [resume_text] + job_texts
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < 2:
        return [0.0]*len(job_texts)
    resume_vec = X[0]
    job_vecs = X[1:]
    sims = cosine_similarity(resume_vec, job_vecs).ravel()
    # cosine already 0..1 for TFIDF non-negative
    sims = np.nan_to_num(sims).clip(0,1)
    return sims.tolist()

def compute_keyword_overlap_score(resume_kws, job_kws):
    """Compute Jaccard-like overlap between two keyword sets; output 0..1"""
    rs = set([k.lower() for k in resume_kws])
    js = set([k.lower() for k in job_kws])
    if not rs and not js:
        return 0.0
    if not rs:
        return 0.0
    inter = rs & js
    union = rs | js
    return len(inter) / len(union) if union else 0.0

def compute_recency_weight(created_ts):
    """Map job created timestamp (ISO or epoch) -> score in 0..1.
       Simple linear decay: score = max(0, 1 - days/90)
    """
    if not created_ts:
        return 0.0
    try:
        # try ISO
        dt = datetime.fromisoformat(created_ts)
    except Exception:
        try:
            # maybe epoch seconds/millis
            t = float(created_ts)
            if t > 1e12:  # millis
                dt = datetime.fromtimestamp(t/1000, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(t, tz=timezone.utc)
        except Exception:
            return 0.0
    now = datetime.now(dt.tzinfo or timezone.utc)
    delta_days = (now - dt).days
    score = max(0.0, 1.0 - (delta_days / 90.0))
    return float(score)

def compute_popularity_score(job_item):
    """Try to extract a popularity metric (views, apply_count, popularity) into 0..1.
       Fallback: check if employer_logo or job_apply_link present -> 0.5
    """
    for k in ("popularity", "views", "apply_count", "num_applicants"):
        v = job_item.get(k)
        if v is None:
            continue
        try:
            v = float(v)
            # naive normalization: assume reasonable maxima
            return float(max(0.0, min(1.0, v / (v + 100.0))))
        except Exception:
            continue
    # fallback heuristic
    if job_item.get("employer_logo") or job_item.get("job_apply_link"):
        return 0.5
    return 0.1

# --- Main compare function ---
def compare_resume_with_jobs(user_id, top_k=10, location_filter=None):
    """
    Returns DataFrame of top_k jobs and suggestions per job:
    columns: job_id, title, company, final_score, semantic, keyword_overlap, recency, popularity, missing_skills
    """
    # 1. load resume
    resume_item = _get_latest_resume_for_user(user_id)
    if not resume_item:
        return {"error": "no_resume", "message": f"No resume found for user {user_id}", "results": []}
    resume_text = _ensure_text(resume_item.get("extracted_text",""))
    if not resume_text:
        return {"error": "empty_resume", "message": "Resume text empty", "results": []}

    # 2. load jobs
    job_items = _scan_jobs(filter_location=location_filter)
    if not job_items:
        return {"error": "no_jobs", "message": "No jobs found in jobs_meta (for given filter)", "results": []}

    # 3. prepare texts
    job_texts = []
    job_ids = []
    for it in job_items:
        desc = it.get("description") or it.get("job_description") or it.get("job_text") or it.get("snippet") or ""
        job_texts.append(_ensure_text(desc))
        job_ids.append(it.get("job_id") or it.get("id") or it.get("url") or str(time.time()))

    # 4. semantic sims
    semantic_sims = compute_semantic_similarity_pairs(resume_text, job_texts)

    # 5. keyword extraction (top keywords via TF-IDF across resume+jobs)
    corpus = [resume_text] + job_texts
    kws_list = _extract_keywords_top_n(corpus, top_n=30)
    resume_kws = kws_list[0]
    job_kws_list = kws_list[1:]

    # 6. compute components & final score
    rows = []
    for idx, it in enumerate(job_items):
        semantic = float(semantic_sims[idx]) if idx < len(semantic_sims) else 0.0
        job_kws = job_kws_list[idx] if idx < len(job_kws_list) else []
        kw_overlap = compute_keyword_overlap_score(resume_kws, job_kws)
        recency = compute_recency_weight(it.get("created_ts") or it.get("created") or it.get("date_posted") or "")
        popularity = compute_popularity_score(it)

        final_score = 0.55*semantic + 0.25*kw_overlap + 0.10*recency + 0.10*popularity
        missing_skills = sorted(list(set([k.lower() for k in job_kws]) - set([k.lower() for k in resume_kws])))
        rows.append({
            "job_id": job_ids[idx],
            "title": it.get("title") or it.get("job_title") or "",
            "company": it.get("company") or it.get("employer_name") or "",
            "semantic": round(semantic, 4),
            "keyword_overlap": round(kw_overlap, 4),
            "recency": round(recency, 4),
            "popularity": round(popularity, 4),
            "final_score": round(float(final_score), 4),
            "missing_skills": missing_skills,
            "raw_item": it
        })

    # sort by final_score desc
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    top_df = df.head(top_k)
    return {"error": None, "message": "ok", "results": top_df, "resume_kws": resume_kws}


def fetch_jobs_5(query, rapidapi_key):
    url = "https://jsearch.p.rapidapi.com/search"

    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    params = {
        "query": query,
        "page": 1,
        "num_pages": 1,
        "results_per_page": 5,     # fetch 5 directly (if supported)
        "language": "en",
        "country": "us"            # guaranteed to work for your free tier
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)

    if r.status_code != 200:
        st.error(f"JSearch Error {r.status_code}: {r.text[:300]}")
        return []

    data = r.json().get("data", [])

    # fallback to top 5 if API ignores results_per_page
    return data[:5]



# env / clients (adjust names to your env)
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "jobs_meta")

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
jobs_table = dynamodb.Table(JOBS_TABLE)

def _normalize_job(raw, user_id=None):
    """Return a normalized dict ready for DynamoDB put_item"""
    # Try multiple common keys across different providers
    job_id = (
        raw.get("job_id")
        or raw.get("id")
        or raw.get("jid")
        or raw.get("job_id_slug")
        or raw.get("uuid")
        or raw.get("link")
    )
    if not job_id:
        job_id = f"job_{int(time.time()*1000)}"

    title = raw.get("job_title") or raw.get("title") or raw.get("position") or ""
    company = raw.get("employer_name") or raw.get("company_name") or raw.get("company") or ""
    location = raw.get("job_city") or raw.get("location") or raw.get("job_country") or ""
    description = raw.get("job_description") or raw.get("description") or raw.get("snippet") or ""
    url = raw.get("job_apply_link") or raw.get("apply_link") or raw.get("url") or raw.get("link") or ""
    created = raw.get("created") or raw.get("date_posted") or raw.get("post_date") or ""

    item = {
        "job_id": str(job_id),
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "url": url,
        "created_ts": created,
        "source": "jsearch",
        # keep a raw snapshot for debugging but truncate so we don't exceed item size
        "raw": json.dumps(raw)[:20000]
    }

    # optional: add user_id if you want to keep job per-user in same table
    if user_id:
        item["user_id"] = user_id

    # add write timestamp
    item["written_ts"] = datetime.now(timezone.utc).astimezone().isoformat()
    return item

def write_jobs_to_dynamo(jobs, user_id=None, dedupe_on_job_id=True):
    """
    Write a list of raw job dicts to DynamoDB jobs_table.
    Returns (written_count, failed_items_list).
    """
    if not isinstance(jobs, (list, tuple)):
        return 0, [{"error": "jobs is not a list"}]

    written = 0
    failed = []
    for raw in jobs:
        try:
            item = _normalize_job(raw, user_id=user_id)

            # If your table uses job_id as PK -> simple put_item.
            # If your table uses composite key like (user_id, job_id), ensure user_id is present.
            jobs_table.put_item(Item=item)
            written += 1

            # If you prefer to avoid overwriting existing items, use conditional put:
            # jobs_table.put_item(Item=item, ConditionExpression="attribute_not_exists(job_id)")
            # wrap that in try/except botocore.exceptions.ClientError to catch ConditionalCheckFailedException

        except Exception as e:
            failed.append({"raw": raw, "error": str(e)})
            # continue writing others

    return written, failed



# --- config ---
S3_BUCKET = "resume-processing-riya96-bucket"   # your bucket
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")  # adjust if needed

# create client (will use env/instance credentials)
s3_client = boto3.client("s3", region_name=AWS_REGION)

st.header("Upload resume (PDF) to S3")

# get user id so we can put file under resumes/{user_id}/...
# put this once at the top of the app (replace any other user_id text_input)
user_id = st.text_input("User ID (used for S3 path)", value="", key="canonical_user_id",
                       help="Enter user id or email used by your app (single authoritative field)")
# show currently used value
st.write("Using user id:", user_id)


uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf", "docx", "txt"])

if st.button("Upload to S3"):
    if uploaded_file is None:
        st.error("Please choose a file first.")
    elif not user_id:
        st.error("Please enter a user id.")
    else:
        # normalize filename and build S3 key
        original_filename = uploaded_file.name
        # generate timestamped filename to avoid collisions
        ts = int(time.time() * 1000)
        safe_filename = f"{ts}_{original_filename.replace(' ', '_')}"
        s3_key = f"resumes/{user_id}/{safe_filename}"

        # detect content type simply from extension (improve if needed)
        ext = original_filename.lower().split(".")[-1]
        content_type = "application/pdf" if ext == "pdf" else "application/octet-stream"
        if ext == "docx":
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext == "txt":
            content_type = "text/plain"

        # Upload using upload_fileobj (streaming, memory-friendly)
        try:
            # uploaded_file is a SpooledTemporaryFile / BytesIO-like object
            # rewind to start (just in case)
            uploaded_file.seek(0)
            s3_client.upload_fileobj(
                Fileobj=uploaded_file,
                Bucket=S3_BUCKET,
                Key=s3_key,
                ExtraArgs={
                    "ContentType": content_type,
                    "ACL": "private"
                }
            )

            st.success(f"Upload successful: s3://{S3_BUCKET}/{s3_key}")
            st.write("S3 key:", s3_key)
            # optionally return key to use in next steps
            # e.g., store s3_key in DynamoDB or trigger processing Lambda
        except ClientError as e:
            st.error(f"Upload failed: {e.response.get('Error', {}).get('Message', str(e))}")
        except Exception as e:
            st.error(f"Unexpected error during upload: {e}")





# ---------- DYNAMIC JOB FETCH FROM RESUME ----------
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def build_dynamic_query_from_resume(resume_text, top_n=6):
    """
    Build a concise query string from resume_text.
    Steps:
      1. Try to extract a short title/headline (first line or 'Profile' heading)
      2. Extract top TF-IDF keywords (unigrams + bigrams)
      3. Heuristic: prefer tech/skill tokens (letters, numbers, hyphens)
      4. Return a comma-separated query (max top_n tokens)
    """
    resume_text = (resume_text or "").strip()
    if not resume_text:
        return "software developer"  # safe fallback

    # 1) try headline / first non-empty line
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    headline = lines[0] if lines else ""
    headline_tokens = []
    if headline and len(headline.split()) <= 6:
        # use words from headline that look like titles/skills
        headline_tokens = re.findall(r"[A-Za-z0-9\+\-#\.]{2,}", headline)
        headline_tokens = [t for t in headline_tokens if len(t) > 1]

    # 2) TF-IDF keywords
    try:
        vect = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
        X = vect.fit_transform([resume_text])
        feat = np.array(vect.get_feature_names_out())
        scores = X.toarray().ravel()
        top_idx = scores.argsort()[::-1]
        tfidf_keywords = [feat[i] for i in top_idx if scores[i] > 0][:top_n*3]
    except Exception:
        tfidf_keywords = []

    # 3) skill-like filter: prefer tokens containing letters/numbers (drop punctuation-only)
    candidates = []
    # prefer headline tokens first
    for t in headline_tokens:
        if t.lower() not in candidates:
            candidates.append(t.lower())
    for t in tfidf_keywords:
        tok = t.lower()
        # drop overly long phrases
        if len(tok) > 60:
            continue
        # filter out generic words
        if tok in ("experience","years","year","worked","responsible","profile","summary"):
            continue
        if tok not in candidates:
            candidates.append(tok)
    # 4) ensure we have at least something relevant
    if not candidates:
        # simple fallback: common tech tokens detection
        found = re.findall(r"(python|java|c\+\+|c#|javascript|react|node|sql|aws|docker|kubernetes|spark|etl)", resume_text, flags=re.I)
        candidates = [f.lower() for f in found]
    # final top N unique
    final = []
    for c in candidates:
        if c not in final:
            final.append(c)
        if len(final) >= top_n:
            break

    # build readable query: prefer comma separated short tokens
    query = ", ".join(final) if final else "software developer"
    return query

# Streamlit UI to trigger dynamic fetch
st.header("🔍 Fetch related Jobs from JSearch (Dynamic Query)")

# Ensure you have a user_id input above in the app; reuse it
# user_id = st.text_input("User ID (used for S3 path)", value="user_test")  # already in your app

# show last resume summary (if any)
resume_item = _get_latest_resume_for_user(user_id)
resume_text = resume_item.get("extracted_text","") if resume_item else ""
# if not resume_item:
#      #st.warning("No resume found for this user. Upload a resume first.")
# else:
#     st.info("Found latest resume for user. Generating query from content...")

# Build query and display to user
#query = build_dynamic_query_from_resume(resume_text, top_n=6)
#st.write("🔎 Dynamic Query Generated:", query)

# allow user to tweak the query if they want (optional)
query = st.text_input("Dynamic query ")

if st.button("Fetch related Jobs (dynamic)"):
    st.info(f"Using dynamic query: {query!r} — calling JSearch...")
    # call the existing fetch function
    try:
        jobs = fetch_jobs_5(query, rapidapi_key=os.environ.get("RAPIDAPI_KEY",""))
    except Exception as e:
        st.error("fetch_jobs_5 call failed: " + str(e))
        jobs = []

    if not jobs:
        st.warning("No jobs returned. Possible causes: API key missing/invalid, rate limit, or query produced no results.")
        # optionally show mock suggestions or let user tweak query
    else:
        st.success(f"Fetched {len(jobs)} jobs from JSearch.")
        # display returned job titles
        st.write("Sample titles returned:")
        for j in jobs[:10]:
            st.write("-", j.get("job_title") or j.get("title") or j.get("position") or "<no title>")

        # write to DynamoDB (and show results)
        try:
            written, failed = write_jobs_to_dynamo(jobs, user_id=user_id.strip())
            st.success(f"Wrote {written} jobs to {JOBS_TABLE}")
            if failed:
                st.error(f"{len(failed)} jobs failed to write — showing up to 5 failed items")
                st.json(failed[:5])
        except Exception as e:
            st.error("Failed to write fetched jobs to DynamoDB: " + str(e))

# def build_dynamic_query_from_resume(resume_text, top_n=5):
#     """
#     Build a smart job search query from resume using TF-IDF keywords.
#     Returns a simple comma-separated string to pass into JSearch.
#     """
#     if not resume_text.strip():
#         return "software developer"  # fallback

#     # Extract keywords using TF-IDF
#     vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
#     X = vectorizer.fit_transform([resume_text])
#     features = vectorizer.get_feature_names_out()
#     tfidf_scores = X.toarray().ravel()

#     # Get top N keywords
#     top_idx = tfidf_scores.argsort()[::-1][:top_n]
#     top_keywords = [features[i] for i in top_idx if tfidf_scores[i] > 0]

#     # Join keywords into a query string
#     if len(top_keywords) == 0:
#         return "software engineer"

#     query = ", ".join(top_keywords)
#     return query
# resume_item = _get_latest_resume_for_user(user_id)
# resume_text = resume_item.get("extracted_text", "") if resume_item else ""

# build dynamic query from resume content
# query = build_dynamic_query_from_resume(resume_text, top_n=5)

# st.write("🔍 Dynamic Query Generated:", query)


# run = st.button("Fetch related Jobs")

# if run:
#     st.info(f"Fetching related jobs for query: '{query}' ...")
    
#     jobs = fetch_jobs_5(query, rapidapi_key="99f297a231msh260adff4d3b771dp14701cjsn5fbbc98e4ca0")

    

    

#     written, failed = write_jobs_to_dynamo(jobs, user_id=user_id)
#     st.success(f"Wrote jobs to {JOBS_TABLE}")
#     if failed:
#         st.error(f"{len(failed)} jobs failed to write — check logs")
#         st.write(failed[:5])

    # if not jobs:
    #     st.warning("No jobs returned. Try a different keyword (sales, developer, US-based, etc.)")

    # for i, job in enumerate(jobs, start=1):
    #     st.subheader(f"Job #{i}: {job.get('job_title','Unknown')}")
    #     st.write("**Company:**", job.get("employer_name", "N/A"))
    #     st.write("**Location:**", job.get("job_city","N/A"), job.get("job_country","N/A"))
    #     st.write("**Description:**")
    #     st.write(job.get("job_description", "No description available."))
    #     st.write("**Apply link:**", job.get("job_apply_link","N/A"))
    #     st.markdown("---")

# after fetching `jobs` via fetch_jobs_5 or call_jsearch_live
# user_id = st.text_input("User ID for writes", value="user_test@example.com")
# if st.button("Write fetched jobs to DynamoDB"):
#     written, failed = write_jobs_to_dynamo(jobs, user_id=user_id)
#     st.success(f"Wrote {written} jobs to {JOBS_TABLE}")
#     if failed:
#         st.error(f"{len(failed)} jobs failed to write — check logs")
#         st.write(failed[:5])
def run_agentic_ui():
    st.header("Agentic Resume ↔ Jobs analysis")
    
    top_k = st.slider("Top K to show", 1, 50, 10)

    if st.button("Run Agentic Compare"):
        with st.spinner("Comparing resume with jobs..."):
            out = compare_resume_with_jobs(user_id=user_id, top_k=top_k)
            # --- DIAGNOSTIC: show what agent sees (paste before running compare) ---
        st.subheader("DEBUG: Agent view (resume -> query -> jobs -> scores)")

        # Ensure canonical user_id is used
        uid = user_id.strip() if 'user_id' in globals() else (st.session_state.get("canonical_user_id","").strip() or "")

        # show resume item
        resume_item_dbg = _get_latest_resume_for_user(uid)
        if not resume_item_dbg:
            st.error(f"No resume found for user '{uid}' (DEBUG). Check user id / upload)")
        else:
            resume_text_dbg = resume_item_dbg.get("extracted_text","")
            st.markdown("**Resume text (first 800 chars):**")
            st.text(resume_text_dbg[:800])

            # generate the current query using your function (if exists) or fallback
            try:
                dyn_q = build_dynamic_query_from_resume(resume_text_dbg, top_n=8)
            except Exception as e:
                dyn_q = "software developer"
                st.write("build_dynamic_query_from_resume error:", e)
            st.write("Generated query:", dyn_q)

            # get jobs currently in DB (or call fetch to get live)
            job_items_dbg = _scan_jobs()
            st.write("Jobs in DB count:", len(job_items_dbg))

            # collect job_texts and titles
            job_texts_dbg = []
            job_titles_dbg = []
            for it in job_items_dbg[:50]:
                desc = it.get("description") or it.get("job_description") or it.get("job_text") or it.get("snippet") or ""
                job_texts_dbg.append(_ensure_text(desc))
                job_titles_dbg.append(it.get("title") or it.get("job_title") or it.get("position") or "<no title>")

            st.write("Sample job titles (first 10):", job_titles_dbg[:10])

            # compute semantic sims (call your function)
            try:
                sims_dbg = compute_semantic_similarity_pairs(resume_text_dbg, job_texts_dbg) if job_texts_dbg else []
            except Exception as e:
                sims_dbg = []
                st.write("compute_semantic_similarity_pairs error:", e)

            # show top 10 sims
            if sims_dbg:
                sims_with_titles = list(zip(job_titles_dbg, sims_dbg))
                sims_with_titles_sorted = sorted(sims_with_titles, key=lambda x: x[1], reverse=True)[:10]
                st.write("Top 10 jobs by semantic sim (title, score):")
                for t,s in sims_with_titles_sorted:
                    st.write(f"- {t}  →  {round(s,4)}")
            else:
                st.write("No semantic sims computed (empty job_texts or error).")

            # TF-IDF keywords for resume and first few jobs
            try:
                c = [resume_text_dbg] + job_texts_dbg[:10]
                kwlists = _extract_keywords_top_n(c, top_n=20)
                st.write("Resume top keywords:", kwlists[0][:30])
                st.write("First job top keywords (first 3 jobs):")
                for i,k in enumerate(kwlists[1:4], start=1):
                    st.write(f"Job {i} kws:", k[:15])
            except Exception as e:
                st.write("Keyword extraction error:", e)

            # show reasoned selection: final compare (but not running full)
            st.info("Run the full agentic compare to produce ranked results in normal UI.")

        if out.get("error"):
            st.error(out.get("message"))
            return
        df = out["results"]
        resume_kws = out.get("resume_kws", [])
        st.subheader("Resume top keywords")
        st.write(resume_kws[:40])

        st.subheader(f"Top {len(df)} matching jobs (by final_score)")
        # show table with essential fields
        display_df = df[["job_id","title","company","final_score","semantic","keyword_overlap","recency","popularity","missing_skills"]]
        st.dataframe(display_df)

        # show details per job expanders
        for i, row in df.iterrows():
            with st.expander(f"{i+1}. {row['title']} — {row['company']} (score {row['final_score']})", expanded=False):
                st.write("Missing skills (suggested to learn):", row["missing_skills"])
                # simple course suggestion: search query (just a text suggestion)
                if row["missing_skills"]:
                    st.markdown("**Suggested learning searches (quick):**")
                    for skill in row["missing_skills"][:6]:
                        st.write(f"- Search: `beginner {skill} course` on Coursera/Udemy/YouTube")
                st.write("Raw job item:")
                st.json(row["raw_item"])

        # heatmap of components (small)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            comp_df = df[["semantic","keyword_overlap","recency","popularity"]].astype(float)
            if not comp_df.empty:
                fig, ax = plt.subplots(figsize=(8, max(2, 0.4*len(comp_df))))
                sns.heatmap(comp_df, annot=True, fmt=".2f", yticklabels=df["job_id"].values, cbar=True, ax=ax)
                st.pyplot(fig)
        except Exception:
            # seaborn not available — skip heatmap
            pass

# call UI function when this script is loaded in Streamlit
# remove guard if you integrate differently

run_agentic_ui()
    
