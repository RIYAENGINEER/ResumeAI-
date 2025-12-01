import os
import json
import boto3
import tempfile
import traceback
import zipfile
import re
import urllib.parse
import time
from datetime import datetime, timezone
from PyPDF2 import PdfReader

# Config
DDB_TABLE = os.environ.get("DDB_TABLE", "resumes_meta")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "300000"))

s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DDB_TABLE)

def extract_text_from_pdf(path):
    text_parts = []
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_docx(path):
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml").decode("utf-8", errors="ignore")
            xml_content = xml_content.replace("</w:p>", "\n")
            cleaned = re.sub(r"<[^>]+>", "", xml_content)
            return cleaned
    except Exception:
        return ""

def extract_text_local(local_path, filename):
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(local_path)
    elif fname.endswith(".docx"):
        return extract_text_from_docx(local_path)
    elif fname.endswith(".txt"):
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError("unsupported_filetype")

def update_dynamo(user_id, resume_id, bucket, key, filename, status, extracted_text=None, error_msg=None):
    """
    Write an item ensuring it contains the required table keys:
      - user_id (Partition Key)
      - resume_id (Sort Key)
    """
    item = {
        "user_id": user_id,
        "resume_id": resume_id,
        "s3_bucket": bucket,
        "s3_key": key,
        "filename": filename,
        "status": status,
        "updated_ts": datetime.now(timezone.utc).astimezone().isoformat()
    }
    if extracted_text is not None:
        item["extracted_text"] = extracted_text[:MAX_TEXT_CHARS]
    if error_msg is not None:
        item["error"] = error_msg[:1000]

    # This is the put_item that previously raised ValidationException when keys were missing.
    table.put_item(Item=item)

def handler(event, context):
    print("Event:", json.dumps(event)[:2000])
    for rec in event.get("Records", []):
        try:
            s3_bucket = rec["s3"]["bucket"]["name"]
            raw_key = rec["s3"]["object"]["key"]
            s3_key = urllib.parse.unquote(raw_key)
            print("RAW_KEY:", raw_key, "DECODED:", s3_key)

            filename = os.path.basename(s3_key)

            # Infer user_id from key path like resumes/{user_id}/...
            parts = s3_key.split("/")
            user_id = "unknown_user"
            if len(parts) >= 2 and parts[0].lower() == "resumes":
                user_id = parts[1]
            elif len(parts) >= 1:
                user_id = parts[0]

            # Create a resume_id that matches your table's sort key (string)
            # Use ms timestamp to avoid collisions
            resume_id = str(int(time.time() * 1000))

            upload_ts = datetime.now(timezone.utc).astimezone().isoformat()

            # Download file to /tmp
            with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                local_path = tmpf.name
            print(f"Downloading s3://{s3_bucket}/{s3_key} -> {local_path}")
            try:
                s3.download_file(s3_bucket, s3_key, local_path)
            except Exception as e:
                print("Download failed:", e)
                update_dynamo(user_id, resume_id, s3_bucket, s3_key, filename, "error_download", error_msg=str(e))
                continue

              # Extract text
            try:
                extracted_text = extract_text_local(local_path, filename)
                status = "processed" if extracted_text else "processed_empty_text"
                print(f"Extraction OK, {len(extracted_text)} chars")
                update_dynamo(user_id, resume_id, s3_bucket, s3_key, filename, status, extracted_text=extracted_text)
                print("DynamoDB updated with extracted text.")
            except ValueError as ve:
                print("Unsupported file type:", ve)
                update_dynamo(user_id, resume_id, s3_bucket, s3_key, filename, "unsupported_filetype", error_msg=str(ve))
            except Exception as e:
                tb = traceback.format_exc()
                print("Extraction failed:", str(e), tb)
                update_dynamo(user_id, resume_id, s3_bucket, s3_key, filename, "error_extraction", error_msg=str(e) + " " + tb)
            finally:
                try:
                    os.remove(local_path)
                except Exception:
                    pass

        except Exception as e:
            print("Top-level processing error:", e, traceback.format_exc())

    return {"status": "done"}
