#!/bin/bash
# Write Google auth files from base64 env vars
mkdir -p data/resume

python -c "
import base64, os
creds = os.getenv('GOOGLE_CREDENTIALS_B64','')
token = os.getenv('GOOGLE_TOKEN_B64','')
resume = os.getenv('RESUME_PDF_B64','')
if creds: open('data/google_credentials.json','wb').write(base64.b64decode(creds))
if token: open('data/google_token.json','wb').write(base64.b64decode(token))
if resume: open('data/resume/resume.pdf','wb').write(base64.b64decode(resume))
print('Files written from env vars')
"

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port $PORT