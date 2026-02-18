import io
import os
import json
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

service_account_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])

creds = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=SCOPES,
)

service = build("drive", "v3", credentials=creds)

ROOT_FOLDER = st.secrets["GDRIVE_FOLDER_ID"]


def get_or_create_folder(name, parent):

    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()

    if res["files"]:
        return res["files"][0]["id"]

    folder = service.files().create(
        body={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent],
        },
        fields="id",
    ).execute()

    return folder["id"]


def upload_file(local_path, drive_folder):

    file_name = os.path.basename(local_path)

    media = MediaIoBaseUpload(
        io.FileIO(local_path, "rb"),
        mimetype="text/csv",
        resumable=True,
    )

    service.files().create(
        body={"name": file_name, "parents": [drive_folder]},
        media_body=media,
        fields="id",
    ).execute()


def save_results_to_drive(target, mode):

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    for f in [
        "elite_parents.csv",
        "generated_complexes.csv",
        "mutation_lineage.csv",
    ]:
        if os.path.exists(f):
            upload_file(f, target_folder)
