import io
import os
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVICE_ACCOUNT_FILE = os.path.join(
    BASE_DIR,
    "zfs-ligand-app-ae10bd1536bb.json"
)

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
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


# ---------------- EXISTING RESULT SAVER ----------------

def save_results_to_drive(target, mode):

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    for f in [
        "elite_parents.csv",
        "generated_complexes.csv",
        "mutation_lineage.csv",
    ]:
        if os.path.exists(f):

            media = MediaIoBaseUpload(
                io.FileIO(f, "rb"),
                mimetype="text/csv",
                resumable=True,
            )

            service.files().create(
                body={"name": f, "parents": [target_folder]},
                media_body=media,
                fields="id",
            ).execute()


# ---------------- NEW: UPLOAD ELITE EACH GENERATION ----------------

def upload_elite_to_drive(target, mode):

    if not os.path.exists("elite_parents.csv"):
        return

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    media = MediaIoBaseUpload(
        io.FileIO("elite_parents.csv", "rb"),
        mimetype="text/csv",
        resumable=True,
    )

    query = f"name='elite_parents.csv' and '{target_folder}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()

    if res["files"]:
        service.files().update(
            fileId=res["files"][0]["id"],
            media_body=media
        ).execute()
    else:
        service.files().create(
            body={"name": "elite_parents.csv", "parents": [target_folder]},
            media_body=media,
            fields="id",
        ).execute()


# ---------------- NEW: DOWNLOAD ELITE BEFORE GA ----------------

def download_elite_from_drive(target, mode):

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    query = f"name='elite_parents.csv' and '{target_folder}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()

    if not res["files"]:
        return False

    file_id = res["files"][0]["id"]

    request = service.files().get_media(fileId=file_id)

    fh = io.FileIO("elite_parents.csv", "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return True
