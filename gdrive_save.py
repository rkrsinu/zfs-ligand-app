import os
import io
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]

PIPELINE_FILES = [
    "ligand_donor_modes.csv",
    "seed_complexes.csv",
    "seed_ligands.csv",
    "mutated_ligands.csv",
    "mutation_lineage.csv",
    "generated_complexes.csv",
    "elite_parents.csv",
]

# ================= AUTH (PERMANENT) =================

creds = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=SCOPES,
)

service = build("drive", "v3", credentials=creds)

ROOT_FOLDER = st.secrets["GDRIVE_FOLDER_ID"]

# ================= FOLDER =================

def get_or_create_folder(name, parent):

    query = (
        f"name='{name}' and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"'{parent}' in parents and trashed=false"
    )

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


def get_target_folder(target, mode):
    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    return get_or_create_folder(str(int(target)), mode_folder)

# ================= DOWNLOAD =================

def download_pipeline_from_drive(target, mode):

    folder = get_target_folder(target, mode)
    restored = False

    for file in PIPELINE_FILES:

        query = f"name='{file}' and '{folder}' in parents and trashed=false"
        res = service.files().list(q=query, fields="files(id)").execute()

        if not res["files"]:
            continue

        restored = True

        request = service.files().get_media(fileId=res["files"][0]["id"])

        with open(file, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    return restored

# ================= UPLOAD (OVERWRITE MODE) =================

def upload_pipeline_to_drive(target, mode):

    folder = get_target_folder(target, mode)

    for file in PIPELINE_FILES:

        if not os.path.exists(file):
            continue

        query = f"name='{file}' and '{folder}' in parents and trashed=false"
        res = service.files().list(q=query, fields="files(id)").execute()

        media = MediaFileUpload(file, mimetype="text/csv", resumable=False)

        # 🔁 UPDATE existing file
        if res["files"]:
            file_id = res["files"][0]["id"]

            service.files().update(
                fileId=file_id,
                media_body=media,
            ).execute()

        # 🆕 CREATE if not exists
        else:
            service.files().create(
                body={"name": file, "parents": [folder]},
                media_body=media,
                fields="id",
            ).execute()
