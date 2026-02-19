import io
import os
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

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

creds = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"], scopes=SCOPES
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
        supportsAllDrives=True,
    ).execute()

    return folder["id"]


def download_pipeline_from_drive(target, mode):

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    found = False

    for file in PIPELINE_FILES:

        query = f"name='{file}' and '{target_folder}' in parents and trashed=false"
        res = service.files().list(
            q=query,
            fields="files(id)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        if not res["files"]:
            continue

        found = True

        file_id = res["files"][0]["id"]

        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file, "wb")

        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

    return found


def upload_pipeline_to_drive(target, mode):

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    for file in PIPELINE_FILES:

        if not os.path.exists(file):
            continue

        query = f"name='{file}' and '{target_folder}' in parents and trashed=false"
        res = service.files().list(
            q=query,
            fields="files(id)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        for f in res.get("files", []):
            service.files().delete(
                fileId=f["id"],
                supportsAllDrives=True
            ).execute()

        media = MediaIoBaseUpload(
            io.BytesIO(open(file, "rb").read()),
            mimetype="text/csv",
        )

        service.files().create(
            body={"name": file, "parents": [target_folder]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
