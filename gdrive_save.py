import streamlit as st
import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ================== AUTH ==================
SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=SCOPES,
)

service = build("drive", "v3", credentials=creds)

about = service.about().get(fields="user").execute()
print("DRIVE USER:", about)


ROOT_FOLDER_ID = st.secrets["GDRIVE_FOLDER_ID"]

# ================== HELPERS ==================
def _get_or_create_folder(name, parent):
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()

    if res["files"]:
        return res["files"][0]["id"]

    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent],
    }

    folder = service.files().create(
        body=file_metadata,
        supportsAllDrives=True,
        fields="id",
    ).execute()

    return folder["id"]


def _upload_file(local_path, drive_folder_id):
    if not os.path.exists(local_path):
        return

    file_name = os.path.basename(local_path)

    media = MediaIoBaseUpload(open(local_path, "rb"), resumable=True)

    service.files().create(
        body={"name": file_name, "parents": [drive_folder_id]},
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


# ================== PUBLIC API ==================
def prepare_drive_folders(target_zfs, mode):
    target_folder = _get_or_create_folder(f"{mode}_{target_zfs}", ROOT_FOLDER_ID)
    return target_folder


def upload_generation_outputs(target_zfs, mode):
    folder = prepare_drive_folders(target_zfs, mode)

    files = [
        "ligand_donor_modes.csv",
        "seed_complexes.csv",
        "seed_ligands.csv",
        "mutated_ligands.csv",
        "mutation_lineage.csv",
        "generated_complexes.csv",
    ]

    for f in files:
        _upload_file(f, folder)


def upload_elite(target_zfs, mode):
    folder = prepare_drive_folders(target_zfs, mode)
    _upload_file("elite_parents.csv", folder)


def download_previous_state(target_zfs, mode):
    folder_name = f"{mode}_{target_zfs}"

    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{ROOT_FOLDER_ID}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()

    if not res["files"]:
        return False

    folder_id = res["files"][0]["id"]

    query = f"'{'elite_parents.csv'}' in name and '{folder_id}' in parents and trashed=false"
    files = service.files().list(q=query, fields="files(id, name)").execute()

    if not files["files"]:
        return False

    file_id = files["files"][0]["id"]

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO("elite_parents.csv", "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while done is False:
        _, done = downloader.next_chunk()

    return True
