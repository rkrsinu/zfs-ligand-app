import io
import os
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ==============================
# CONFIG
# ==============================

SCOPES = ["https://www.googleapis.com/auth/drive"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVICE_ACCOUNT_FILE = os.path.join(
    BASE_DIR,
    "zfs-ligand-app-ae10bd1536bb.json"
)

ROOT_FOLDER = st.secrets["GDRIVE_FOLDER_ID"]

# ==============================
# AUTH
# ==============================

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES,
)

service = build("drive", "v3", credentials=creds)


# ==============================
# FOLDER HANDLING
# ==============================

def get_or_create_folder(name, parent):

    query = (
        f"name='{name}' and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"'{parent}' in parents and trashed=false"
    )

    res = service.files().list(
        q=query,
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

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


# ==============================
# SAVE FINAL RESULTS
# ==============================

def save_results_to_drive(target, mode):

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    for f in [
        "elite_parents.csv",
        "generated_complexes.csv",
        "mutation_lineage.csv",
    ]:
        if os.path.exists(f):

            with open(f, "rb") as file_data:

                media = MediaIoBaseUpload(
                    io.BytesIO(file_data.read()),
                    mimetype="text/csv",
                    resumable=False,
                )

            service.files().create(
                body={"name": f, "parents": [target_folder]},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()


# ==============================
# UPLOAD ELITE EACH GENERATION
# ==============================

def upload_elite_to_drive(target, mode):

    file_path = "elite_parents.csv"

    if not os.path.exists(file_path):
        return

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    file_name = "elite_parents.csv"

    # delete old elite if exists
    query = (
        f"name='{file_name}' and "
        f"'{target_folder}' in parents and trashed=false"
    )

    res = service.files().list(
        q=query,
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    for f in res.get("files", []):
        service.files().delete(
            fileId=f["id"],
            supportsAllDrives=True
        ).execute()

    # upload new elite
    with open(file_path, "rb") as file_data:

        media = MediaIoBaseUpload(
            io.BytesIO(file_data.read()),
            mimetype="text/csv",
            resumable=False,
        )

    service.files().create(
        body={
            "name": file_name,
            "parents": [target_folder],
        },
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


# ==============================
# DOWNLOAD ELITE FOR NEXT RUN
# ==============================

def download_elite_from_drive(target, mode):

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    query = (
        f"name='elite_parents.csv' and "
        f"'{target_folder}' in parents and trashed=false"
    )

    res = service.files().list(
        q=query,
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    if not res["files"]:
        return False

    file_id = res["files"][0]["id"]

    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True
    )

    fh = io.FileIO("elite_parents.csv", "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return True
