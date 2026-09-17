import os
import time
import streamlit as st
from google.oauth2.credentials import Credentials
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

# ================= AUTH =================

creds = Credentials(
    None,
    refresh_token=st.secrets["GDRIVE_REFRESH_TOKEN"],
    token_uri="https://oauth2.googleapis.com/token",
    client_id=st.secrets["GDRIVE_CLIENT_ID"],
    client_secret=st.secrets["GDRIVE_CLIENT_SECRET"],
    scopes=SCOPES,
)

ROOT_FOLDER = st.secrets["GDRIVE_FOLDER_ID"]


def _build_service():
    # A fresh Drive client is used after a broken/stale HTTPS connection.
    return build("drive", "v3", credentials=creds, cache_discovery=False)


service = _build_service()


# ================= ROBUST DRIVE REQUEST =================

def _execute_drive(operation, retries=3):
    """Execute a Drive request and retry transient HTTPS/connection failures."""
    global service

    last_error = None

    for attempt in range(retries):
        try:
            return operation(service)
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            TimeoutError,
        ) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                service = _build_service()
        except OSError as exc:
            # httplib2 can surface SSL/socket failures as OSError.
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                service = _build_service()
            else:
                raise

    raise last_error


# ================= FOLDER =================

def get_or_create_folder(name, parent):

    query = (
        f"name='{name}' and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"'{parent}' in parents and trashed=false"
    )

    res = _execute_drive(
        lambda s: s.files().list(q=query, fields="files(id)").execute()
    )

    files = res.get("files", [])

    if files:
        return files[0]["id"]

    folder = _execute_drive(
        lambda s: s.files().create(
            body={
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent],
            },
            fields="id",
        ).execute()
    )

    return folder["id"]


def get_target_folder(target, mode):
    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    return get_or_create_folder(str(int(target)), mode_folder)


# ================= DOWNLOAD =================

def download_pipeline_from_drive(target, mode):

    global service

    folder = get_target_folder(target, mode)
    restored = False

    for file in PIPELINE_FILES:

        query = f"name='{file}' and '{folder}' in parents and trashed=false"
        res = _execute_drive(
            lambda s: s.files().list(q=query, fields="files(id)").execute()
        )

        files = res.get("files", [])

        if not files:
            continue

        restored = True

        request = service.files().get_media(fileId=files[0]["id"])

        with open(file, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                try:
                    _, done = downloader.next_chunk()
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    ConnectionAbortedError,
                    TimeoutError,
                    OSError,
                ):
                    # Re-create the media request if the HTTPS connection drops.
                    service = _build_service()
                    request = service.files().get_media(fileId=files[0]["id"])
                    downloader = MediaIoBaseDownload(fh, request)

    return restored


# ================= UPLOAD (OVERWRITE MODE) =================

def upload_pipeline_to_drive(target, mode):

    folder = get_target_folder(target, mode)

    for file in PIPELINE_FILES:

        if not os.path.exists(file):
            continue

        query = f"name='{file}' and '{folder}' in parents and trashed=false"
        res = _execute_drive(
            lambda s: s.files().list(q=query, fields="files(id)").execute()
        )

        files = res.get("files", [])
        media = MediaFileUpload(file, mimetype="text/csv", resumable=False)

        # UPDATE existing file
        if files:
            file_id = files[0]["id"]

            _execute_drive(
                lambda s: s.files().update(
                    fileId=file_id,
                    media_body=media,
                ).execute()
            )

        # CREATE if not exists
        else:
            _execute_drive(
                lambda s: s.files().create(
                    body={"name": file, "parents": [folder]},
                    media_body=media,
                    fields="id",
                ).execute()
            )
