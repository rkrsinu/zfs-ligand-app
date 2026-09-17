# ==========================================================
# gdrive_save.py
#
# Google Drive persistence for resumable GA campaigns.
# Only the files required to resume are synchronized by default.
# Authentication is lazy so the GA worker can run without importing
# Streamlit secrets at module import time.
# ==========================================================

import os

RESUME_FILES = [
    "ga_checkpoint.csv",
    "elite_parents.csv",
]

OPTIONAL_FILES = [
    "mutation_lineage.csv",
]


def _get_secrets():
    # Streamlit secrets are available in the deployed app. Environment
    # variables make the same module usable from a local/CLI worker.
    token = os.environ.get("GDRIVE_REFRESH_TOKEN")
    client_id = os.environ.get("GDRIVE_CLIENT_ID")
    client_secret = os.environ.get("GDRIVE_CLIENT_SECRET")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    if token and client_id and client_secret and folder_id:
        return token, client_id, client_secret, folder_id

    try:
        import streamlit as st
        token = st.secrets["GDRIVE_REFRESH_TOKEN"]
        client_id = st.secrets["GDRIVE_CLIENT_ID"]
        client_secret = st.secrets["GDRIVE_CLIENT_SECRET"]
        folder_id = st.secrets["GDRIVE_FOLDER_ID"]
        return token, client_id, client_secret, folder_id
    except Exception as exc:
        raise RuntimeError(
            "Google Drive credentials are not available. Set GDRIVE_REFRESH_TOKEN, "
            "GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET and GDRIVE_FOLDER_ID, or configure "
            "the corresponding Streamlit secrets."
        ) from exc


def _build_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token, client_id, client_secret, _ = _get_secrets()
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = Credentials(
        None,
        refresh_token=token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
    )
    return build("drive", "v3", credentials=creds)


def _root_folder_id():
    return _get_secrets()[3]


def get_or_create_folder(service, name, parent):
    safe_name = str(name).replace("'", "\\'")
    query = (
        f"name='{safe_name}' and "
        "mimeType='application/vnd.google-apps.folder' and "
        f"'{parent}' in parents and trashed=false"
    )
    res = service.files().list(q=query, fields="files(id)", pageSize=10).execute()
    if res.get("files"):
        return res["files"][0]["id"]

    folder = service.files().create(
        body={
            "name": str(name),
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent],
        },
        fields="id",
    ).execute()
    return folder["id"]


def get_target_folder(service, target, mode):
    root = _root_folder_id()
    mode_folder = get_or_create_folder(service, str(mode), root)
    target_name = str(int(float(target))) if float(target).is_integer() else str(target)
    return get_or_create_folder(service, target_name, mode_folder)


def _find_file(service, name, folder):
    safe_name = str(name).replace("'", "\\'")
    query = f"name='{safe_name}' and '{folder}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id,name)", pageSize=10).execute()
    return res.get("files", [None])[0]


def download_pipeline_from_drive(target, mode, include_optional=False):
    """Restore the minimum state needed to resume a campaign.

    Returns True when at least one state file was restored.
    """
    service = _build_service()
    folder = get_target_folder(service, target, mode)
    files = RESUME_FILES + (OPTIONAL_FILES if include_optional else [])
    restored = False

    from googleapiclient.http import MediaIoBaseDownload

    for name in files:
        found = _find_file(service, name, folder)
        if not found:
            continue
        request = service.files().get_media(fileId=found["id"])
        with open(name, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        restored = True

    return restored


def upload_pipeline_to_drive(target, mode, include_optional=False):
    """Upload only resumable state instead of all generation CSVs."""
    service = _build_service()
    folder = get_target_folder(service, target, mode)
    files = RESUME_FILES + (OPTIONAL_FILES if include_optional else [])

    from googleapiclient.http import MediaFileUpload

    for name in files:
        if not os.path.exists(name):
            continue
        found = _find_file(service, name, folder)
        media = MediaFileUpload(name, mimetype="text/csv", resumable=False)
        if found:
            service.files().update(fileId=found["id"], media_body=media).execute()
        else:
            service.files().create(
                body={"name": name, "parents": [folder]},
                media_body=media,
                fields="id",
            ).execute()
