import io
import json
import pandas as pd
import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# =========================================================
# AUTHENTICATION
# =========================================================

SCOPES = ["https://www.googleapis.com/auth/drive"]

service_account_info = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

creds = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=SCOPES,
)

service = build("drive", "v3", credentials=creds)

# 🔍 DEBUG — WHO IS THE ACTIVE DRIVE USER
about = service.about().get(fields="user").execute()
print("DRIVE USER:", about)

ROOT_FOLDER_ID = st.secrets["GDRIVE_FOLDER_ID"]


# =========================================================
# HELPERS
# =========================================================

def get_or_create_folder(name, parent_id):
    query = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_id}' in parents and trashed=false"
    )

    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    files = results.get("files", [])

    if files:
        return files[0]["id"]

    folder_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }

    folder = service.files().create(
        body=folder_metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()

    return folder.get("id")


def upload_df(df, filename, folder_id):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    media = MediaIoBaseUpload(buffer, mimetype="text/csv", resumable=True)

    file_metadata = {
        "name": filename,
        "parents": [folder_id],
    }

    service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


# =========================================================
# PUBLIC FUNCTIONS
# =========================================================

def prepare_drive_folders(target_zfs, mode):
    try:
        mode_folder = get_or_create_folder(mode, ROOT_FOLDER_ID)
        target_folder = get_or_create_folder(str(target_zfs), mode_folder)
        return target_folder
    except Exception as e:
        st.warning(f"Drive folder creation failed: {e}")
        return None


def upload_generation_outputs(target_zfs, mode):
    try:
        folder_id = prepare_drive_folders(target_zfs, mode)
        if folder_id is None:
            return

        files = [
            "ligand_donor_modes.csv",
            "seed_complexes.csv",
            "seed_ligands.csv",
            "mutated_ligands.csv",
            "mutation_lineage.csv",
            "generated_complexes.csv",
        ]

        for f in files:
            try:
                df = pd.read_csv(f)
                upload_df(df, f, folder_id)
            except:
                pass

    except Exception as e:
        st.warning(f"Drive upload failed: {e}")


def upload_elite(target_zfs, mode):
    try:
        folder_id = prepare_drive_folders(target_zfs, mode)
        if folder_id is None:
            return

        df = pd.read_csv("elite_parents.csv")
        upload_df(df, "elite_parents.csv", folder_id)

    except Exception as e:
        st.warning(f"Elite upload failed: {e}")


def download_previous_state(target_zfs, mode):
    try:
        query = (
            f"name='{mode}' and '{ROOT_FOLDER_ID}' in parents and trashed=false"
        )

        results = service.files().list(
            q=query,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        if not results["files"]:
            return False

        mode_folder_id = results["files"][0]["id"]

        query = (
            f"name='{target_zfs}' and '{mode_folder_id}' in parents and trashed=false"
        )

        results = service.files().list(
            q=query,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        if not results["files"]:
            return False

        target_folder_id = results["files"][0]["id"]

        results = service.files().list(
            q=f"'{target_folder_id}' in parents and trashed=false",
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        for file in results.get("files", []):
            request = service.files().get_media(fileId=file["id"])
            fh = io.BytesIO()
            downloader = MediaIoBaseUpload(fh, mimetype="text/csv")
            data = request.execute()
            with open(file["name"], "wb") as f:
                f.write(data)

        return True

    except Exception as e:
        st.warning(f"Drive elite download skipped: {e}")
        return False
