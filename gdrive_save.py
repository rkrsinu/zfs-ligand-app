import os
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# ---------------------------------------------------------
# OAuth credentials
# ---------------------------------------------------------

creds = Credentials(
    None,
    refresh_token=st.secrets["GDRIVE_REFRESH_TOKEN"],
    token_uri="https://oauth2.googleapis.com/token",
    client_id=st.secrets["GDRIVE_CLIENT_ID"],
    client_secret=st.secrets["GDRIVE_CLIENT_SECRET"],
    scopes=SCOPES,
)

service = build("drive", "v3", credentials=creds)

ROOT_FOLDER = st.secrets["GDRIVE_FOLDER_ID"]

PIPELINE_FILES = [
    "elite_parents.csv",
    "generated_complexes.csv",
    "ligand_donor_modes.csv",
    "mutated_ligands.csv",
    "mutation_lineage.csv",
]

# ---------------------------------------------------------
# Create / get folder
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# Upload pipeline state
# ---------------------------------------------------------

def upload_pipeline_to_drive(target, mode):

    # create only the selected mode folder
    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)

    # create target folder inside mode
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    for file in PIPELINE_FILES:

        if not os.path.exists(file):
            continue

        query = f"name='{file}' and '{target_folder}' in parents and trashed=false"

        res = service.files().list(
            q=query,
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        media = MediaFileUpload(file, mimetype="text/csv", resumable=False)

        # update existing
        if res["files"]:

            file_id = res["files"][0]["id"]

            service.files().update(
                fileId=file_id,
                media_body=media,
                supportsAllDrives=True,
            ).execute()

        # create new
        else:

            service.files().create(
                body={"name": file, "parents": [target_folder]},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
