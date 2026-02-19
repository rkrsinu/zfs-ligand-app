import io
import os
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]
ROOT_FOLDER = st.secrets["GDRIVE_FOLDER_ID"]

creds = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=SCOPES,
)

service = build("drive", "v3", credentials=creds)


def get_or_create_folder(name, parent):

    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent}' in parents and trashed=false"

    res = service.files().list(q=query, fields="files(id)").execute()

    if res["files"]:
        return res["files"][0]["id"]

    folder = service.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]},
        fields="id"
    ).execute()

    return folder["id"]


def upload_elite_to_drive(target, mode):

    if not os.path.exists("elite_parents.csv"):
        return

    mode_folder = get_or_create_folder(mode, ROOT_FOLDER)
    target_folder = get_or_create_folder(str(int(target)), mode_folder)

    query = f"name='elite_parents.csv' and '{target_folder}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()

    for f in res.get("files", []):
        service.files().delete(fileId=f["id"]).execute()

    media = MediaIoBaseUpload(
        io.BytesIO(open("elite_parents.csv", "rb").read()),
        mimetype="text/csv"
    )

    service.files().create(
        body={"name": "elite_parents.csv", "parents": [target_folder]},
        media_body=media,
        fields="id"
    ).execute()


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
