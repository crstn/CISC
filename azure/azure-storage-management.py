import os, uuid, sys
from azure.storage.blob import BlockBlobService, PublicAccess
# create a file credentials.py with the following content:
# account_name='your account name'
# account_key='your account key'
# find the account name and key in azure portal under
# storage account -> access keys
import credentials as cred

# ---------------------------------------------------------------------------------------------------------
# Method that creates a test file in the 'Documents' folder.
# This sample application creates a test file, uploads the test file to the Blob storage,
# lists the blobs in the container, and downloads the file with a new name.
# ---------------------------------------------------------------------------------------------------------
# Documentation References:
# Associated Article - https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python
# What is a Storage Account - http://azure.microsoft.com/en-us/documentation/articles/storage-whatis-account/
# Getting Started with Blobs-https://docs.microsoft.com/en-us/azure/storage/blobs/storage-python-how-to-use-blob-storage
# Blob Service Concepts - http://msdn.microsoft.com/en-us/library/dd179376.aspx
# Blob Service REST API - http://msdn.microsoft.com/en-us/library/dd135733.aspx
# ----------------------------------------------------------------------------------------------------------


def run_sample():
    try:
        # Create the BlockBlockService that is used to call the Blob service for the storage account
        block_blob_service = BlockBlobService(account_name=cred.account_name, account_key=cred.account_key)

        # Create a container called 'quickstartblobs'.
        container_name ='quickstartblobs'
        block_blob_service.create_container(container_name)

        # Set the permission so the blobs are public.
        block_blob_service.set_container_acl(container_name, public_access=PublicAccess.Container)

        # Create a file in Documents to test the upload and download.
        local_path=os.path.expanduser("~/Documents")
        local_file_name ="QuickStart_" + str(uuid.uuid4()) + ".txt"
        full_path_to_file =os.path.join(local_path, local_file_name)

        # Write text to the file.
        file = open(full_path_to_file,  'w')
        file.write("Hello, World!")
        file.close()

        print("Temp file = " + full_path_to_file)
        print("\nUploading to Blob storage as blob" + local_file_name)

        # Upload the created file, use local_file_name for the blob name
        block_blob_service.create_blob_from_path(container_name, local_file_name, full_path_to_file)

        # List the blobs in the container
        print("\nList blobs in the container")
        generator = block_blob_service.list_blobs(container_name)
        for blob in generator:
            print("\t Blob name: " + blob.name)

        # Download the blob(s).
        # Add '_DOWNLOADED' as prefix to '.txt' so you can see both files in Documents.
        full_path_to_file2 = os.path.join(local_path, str.replace(local_file_name ,'.txt', '_DOWNLOADED.txt'))
        print("\nDownloading blob to " + full_path_to_file2)
        block_blob_service.get_blob_to_path(container_name, local_file_name, full_path_to_file2)

        # Clean up resources. This includes the container and the temp files
        #block_blob_service.delete_container(container_name)
        #os.remove(full_path_to_file)
        #os.remove(full_path_to_file2)
    except Exception as e:
        print(e)


# Main method.
if __name__ == '__main__':
    run_sample()
