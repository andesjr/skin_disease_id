from gcloud import storage

storage_client = storage.Client()

source_blob_name =
buckets = list(storage_client.list_buckets())
print(buckets)


# give blob credentials
source_blob_name= 'DF_train_final_60.csv'
#destination_file_name = 'downloaded_testing.txt'
bucket_name = 'bucket_skin_disease/dataset_skin'
# get bucket object


try:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('file: ',destination_file_name,' downloaded from bucket: ',bucket_name,' successfully')
except Exception as e:
    print(e)
