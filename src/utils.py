import pickle
from google.cloud import storage
from google.oauth2 import service_account


def upload_file_to_gcs(local_file_path, remote_file_path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(remote_file_path).upload_from_filename(local_file_path)

def download_file_from_gcs(remote_file_path, local_file_path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(remote_file_path).download_to_filename(local_file_path)


def persist_to_file(file_name):

    def decorator(original_func):

        try:
            cache = pickle.load(open(file_name, 'rb'))
        except (IOError, ValueError):
            cache = {}

        def new_func(*args, **kwargs):
            key = (args, tuple(kwargs.keys()))
            if key not in cache:
                cache[key] = original_func(*args, **kwargs)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache[key]

        return new_func

    return decorator

