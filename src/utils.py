import pickle
from google.cloud import storage
from google.oauth2 import service_account
import time


def upload_file_to_gcs(local_file_path, remote_file_path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(remote_file_path).upload_from_filename(local_file_path)


def download_file_from_gcs(remote_file_path, local_file_path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(remote_file_path).download_to_filename(local_file_path)


def persist_to_file(file_name):
    def persist_to_file_decorator(original_func):
        try:
            cache = pickle.load(open(file_name, "rb"))
        except (IOError, ValueError):
            cache = {}

        def new_func(*args, **kwargs):
            key = (args, tuple(kwargs.keys()))
            if key not in cache:
                cache[key] = original_func(*args, **kwargs)
                pickle.dump(cache, open(file_name, "wb"))
            return cache[key]

        return new_func

    return persist_to_file_decorator


def cache_with_ttl(ttl: int = 60 * 60 * 24):
    def cache_with_ttl_decorator(original_func):
        if not hasattr(cache_with_ttl_decorator, "cache"):
            cache_with_ttl_decorator.cache = {}

        def new_func(*args, **kwargs):
            name = original_func.__name__
            cache_entry = cache_with_ttl_decorator.cache.get(name, {})
            last_updated = cache_entry.get("last_updated", None)
            current_time = time.time()
            if not cache_entry.get("result") or (
                last_updated and current_time - last_updated > ttl
            ):
                cache_entry["last_updated"] = current_time
                res = original_func(*args, **kwargs)
                cache_entry["result"] = res
                cache_with_ttl_decorator.cache[name] = cache_entry
            return cache_entry["result"]

        return new_func

    return cache_with_ttl_decorator
