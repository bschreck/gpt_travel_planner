import pickle

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
