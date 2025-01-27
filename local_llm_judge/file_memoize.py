import os
import json
import hashlib

class FileMemoizer:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, func_name, args, kwargs):
        """Generate a unique file path for the given function and its arguments."""
        key = f"{func_name}:{args}:{kwargs}"
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{func_name}_{hashed_key}.json")

    def memoize(self, func):
        def wrapper(*args, **kwargs):
            cache_path = self._get_cache_path(func.__name__, args, kwargs)
            if os.path.exists(cache_path):
                with open(cache_path, "r") as file:
                    print(f"Loading cached result for {func.__name__} with args {args} and kwargs {kwargs}")
                    return json.load(file)

            result = func(*args, **kwargs)
            with open(cache_path, "w") as file:
                json.dump(result, file)
                print(f"Saving result for {func.__name__} with args {args} and kwargs {kwargs}")
            return result
        return wrapper

# Example usage
memoizer = FileMemoizer()
