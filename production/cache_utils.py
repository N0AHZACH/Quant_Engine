import os
import joblib


def load_or_build(cache_path: str, builder_fn):
    """
    Generic disk cache helper.
    """
    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    obj = builder_fn()
    joblib.dump(obj, cache_path)
    return obj
