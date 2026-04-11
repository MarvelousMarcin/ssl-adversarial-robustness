from pathlib import Path
import pickle

class Cache:
    def __init__(self, path: str = "cache.pkl"):
        self.path = Path(path)

    def exists(self):
        return self.path.exists()

    def save(self, **data):
        with open(self.path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)

        return data