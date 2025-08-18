import os


def mkdir_p(path: str) -> None:
    """Create directory path if non-empty, ignoring if it already exists."""
    if not path:
        return
    os.makedirs(path, exist_ok=True) 