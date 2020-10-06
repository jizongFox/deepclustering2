import subprocess

from deepclustering2.utils import path2Path, path2str


def gethash(file_path):
    file_path = path2Path(file_path)
    if file_path.is_file():
        file_path = file_path.parent
    try:
        __git_hash__ = (
            subprocess.check_output(
                [f"cd {path2str(file_path)}; git rev-parse HEAD"], shell=True
            )
            .strip()
            .decode()
        )
    except Exception:
        __git_hash__ = None
    return __git_hash__
