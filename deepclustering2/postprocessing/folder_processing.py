import argparse
import re
import shutil
from itertools import repeat
from typing import Optional, List

from deepclustering2.utils import path2Path


def none_or_str(value):
    if value.lower() == "none":
        return None
    return value


def get_args():
    parser = argparse.ArgumentParser(
        description="Parser for folder extractor",  # noqa
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # noqa
    )  # noqa
    folder_parse = parser.add_argument_group(title="IO")
    folder_parse.add_argument(
        "-r", "--root", required=True, type=str, help="folder root"
    )
    folder_parse.add_argument(
        "-o", "--out", required=True, type=str, help="output folder path"
    )
    folder_parse.add_argument(
        "-e",
        "--exist_ok",
        default=False,
        action="store_true",
        help="output folder path",
    )
    content_paraser = parser.add_argument_group(title="Content")
    content_paraser.add_argument(
        "--content_type", required=True, type=str, nargs="+", help="content type"
    )
    content_paraser.add_argument(
        "--content_regex",
        type=none_or_str,
        default=None,
        nargs="+",
        help="content filter regex",
    )

    args = parser.parse_args()
    if args.content_regex is not None:
        assert len(args.content_regex) == len(args.content_type)
    else:
        args.content_regex = [None] * len(args.content_type)
    return args


def file_searcher(root, content_types: List[str], content_regex: List[Optional[str]]):
    assert len(content_types) == len(content_regex), (content_types, content_regex)
    root = path2Path(root)
    assert root.exists() and root.is_dir(), root

    def typed_file_generator():
        for content_type in content_types:
            _content_type = content_type
            if content_type[0] == ".":
                _content_type = content_type[1:]
            files = root.rglob(f"*/*.{_content_type}")
            yield from zip(files, repeat(content_type))

    def find_regex_match(context_regex, file_path):
        syntex = re.compile(context_regex).search(str(file_path))
        if syntex is not None:
            return True
        return False

    typed_files = typed_file_generator()

    for (path, ctype) in typed_files:
        if content_regex[content_types.index(ctype)] is not None:
            regex = content_regex[content_types.index(ctype)]
            if find_regex_match(regex, path):
                yield path
        else:
            yield path


def file_copier(file_path: str, root_dir: str, save_dir: str, exist_ok=False):
    file_path = path2Path(file_path)
    assert file_path.is_file(), file_path
    relative_path = file_path.relative_to(root_dir)

    out_path = path2Path(save_dir) / relative_path
    if out_path.exists():
        if not exist_ok:
            raise FileExistsError(str(out_path))
    try:
        shutil.copyfile(str(file_path), str(out_path))
    except FileNotFoundError:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(file_path), str(out_path))


def main():
    args = get_args()
    file_generator = file_searcher(
        root=args.root,
        content_types=args.content_type,
        content_regex=args.content_regex,
    )
    for file in file_generator:
        file_copier(file, args.root, save_dir=args.out, exist_ok=args.exist_ok)


if __name__ == "__main__":
    main()
