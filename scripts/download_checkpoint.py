"""下载或导入推理附件，校验 SHA-256，只提取清单中的文件。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="import an already downloaded ZIP")
    parser.add_argument("--repo", help="GitHub owner/repository; overrides the release manifest")
    parser.add_argument("--output", type=Path, default=PROJECT / "weights")
    args = parser.parse_args()
    manifest = json.loads((PROJECT / "results/checkpoint_manifest.json").read_text())
    repository = args.repo or manifest["repository"]
    destination = args.output / "act_push"
    if destination.exists():
        raise FileExistsError(f"Already exists: {destination}")
    archive = args.archive
    if archive is None:
        directory = PROJECT / "artifacts"
        directory.mkdir(exist_ok=True)
        archive = directory / "act_push_checkpoint.zip"
        if not archive.exists():
            if shutil.which("gh"):
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        manifest["release"],
                        "--repo",
                        repository,
                        "--pattern",
                        archive.name,
                        "--dir",
                        str(directory),
                    ],
                    check=True,
                )
            else:
                url = f"https://github.com/{repository}/releases/download/{manifest['release']}/{archive.name}"
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    archive.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output)
    if digest(archive) != manifest["sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch; refusing to extract")
    expected = {f"act_push/{name}" for name in manifest["files"]}
    with zipfile.ZipFile(archive) as bundle:
        actual = {item.filename for item in bundle.infolist()}
        if actual != expected or len(bundle.infolist()) != len(expected):
            raise ValueError("Unexpected archive members")
        if any(Path(name).name != name for name in manifest["files"]):
            raise ValueError("Manifest filenames must be basenames")
        # 不使用 extractall；仅按已验证的文件名写入新目录。
        destination.mkdir(parents=True, exist_ok=False)
        for name in manifest["files"]:
            with (
                bundle.open(f"act_push/{name}") as source,
                (destination / name).open("wb") as output,
            ):
                shutil.copyfileobj(source, output)
    print(f"Verified checkpoint step {manifest['checkpoint_step']}: {destination}")


if __name__ == "__main__":
    main()
