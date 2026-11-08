"""Create a source-only ZIP without datasets, runs, caches or Python environments."""
from pathlib import Path
import argparse
import hashlib
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root.parent / 'pu-bagging-framework.zip')
    args = parser.parse_args()
    folders = ('src', 'examples', 'tests', 'docs', 'scripts', '.github')
    files = [root / name for name in ('pyproject.toml', '.gitignore', 'README.md', 'README_zh.md')]
    for folder in folders:
        files.extend(p for p in (root / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert not any('/data/' in n or '/outputs/' in n or '/.test-env/' in n or '/.venv/' in n for n in archive.namelist())
    print(f'Source files: {len(files)}')
    print(f'ZIP: {args.output}')
    print(f'SHA256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
