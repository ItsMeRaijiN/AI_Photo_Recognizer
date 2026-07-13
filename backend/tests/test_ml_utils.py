from __future__ import annotations

import hashlib

from backend.ml.utils import HashCache, save_confusion_matrix


def test_hash_cache_invalidates_entry_when_file_changes(tmp_path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"first payload")

    cache = HashCache(str(tmp_path / "hash_cache.pkl"))
    first_hash = cache.get_md5(str(source))

    source.write_bytes(b"changed payload with a different size")
    second_hash = cache.get_md5(str(source))

    assert first_hash != second_hash
    assert second_hash == hashlib.md5(source.read_bytes()).hexdigest()


def test_confusion_matrix_plot_accepts_serializable_list(tmp_path) -> None:
    output_dir = tmp_path / "plots"

    save_confusion_matrix([[3, 1], [0, 4]], str(output_dir), threshold=0.5)

    assert (output_dir / "confusion_matrix.png").is_file()
