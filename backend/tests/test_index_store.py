"""
Tests for storage/index_store.py — the only module that touches disk.

Each test monkeypatches the module's own path constants (which were bound
from config.* at import time) to point at pytest's tmp_path, so tests
never touch the real dinov2_faiss_store/ directory and can run in
parallel / in any order.
"""

import faiss
import numpy as np
import pytest

from storage import index_store
from storage.index_store import IndexData


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirect index_store's file paths into a throwaway tmp_path."""
    store_dir = str(tmp_path / "store")
    monkeypatch.setattr(index_store, "STORE_DIR", store_dir)
    monkeypatch.setattr(index_store, "INDEX_FILE", store_dir + "/dinov2.index")
    monkeypatch.setattr(index_store, "VECTORS_FILE", store_dir + "/image_vectors.npy")
    monkeypatch.setattr(index_store, "PATHS_FILE", store_dir + "/image_paths.npy")
    monkeypatch.setattr(index_store, "HASHES_FILE", store_dir + "/image_hashes.npy")
    return store_dir


def _vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((1, dim)).astype("float32")
    return v / np.linalg.norm(v)


def _empty_index_data(dim: int = 8) -> IndexData:
    """
    Build an empty IndexData at a given dimension.

    _empty_index_data() hardcodes config.DINO_EMBEDDING_DIM (768) —
    correct for the real application, but these tests use small 8-dim
    vectors for speed, so we build our own empty index at the matching
    dimension instead of relying on load_empty()'s fixed size.
    """
    return IndexData(
        vectors=np.empty((0, dim), dtype="float32"),
        paths=np.array([], dtype=object),
        hashes=np.array([], dtype=object),
        index=faiss.IndexFlatIP(dim),
    )


class TestIsInitialised:
    def test_false_before_any_save(self, isolated_store):
        assert index_store.is_initialised() is False

    def test_true_after_save(self, isolated_store):
        data = _empty_index_data()
        index_store.save(data)
        assert index_store.is_initialised() is True


class TestSaveAndLoad:
    def test_round_trip_preserves_data(self, isolated_store):
        vectors = np.vstack([_vector(1), _vector(2)])
        paths = np.array(["a.jpg", "b.jpg"], dtype=object)
        hashes = np.array(["hash_a", "hash_b"], dtype=object)
        data = _empty_index_data()
        data.index.add(vectors)

        index_store.save(IndexData(vectors=vectors, paths=paths, hashes=hashes, index=data.index))
        loaded = index_store.load()

        assert loaded.paths.tolist() == ["a.jpg", "b.jpg"]
        assert loaded.hashes.tolist() == ["hash_a", "hash_b"]
        assert loaded.index.ntotal == 2

    def test_load_raises_when_not_initialised(self, isolated_store):
        with pytest.raises(FileNotFoundError):
            index_store.load()


class TestAppendOne:
    def test_appends_to_existing_index(self, isolated_store):
        index_store.save(_empty_index_data())
        vector = _vector(1)
        added = index_store.append_one("new.jpg", vector, "hash_new")

        assert added is True
        loaded = index_store.load()
        assert "new.jpg" in loaded.paths.tolist()
        assert loaded.index.ntotal == 1

    def test_rejects_duplicate_path(self, isolated_store):
        index_store.save(_empty_index_data())
        vector = _vector(1)
        index_store.append_one("dup.jpg", vector, "hash_dup")
        added_again = index_store.append_one("dup.jpg", vector, "hash_dup")
        assert added_again is False

    def test_returns_false_when_index_missing(self, isolated_store):
        # No save() called yet — index doesn't exist on disk.
        added = index_store.append_one("x.jpg", _vector(1), "hash_x")
        assert added is False


class TestRemoveOne:
    def test_removes_matching_basename(self, isolated_store):
        vectors = np.vstack([_vector(1), _vector(2)])
        paths = np.array(["/pool/keep.jpg", "/pool/remove.jpg"], dtype=object)
        hashes = np.array(["h1", "h2"], dtype=object)
        data = _empty_index_data()
        data.index.add(vectors)
        index_store.save(IndexData(vectors=vectors, paths=paths, hashes=hashes, index=data.index))

        removed = index_store.remove_one("remove.jpg")

        assert removed is True
        loaded = index_store.load()
        assert loaded.paths.tolist() == ["/pool/keep.jpg"]
        assert loaded.index.ntotal == 1

    def test_returns_false_for_unknown_filename(self, isolated_store):
        index_store.save(_empty_index_data())
        assert index_store.remove_one("ghost.jpg") is False


class TestClear:
    def test_clear_removes_all_store_files(self, isolated_store):
        index_store.save(_empty_index_data())
        assert index_store.is_initialised() is True
        index_store.clear()
        assert index_store.is_initialised() is False
