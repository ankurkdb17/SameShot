"""
Tests for security.safe_filename and security.has_allowed_extension.

These guard the path-traversal fix in routes/images.py — every test here
maps directly to an attack vector that was previously possible when
file.filename was used unsanitised in os.path.join().
"""

from security import has_allowed_extension, safe_filename


class TestSafeFilename:
    def test_plain_filename_is_unchanged(self):
        assert safe_filename("photo.jpg") == "photo.jpg"

    def test_strips_unix_directory_traversal(self):
        result = safe_filename("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result

    def test_strips_windows_directory_traversal(self):
        result = safe_filename("..\\..\\Windows\\System32\\evil.exe")
        assert "\\" not in result
        assert ".." not in result

    def test_strips_absolute_unix_path(self):
        result = safe_filename("/etc/passwd")
        assert result == "passwd"

    def test_rejects_dot_and_dotdot(self):
        assert safe_filename(".").startswith("upload_")
        assert safe_filename("..").startswith("upload_")

    def test_rejects_empty_string(self):
        assert safe_filename("").startswith("upload_")

    def test_strips_null_byte_and_shell_chars(self):
        result = safe_filename("photo.jpg\x00; rm -rf ~")
        assert "\x00" not in result
        assert ";" not in result
        assert " " not in result

    def test_preserves_legitimate_dots_and_dashes(self):
        assert safe_filename("my-holiday_photo.v2.jpg") == "my-holiday_photo.v2.jpg"

    def test_nested_traversal_collapses_to_basename(self):
        # Even a deeply nested traversal attempt only ever yields the
        # final path component, never anything above it.
        result = safe_filename("a/../../../b/c/d.png")
        assert result == "d.png"


class TestHasAllowedExtension:
    def test_accepts_allowed_extension(self):
        assert has_allowed_extension("photo.jpg", {".jpg", ".png"})

    def test_rejects_disallowed_extension(self):
        assert not has_allowed_extension("script.py", {".jpg", ".png"})

    def test_case_insensitive(self):
        assert has_allowed_extension("PHOTO.JPG", {".jpg"})

    def test_rejects_no_extension(self):
        assert not has_allowed_extension("noextension", {".jpg"})

    def test_rejects_double_extension_disguise(self):
        # "photo.jpg.exe" is a classic disguise trick — only the final
        # extension counts, and it correctly fails here.
        assert not has_allowed_extension("photo.jpg.exe", {".jpg"})
