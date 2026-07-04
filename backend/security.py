"""
security.py
------------
Filename sanitisation for anything derived from client input.

Why this exists
----------------
FastAPI's UploadFile.filename and any path parameter (e.g. the
{filename} in DELETE /delete/pool/{filename}) are attacker-controlled
strings. Before this module existed, route handlers built filesystem
paths directly from those values:

    file_path = os.path.join(POOL_DIR, file.filename)

A client sending a filename like "../../app.py" or an absolute path
could write to or reference files outside POOL_DIR / QUERY_DIR. This
module is the single place that turns untrusted input into a safe
basename before it ever touches os.path.join.
"""

import os
import re
import uuid

# Anything that isn't a letter, digit, dot, dash, or underscore is stripped.
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(raw_name: str) -> str:
    """
    Convert an untrusted filename into a safe basename.

    Steps
    -----
    1. Strip any directory components (os.path.basename handles both
       "../x" and "/etc/passwd" style traversal attempts, and Windows-
       style "..\\x" once normalised).
    2. Reject the result if it's empty or resolves to "." / "..".
    3. Strip characters outside a small allow-list to avoid null bytes,
       control characters, or shell-special characters ending up in a
       path used later by shutil/os calls.

    Parameters
    ----------
    raw_name : str
        The client-supplied filename (from UploadFile.filename or a
        path parameter).

    Returns
    -------
    str
        A safe basename. If the input sanitises to nothing usable,
        a random name is generated instead so the caller always gets
        back a valid filename rather than having to handle a None/empty
        case separately.
    """
    # Normalise Windows-style separators too, then take the basename.
    normalised = raw_name.replace("\\", "/")
    candidate = os.path.basename(normalised)

    if candidate in ("", ".", ".."):
        return f"upload_{uuid.uuid4().hex}"

    candidate = _SAFE_CHARS.sub("_", candidate)

    if candidate in ("", ".", ".."):
        return f"upload_{uuid.uuid4().hex}"

    return candidate


def has_allowed_extension(filename: str, allowed_extensions: set[str]) -> bool:
    """
    Case-insensitive extension check against an allow-list.

    Parameters
    ----------
    filename : str
        A filename (already sanitised or not — only the extension is read).
    allowed_extensions : set[str]
        Lowercase extensions including the dot, e.g. {".jpg", ".png"}.
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() in allowed_extensions
