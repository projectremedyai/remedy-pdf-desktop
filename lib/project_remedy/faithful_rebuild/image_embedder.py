"""Copy image XObjects and other resources between PDFs verbatim.

This module is the bridge between source-PDF image extraction and the
faithful-rebuild page renderer.  It copies raster images (and ancillary
resource dicts like ExtGState, ColorSpace, Pattern, Shading, Properties)
from a source page into a target page, preserving compression filters,
color spaces, masks, and all other stream attributes exactly.

Two public functions are exposed:

- :func:`copy_page_images` — copy only ``/XObject`` entries whose
  ``/Subtype`` is ``/Image``.
- :func:`copy_page_resources` — copy non-font, non-image resource dicts
  (``/ExtGState``, ``/ColorSpace``, ``/Pattern``, ``/Shading``,
  ``/Properties``).
"""

from __future__ import annotations

import hashlib
from typing import Optional

import pikepdf
from pikepdf import Dictionary, Name


# ---------------------------------------------------------------------------
# Default resource types copied by copy_page_resources
# ---------------------------------------------------------------------------

_DEFAULT_RESOURCE_TYPES: list[str] = [
    "/ExtGState",
    "/ColorSpace",
    "/Pattern",
    "/Shading",
    "/Properties",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stream_hash(stream: pikepdf.Stream) -> str:
    """Return a hex SHA-256 digest of *stream*'s raw (compressed) bytes."""
    raw = stream.read_raw_bytes()
    return hashlib.sha256(raw).hexdigest()


def _get_or_create_xobject_dict(
    pdf: pikepdf.Pdf, page: pikepdf.Page
) -> pikepdf.Dictionary:
    """Return the ``/Resources/XObject`` dict for *page*, creating it if absent."""
    page_obj = page.obj
    if "/Resources" not in page_obj:
        page_obj["/Resources"] = Dictionary()
    res = page_obj["/Resources"]
    if "/XObject" not in res:
        res["/XObject"] = Dictionary()
    return res["/XObject"]  # type: ignore[return-value]


def _unique_name(
    base: str, existing: pikepdf.Dictionary
) -> str:
    """Return ``base`` if not taken in *existing*, else ``base_1``, ``base_2``, …"""
    candidate = base
    counter = 0
    while pikepdf.Name(f"/{candidate}") in existing:
        counter += 1
        candidate = f"{base}_{counter}"
    return candidate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def copy_page_images(
    source_pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
    target_pdf: pikepdf.Pdf,
    target_page: pikepdf.Page,
    *,
    dedup_cache: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Copy image XObjects from *source_page* into *target_page*.

    Only ``/XObject`` entries with ``/Subtype /Image`` are processed.  All
    other XObject subtypes (Form, PS) are ignored.

    The raw (compressed) stream bytes are hashed to detect duplicates.  When
    *dedup_cache* is supplied, images that have already been copied in a
    previous call (i.e. on an earlier page of the same rebuild job) are
    re-used: the existing target name is registered in ``target_page``'s
    ``/Resources/XObject`` without creating a second copy in ``target_pdf``.

    The dedup_cache maps ``content_hash → (target_name, indirect_object)``.
    Callers should treat the dict as opaque and pass the same instance across
    pages.  For the common single-page case the cache is created locally.

    Args:
        source_pdf: The open source :class:`pikepdf.Pdf`.
        source_page: The source page whose ``/XObject`` resources to copy.
        target_pdf: The open target :class:`pikepdf.Pdf` to copy into.
        target_page: The target page that will reference the copied images.
        dedup_cache: Optional dict[content_hash → target_name] shared
            across multiple calls for multi-page deduplication.  The values
            are ``(target_name, indirect_pikepdf_object)`` tuples internally,
            but the *public* type hint is ``dict[str, str]`` for caller
            convenience — callers should simply pass an empty ``{}`` and reuse
            it across pages.

    Returns:
        A ``dict[source_name, target_name]`` mapping (without leading ``/``).
        Empty when the source page has no image XObjects.
    """
    # We use the public dict[str, str] hint but internally store
    # (target_name, indirect_obj) to support cross-page re-registration.
    # A sentinel prefix distinguishes the internal format.
    _CACHE_OBJ_KEY = "_obj_"

    if dedup_cache is None:
        _local_cache: dict[str, object] = {}
        _cache: dict[str, object] = _local_cache
    else:
        _cache = dedup_cache  # type: ignore[assignment]

    src_res = source_page.obj.get("/Resources")
    if src_res is None:
        return {}

    src_xobjects = src_res.get("/XObject")
    if src_xobjects is None:
        return {}

    mapping: dict[str, str] = {}
    tgt_xobjects = _get_or_create_xobject_dict(target_pdf, target_page)

    for key in src_xobjects.keys():
        xobj = src_xobjects[key]

        # Must be a stream with /Subtype /Image
        try:
            subtype = xobj.get("/Subtype")
        except AttributeError:
            continue
        if subtype is None or str(subtype) != "/Image":
            continue

        # Hash the raw bytes for dedup
        try:
            content_hash = _stream_hash(xobj)
        except Exception:
            # If we can't hash (e.g. unusual stream), fall through to a plain copy
            content_hash = None

        # Determine source name (strip leading /)
        src_name = str(key).lstrip("/")

        name_key = content_hash
        obj_key = f"{_CACHE_OBJ_KEY}{content_hash}" if content_hash else None

        if name_key is not None and name_key in _cache:
            # Re-use the already-copied indirect object from a prior page
            existing_target_name = _cache[name_key]  # type: ignore[assignment]
            existing_obj = _cache[obj_key]  # type: ignore[index]
            # Register the same indirect object on this page's XObject dict
            tgt_xobjects[Name(f"/{existing_target_name}")] = existing_obj
            mapping[src_name] = existing_target_name
            continue

        # Copy the XObject into target_pdf (verbatim, preserving all keys)
        copied = target_pdf.copy_foreign(xobj)
        indirect_copy = target_pdf.make_indirect(copied)

        # Find a unique name in the target XObject dict
        tgt_name = _unique_name(src_name, tgt_xobjects)
        tgt_xobjects[Name(f"/{tgt_name}")] = indirect_copy

        if name_key is not None:
            _cache[name_key] = tgt_name  # type: ignore[assignment]
            _cache[obj_key] = indirect_copy  # type: ignore[index]

        mapping[src_name] = tgt_name

    return mapping


def copy_page_resources(
    source_pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
    target_pdf: pikepdf.Pdf,
    target_page: pikepdf.Page,
    *,
    resource_types: Optional[list[str]] = None,
) -> None:
    """Copy non-font, non-image resource dicts from *source_page* to *target_page*.

    By default the following resource dict keys are copied:

    - ``/ExtGState``
    - ``/ColorSpace``
    - ``/Pattern``
    - ``/Shading``
    - ``/Properties``

    ``/Font`` and ``/XObject`` are intentionally excluded; fonts are handled
    by :mod:`font_embedder` and images by :func:`copy_page_images`.

    Each entry inside the resource dict is copied individually via
    ``target_pdf.copy_foreign()``, so entries that are indirect objects are
    properly owned by *target_pdf*.  Existing entries in the target dict with
    the same name are overwritten.

    Args:
        source_pdf: The open source :class:`pikepdf.Pdf`.
        source_page: The source page to read resources from.
        target_pdf: The open target :class:`pikepdf.Pdf` to copy into.
        target_page: The target page whose ``/Resources`` will be updated.
        resource_types: List of ``/Key`` strings to copy.  Defaults to
            ``[/ExtGState, /ColorSpace, /Pattern, /Shading, /Properties]``.
    """
    if resource_types is None:
        resource_types = _DEFAULT_RESOURCE_TYPES

    src_res = source_page.obj.get("/Resources")
    if src_res is None:
        return

    tgt_page_obj = target_page.obj
    if "/Resources" not in tgt_page_obj:
        tgt_page_obj["/Resources"] = Dictionary()
    tgt_res = tgt_page_obj["/Resources"]

    for res_key in resource_types:
        name = Name(res_key)
        src_dict = src_res.get(res_key)
        if src_dict is None:
            continue

        # Ensure target has the resource sub-dict
        if name not in tgt_res:
            tgt_res[name] = Dictionary()
        tgt_sub = tgt_res[name]

        # Copy each entry inside the resource dict.
        # copy_foreign requires an indirect object, so we ensure the source
        # entry is indirect before crossing PDF boundaries.
        for entry_key in src_dict.keys():
            entry_val = src_dict[entry_key]
            if not entry_val.is_indirect:
                entry_val = source_pdf.make_indirect(entry_val)
            copied = target_pdf.copy_foreign(entry_val)
            tgt_sub[entry_key] = target_pdf.make_indirect(copied)
