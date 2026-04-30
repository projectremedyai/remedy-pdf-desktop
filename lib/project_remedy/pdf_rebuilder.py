"""Hybrid rebuild fallback for PDFs that can't be structure-repaired.

Extracts page content with fitz, rebuilds with reportlab + embeddable fonts,
then adds a fresh tag tree with pikepdf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pikepdf


@dataclass
class RebuildResult:
    success: bool
    source_path: Path
    output_path: Path | None
    visual_diff: float = 0.0
    error: str = ""
    page_count: int = 0


def rebuild_pdf(
    source_path: Path,
    output_path: Path,
    *,
    visual_tolerance: float = 0.02,
    config=None,
) -> RebuildResult:
    try:
        source_doc = fitz.open(str(source_path))
    except Exception as exc:
        return RebuildResult(
            success=False, source_path=source_path, output_path=None,
            error=f"Cannot open source: {exc}",
        )

    page_count = len(source_doc)

    try:
        pages_data = _extract_pages(source_doc)
        _rebuild_with_reportlab(pages_data, output_path)
        _add_fresh_tag_tree(output_path, pages_data, source_path)
        diff = _compute_visual_diff(source_path, output_path)

        if diff > visual_tolerance:
            return RebuildResult(
                success=False, source_path=source_path, output_path=output_path,
                visual_diff=diff, page_count=page_count,
                error=f"Visual diff {diff:.4f} exceeds tolerance {visual_tolerance}",
            )

        return RebuildResult(
            success=True, source_path=source_path, output_path=output_path,
            visual_diff=diff, page_count=page_count,
        )
    except Exception as exc:
        return RebuildResult(
            success=False, source_path=source_path, output_path=None,
            error=str(exc), page_count=page_count,
        )
    finally:
        source_doc.close()


def _extract_pages(doc: fitz.Document) -> list[dict]:
    pages = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        pages.append({
            "width": page.rect.width,
            "height": page.rect.height,
            "blocks": blocks,
        })
    return pages


def _rebuild_with_reportlab(pages_data: list[dict], output_path: Path) -> None:
    from reportlab.pdfgen import canvas

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path))

    for page_data in pages_data:
        w, h = page_data["width"], page_data["height"]
        c.setPageSize((w, h))

        for block in page_data["blocks"]:
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        x = span["origin"][0]
                        y = h - span["origin"][1]
                        size = span.get("size", 12)
                        text = span.get("text", "")
                        if text.strip():
                            c.setFont("Helvetica", size)
                            c.drawString(x, y, text)

        c.showPage()

    c.save()


def _mark_page_content(page: pikepdf.Page, mcid_start: int = 0) -> int:
    """Wrap each text/drawing run in BDC/EMC operators with incremental MCIDs.

    Returns the next available MCID after this page.
    """
    instructions = list(pikepdf.parse_content_stream(page))
    if not instructions:
        return mcid_start

    marked: list[tuple] = []
    mcid = mcid_start
    in_text = False
    text_ops: list[tuple] = []

    def _flush_text():
        nonlocal mcid
        if not text_ops:
            return
        # Wrap accumulated text block in BDC/EMC
        marked.append((
            [pikepdf.Name("/P"), pikepdf.Dictionary({"/MCID": mcid})],
            pikepdf.Operator("BDC"),
        ))
        marked.extend(text_ops)
        marked.append(([], pikepdf.Operator("EMC")))
        text_ops.clear()
        mcid += 1

    for operands, operator in instructions:
        op = str(operator)
        if op == "BT":
            in_text = True
            text_ops.append((operands, operator))
        elif op == "ET":
            text_ops.append((operands, operator))
            in_text = False
            _flush_text()
        elif in_text:
            text_ops.append((operands, operator))
        elif op in ("BDC", "BMC"):
            # Already marked — pass through as-is until matching EMC
            marked.append((operands, operator))
        else:
            marked.append((operands, operator))

    # Flush any remaining text ops (shouldn't happen in well-formed PDF)
    _flush_text()

    new_stream = pikepdf.unparse_content_stream(marked)
    page.contents_coalesce()
    page["/Contents"] = page.obj.make_stream(new_stream)
    return mcid


def _add_xmp_metadata(pdf: pikepdf.Pdf, title: str = "", lang: str = "en") -> None:
    """Add XMP metadata stream to satisfy veraPDF 7.1-8."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    if not title:
        title = str(pdf.docinfo.get("/Title", "Remediated Document"))

    xmp = f"""<?xpacket begin='\xef\xbb\xbf' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description rdf:about=''
      xmlns:dc='http://purl.org/dc/elements/1.1/'
      xmlns:xmp='http://ns.adobe.com/xap/1.0/'
      xmlns:pdfuaid='http://www.aiim.org/pdfua/ns/id/'>
      <dc:title><rdf:Alt><rdf:li xml:lang='x-default'>{title}</rdf:li></rdf:Alt></dc:title>
      <dc:language><rdf:Bag><rdf:li>{lang}</rdf:li></rdf:Bag></dc:language>
      <xmp:CreateDate>{now}</xmp:CreateDate>
      <xmp:ModifyDate>{now}</xmp:ModifyDate>
      <pdfuaid:part>1</pdfuaid:part>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""

    metadata_stream = pdf.make_stream(xmp.encode("utf-8"))
    metadata_stream["/Type"] = pikepdf.Name("/Metadata")
    metadata_stream["/Subtype"] = pikepdf.Name("/XML")
    pdf.Root["/Metadata"] = metadata_stream


def _add_fresh_tag_tree(
    output_path: Path, pages_data: list[dict], source_path: Path,
) -> None:
    with pikepdf.open(output_path, allow_overwriting_input=True) as pdf:
        try:
            with pikepdf.open(source_path) as source:
                lang = str(source.Root.get("/Lang", "en"))
                title = str(source.docinfo.get("/Title", ""))
        except Exception:
            lang = "en"
            title = ""

        doc_elem = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/Document"),
            "/K": pikepdf.Array(),
        }))

        # First pass: write BDC/EMC operators into each page's content stream.
        page_mcid_ranges: list[tuple[int, int]] = []
        next_mcid = 0
        for page in pdf.pages:
            start_mcid = next_mcid
            next_mcid = _mark_page_content(page, start_mcid)
            # If no content was marked, create at least one MCID so the
            # structure tree has something to reference.
            if next_mcid == start_mcid:
                next_mcid = start_mcid + 1
            page_mcid_ranges.append((start_mcid, next_mcid))

        # Second pass: build structure tree referencing the MCIDs we just wrote.
        parent_nums = pikepdf.Array()
        for page_idx, page in enumerate(pdf.pages):
            start, end = page_mcid_ranges[page_idx]
            mcr_kids = pikepdf.Array()
            parent_arr_entries = []

            for mcid in range(start, end):
                p_elem = pdf.make_indirect(pikepdf.Dictionary({
                    "/Type": pikepdf.Name("/StructElem"),
                    "/S": pikepdf.Name("/P"),
                    "/P": doc_elem,
                    "/Pg": page.obj,
                    "/K": pikepdf.Array([
                        pikepdf.Dictionary({
                            "/Type": pikepdf.Name("/MCR"),
                            "/MCID": mcid,
                            "/Pg": page.obj,
                        })
                    ]),
                }))
                doc_elem["/K"].append(p_elem)
                parent_arr_entries.append(p_elem)

            page["/StructParents"] = page_idx
            parent_arr = pdf.make_indirect(pikepdf.Array(parent_arr_entries))
            parent_nums.append(page_idx)
            parent_nums.append(parent_arr)

        struct_root = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructTreeRoot"),
            "/K": doc_elem,
            "/ParentTree": pikepdf.Dictionary({"/Nums": parent_nums}),
            "/ParentTreeNextKey": len(pdf.pages),
        }))
        doc_elem["/P"] = struct_root
        pdf.Root["/StructTreeRoot"] = struct_root
        pdf.Root["/MarkInfo"] = pikepdf.Dictionary({"/Marked": True})
        pdf.Root["/Lang"] = lang
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary({
            "/DisplayDocTitle": True,
        })

        # Add XMP metadata (veraPDF 7.1-8).
        _add_xmp_metadata(pdf, title=title, lang=lang)

        pdf.save(str(output_path))


def _compute_visual_diff(source_path: Path, rebuilt_path: Path) -> float:
    try:
        source_doc = fitz.open(str(source_path))
        rebuilt_doc = fitz.open(str(rebuilt_path))
    except Exception:
        return 1.0

    if len(source_doc) != len(rebuilt_doc):
        source_doc.close()
        rebuilt_doc.close()
        return 1.0

    num_pages = len(source_doc)
    total_diff = 0.0
    for i in range(num_pages):
        src_pix = source_doc[i].get_pixmap(dpi=72)
        dst_pix = rebuilt_doc[i].get_pixmap(dpi=72)

        src_bytes = src_pix.samples
        dst_bytes = dst_pix.samples

        if len(src_bytes) != len(dst_bytes):
            total_diff += 1.0
            continue

        pixel_diffs = sum(abs(a - b) for a, b in zip(src_bytes, dst_bytes))
        max_diff = len(src_bytes) * 255
        total_diff += pixel_diffs / max_diff if max_diff > 0 else 0.0

    source_doc.close()
    rebuilt_doc.close()

    return total_diff / num_pages if num_pages > 0 else 1.0
