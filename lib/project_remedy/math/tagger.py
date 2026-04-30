"""Embed /Formula structure elements and MathML in PDFs."""

from __future__ import annotations

import logging
from pathlib import Path

import pikepdf

from project_remedy.math.converter import latex_to_alt_text, latex_to_mathml
from project_remedy.math.extractor import DetectedFormula

logger = logging.getLogger(__name__)


def tag_formulas(
    pdf_path: Path,
    formulas: list[DetectedFormula],
    output_path: Path | None = None,
) -> list[str]:
    """Add /Formula structure elements to a PDF for detected math formulas.

    For each formula:
    - Creates a /Formula structure element under the document's StructTreeRoot
    - Sets /Alt with a screen-reader-friendly text description
    - Stores MathML in /AF (Associated Files) when conversion succeeds (PDF 2.0)

    Returns a list of change descriptions.
    """
    if not formulas:
        return []

    if output_path is None:
        output_path = pdf_path

    changes: list[str] = []

    with pikepdf.open(pdf_path, allow_overwriting_input=(output_path == pdf_path)) as pdf:
        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root is None:
            struct_root = pdf.make_indirect(pikepdf.Dictionary({
                "/Type": pikepdf.Name("/StructTreeRoot"),
                "/K": pikepdf.Array([]),
                "/ParentTree": pikepdf.Dictionary({"/Nums": pikepdf.Array([])}),
            }))
            pdf.Root["/StructTreeRoot"] = struct_root

        kids = struct_root.get("/K")
        if kids is None:
            kids = pikepdf.Array([])
            struct_root["/K"] = kids

        # Find or create a top-level /Document element to parent formulas under
        doc_elem = None
        if isinstance(kids, pikepdf.Array):
            for child in kids:
                if isinstance(child, pikepdf.Dictionary):
                    stype = str(child.get("/S", ""))
                    if stype in ("/Document", "/Part", "/Sect"):
                        doc_elem = child
                        break

        parent = doc_elem if doc_elem is not None else struct_root

        for formula in formulas:
            alt = formula.alt_text
            if not alt:
                alt = latex_to_alt_text(formula.latex)

            mathml = latex_to_mathml(formula.latex)

            formula_elem = pdf.make_indirect(pikepdf.Dictionary({
                "/Type": pikepdf.Name("/StructElem"),
                "/S": pikepdf.Name("/Formula"),
                "/P": parent,
                "/Alt": pikepdf.String(alt),
            }))

            # Attach MathML as Associated File (PDF 2.0 convention)
            if mathml:
                mathml_bytes = mathml.encode("utf-8")
                mathml_stream = pdf.make_stream(mathml_bytes)
                mathml_stream["/Type"] = pikepdf.Name("/EmbeddedFile")
                mathml_stream["/Subtype"] = pikepdf.Name("/application#2Fmathml+xml")

                filespec = pdf.make_indirect(pikepdf.Dictionary({
                    "/Type": pikepdf.Name("/Filespec"),
                    "/F": pikepdf.String(f"formula_p{formula.page + 1}.mml"),
                    "/UF": pikepdf.String(f"formula_p{formula.page + 1}.mml"),
                    "/EF": pikepdf.Dictionary({"/F": mathml_stream}),
                    "/AFRelationship": pikepdf.Name("/Supplement"),
                }))

                formula_elem["/AF"] = pikepdf.Array([filespec])

            # Add to parent's /K array
            parent_k = parent.get("/K")
            if parent_k is None:
                parent["/K"] = pikepdf.Array([formula_elem])
            elif isinstance(parent_k, pikepdf.Array):
                parent_k.append(formula_elem)
            else:
                parent["/K"] = pikepdf.Array([parent_k, formula_elem])

            location = f" at {formula.location}" if formula.location else ""
            changes.append(
                f"Page {formula.page + 1}: Tagged formula \"{formula.latex[:40]}\" "
                f"with /Alt and MathML{location}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(str(output_path))

    logger.info("Tagged %d formulas in %s", len(formulas), output_path.name)
    return changes
