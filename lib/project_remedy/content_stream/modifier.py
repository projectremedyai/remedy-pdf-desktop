"""Content stream rewriter — applies targeted modifications to a page.

Uses pikepdf.parse_content_stream() / pikepdf.unparse_content_stream() to
round-trip a page's content stream with specific instruction replacements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pikepdf
from pikepdf import Name, Object, Operator

logger = logging.getLogger(__name__)


def _to_pikepdf_operand(value):
    """Convert a Python numeric value to a pikepdf-compatible operand."""
    if isinstance(value, (int, float)):
        return Object.parse(str(value).encode())
    return value


@dataclass
class ColorModification:
    """A targeted modification to a content stream instruction."""

    instruction_index: int
    new_operands: list
    operator: str  # "rg", "RG", "g", "G", etc.
    insert_before: int | None = None  # insert a new instruction before this index


@dataclass
class InstructionInsert:
    """A new instruction to insert at a given position."""

    position: int  # insert before this index
    operands: list
    operator: str


class ContentStreamModifier:
    """Rewrites page content streams with targeted modifications."""

    def apply(
        self,
        page: pikepdf.Page,
        modifications: list[ColorModification],
        pdf: pikepdf.Pdf | None = None,
    ) -> None:
        """Parse the content stream, apply modifications, write back."""
        if not modifications:
            return

        try:
            instructions = list(pikepdf.parse_content_stream(page))
        except Exception:
            logger.warning("Failed to parse content stream for modification")
            return

        replacements: dict[int, ColorModification] = {}
        inserts: list[InstructionInsert] = []

        for mod in modifications:
            if mod.insert_before is not None:
                inserts.append(InstructionInsert(
                    position=mod.insert_before,
                    operands=mod.new_operands,
                    operator=mod.operator,
                ))
            else:
                replacements[mod.instruction_index] = mod

        # Apply replacements
        for idx, mod in replacements.items():
            if 0 <= idx < len(instructions):
                instructions[idx] = (
                    [_to_pikepdf_operand(o) for o in mod.new_operands],
                    Operator(mod.operator),
                )

        # Apply inserts (process in reverse order so indices stay valid)
        inserts.sort(key=lambda i: i.position, reverse=True)
        for ins in inserts:
            new_instruction = (
                [_to_pikepdf_operand(o) for o in ins.operands],
                Operator(ins.operator),
            )
            pos = min(ins.position, len(instructions))
            instructions.insert(pos, new_instruction)

        # Write back
        new_stream = pikepdf.unparse_content_stream(instructions)
        page.contents_coalesce()
        page.obj["/Contents"].write(new_stream)

    def replace_color_at(
        self,
        page: pikepdf.Page,
        instruction_index: int,
        new_color: tuple[float, ...],
        operator: str,
    ) -> None:
        """Convenience: replace a single color instruction."""
        mod = ColorModification(
            instruction_index=instruction_index,
            new_operands=list(new_color),
            operator=operator,
        )
        self.apply(page, [mod])

    def insert_color_before(
        self,
        page: pikepdf.Page,
        before_index: int,
        color: tuple[float, ...],
        operator: str,
    ) -> None:
        """Insert a color instruction before the given index."""
        mod = ColorModification(
            instruction_index=-1,
            new_operands=list(color),
            operator=operator,
            insert_before=before_index,
        )
        self.apply(page, [mod])
