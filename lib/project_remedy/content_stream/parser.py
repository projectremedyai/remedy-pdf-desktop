"""Graphics state tracker — walks a page content stream tracking the full state.

Produces a list of AnnotatedInstruction objects, each carrying a snapshot of
the graphics state at the time the instruction was encountered.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import pikepdf
from pikepdf import Name, Object

logger = logging.getLogger(__name__)


@dataclass
class GraphicsState:
    """Snapshot of PDF graphics state at a point in the content stream."""

    fill_color: tuple[float, ...] = (0.0, 0.0, 0.0)
    stroke_color: tuple[float, ...] = (0.0, 0.0, 0.0)
    fill_colorspace: str = "DeviceGray"
    stroke_colorspace: str = "DeviceGray"
    font_name: str = ""
    font_size: float = 0.0
    ctm: list[float] = field(default_factory=lambda: [1, 0, 0, 1, 0, 0])
    text_matrix: list[float] = field(default_factory=lambda: [1, 0, 0, 1, 0, 0])
    line_matrix: list[float] = field(default_factory=lambda: [1, 0, 0, 1, 0, 0])
    in_text_block: bool = False
    mcid: int | None = None
    marked_tag: str = ""


@dataclass
class AnnotatedInstruction:
    """A content stream instruction annotated with the state at that point."""

    operands: list
    operator: str
    index: int
    state: GraphicsState


def _to_float(obj: Object) -> float:
    """Safely convert a pikepdf operand to float."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return 0.0


def _to_floats(operands: list) -> tuple[float, ...]:
    """Convert a list of pikepdf operands to a tuple of floats."""
    return tuple(_to_float(o) for o in operands)


class GraphicsStateTracker:
    """Walks a page's content stream and tracks the full graphics state.

    Returns a list of AnnotatedInstruction objects — one per instruction in
    the stream, each carrying a snapshot of the graphics state.
    """

    def track(self, page: pikepdf.Page) -> list[AnnotatedInstruction]:
        """Parse the page content stream and return annotated instructions."""
        try:
            instructions = pikepdf.parse_content_stream(page)
        except Exception:
            logger.warning("Failed to parse content stream for page")
            return []

        state = GraphicsState()
        state_stack: list[GraphicsState] = []
        result: list[AnnotatedInstruction] = []

        for idx, (operands, operator) in enumerate(instructions):
            op = str(operator)

            self._handle_operator(op, operands, state, state_stack)

            result.append(AnnotatedInstruction(
                operands=list(operands),
                operator=op,
                index=idx,
                state=copy.deepcopy(state),
            ))

        return result

    def track_with_form_xobjects(
        self,
        page: pikepdf.Page,
        pdf: pikepdf.Pdf,
    ) -> list[AnnotatedInstruction]:
        """Like track(), but recurses into Form XObjects referenced by `Do`."""
        result = self.track(page)

        resources = page.obj.get("/Resources", pikepdf.Dictionary())
        xobjects = resources.get("/XObject", pikepdf.Dictionary())

        expanded: list[AnnotatedInstruction] = []
        for ann in result:
            expanded.append(ann)
            if ann.operator == "Do" and ann.operands:
                xobj_name = str(ann.operands[0])
                xobj = xobjects.get(xobj_name)
                if xobj is not None:
                    try:
                        subtype = str(xobj.get("/Subtype", ""))
                        if subtype == "/Form":
                            sub_instructions = pikepdf.parse_content_stream(xobj)
                            sub_state = copy.deepcopy(ann.state)
                            sub_stack: list[GraphicsState] = []
                            for sub_idx, (ops, op) in enumerate(sub_instructions):
                                op_str = str(op)
                                self._handle_operator(
                                    op_str, ops, sub_state, sub_stack
                                )
                                expanded.append(AnnotatedInstruction(
                                    operands=list(ops),
                                    operator=op_str,
                                    index=len(result) + sub_idx,
                                    state=copy.deepcopy(sub_state),
                                ))
                    except Exception:
                        logger.debug("Could not parse Form XObject %s", xobj_name)

        return expanded

    def _handle_operator(
        self,
        op: str,
        operands: list,
        state: GraphicsState,
        state_stack: list[GraphicsState],
    ) -> None:
        """Update state based on the operator."""
        # Graphics state stack
        if op == "q":
            state_stack.append(copy.deepcopy(state))
        elif op == "Q":
            if state_stack:
                restored = state_stack.pop()
                in_text = state.in_text_block
                tm = state.text_matrix
                lm = state.line_matrix
                state.__dict__.update(restored.__dict__)
                state.in_text_block = in_text
                state.text_matrix = tm
                state.line_matrix = lm

        # Color operators — DeviceGray
        elif op == "g":
            g = _to_float(operands[0]) if operands else 0.0
            state.fill_color = (g,)
            state.fill_colorspace = "DeviceGray"
        elif op == "G":
            g = _to_float(operands[0]) if operands else 0.0
            state.stroke_color = (g,)
            state.stroke_colorspace = "DeviceGray"

        # Color operators — DeviceRGB
        elif op == "rg":
            state.fill_color = _to_floats(operands[:3])
            state.fill_colorspace = "DeviceRGB"
        elif op == "RG":
            state.stroke_color = _to_floats(operands[:3])
            state.stroke_colorspace = "DeviceRGB"

        # Color operators — DeviceCMYK
        elif op == "k":
            state.fill_color = _to_floats(operands[:4])
            state.fill_colorspace = "DeviceCMYK"
        elif op == "K":
            state.stroke_color = _to_floats(operands[:4])
            state.stroke_colorspace = "DeviceCMYK"

        # Color space
        elif op == "cs":
            if operands:
                state.fill_colorspace = str(operands[0]).lstrip("/")
        elif op == "CS":
            if operands:
                state.stroke_colorspace = str(operands[0]).lstrip("/")

        # Color operators — generic (sc/SC/scn/SCN)
        elif op in ("sc", "scn"):
            state.fill_color = _to_floats(operands)
        elif op in ("SC", "SCN"):
            state.stroke_color = _to_floats(operands)

        # Text state
        elif op == "BT":
            state.in_text_block = True
            state.text_matrix = [1, 0, 0, 1, 0, 0]
            state.line_matrix = [1, 0, 0, 1, 0, 0]
        elif op == "ET":
            state.in_text_block = False

        elif op == "Tf":
            if len(operands) >= 2:
                state.font_name = str(operands[0]).lstrip("/")
                state.font_size = _to_float(operands[1])

        elif op == "Td":
            if len(operands) >= 2:
                tx = _to_float(operands[0])
                ty = _to_float(operands[1])
                lm = state.line_matrix
                state.line_matrix = [lm[0], lm[1], lm[2], lm[3],
                                     lm[4] + tx, lm[5] + ty]
                state.text_matrix = list(state.line_matrix)

        elif op == "TD":
            if len(operands) >= 2:
                tx = _to_float(operands[0])
                ty = _to_float(operands[1])
                lm = state.line_matrix
                state.line_matrix = [lm[0], lm[1], lm[2], lm[3],
                                     lm[4] + tx, lm[5] + ty]
                state.text_matrix = list(state.line_matrix)

        elif op == "Tm":
            if len(operands) >= 6:
                state.text_matrix = [_to_float(o) for o in operands[:6]]
                state.line_matrix = list(state.text_matrix)

        elif op == "T*":
            lm = state.line_matrix
            state.line_matrix = [lm[0], lm[1], lm[2], lm[3], lm[4], lm[5]]
            state.text_matrix = list(state.line_matrix)

        # Transform
        elif op == "cm":
            if len(operands) >= 6:
                new = [_to_float(o) for o in operands[:6]]
                ctm = state.ctm
                state.ctm = [
                    new[0] * ctm[0] + new[1] * ctm[2],
                    new[0] * ctm[1] + new[1] * ctm[3],
                    new[2] * ctm[0] + new[3] * ctm[2],
                    new[2] * ctm[1] + new[3] * ctm[3],
                    new[4] * ctm[0] + new[5] * ctm[2] + ctm[4],
                    new[4] * ctm[1] + new[5] * ctm[3] + ctm[5],
                ]

        # Marked content
        elif op == "BMC":
            if operands:
                state.marked_tag = str(operands[0]).lstrip("/")
        elif op == "BDC":
            if len(operands) >= 2:
                state.marked_tag = str(operands[0]).lstrip("/")
                props = operands[1]
                if isinstance(props, pikepdf.Dictionary):
                    mcid_val = props.get("/MCID")
                    if mcid_val is not None:
                        state.mcid = int(mcid_val)
        elif op == "EMC":
            state.marked_tag = ""
            state.mcid = None
