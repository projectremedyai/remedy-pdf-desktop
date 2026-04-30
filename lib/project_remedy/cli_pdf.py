"""Click ``pdf`` subgroup — PDF accessibility check, fix, and inspection commands.

Registered on the main CLI via ``cli.add_command(pdf_group)``.

Commands::

    remedy pdf check <file>
    remedy pdf fix <file>
    remedy pdf escalate <file>
    remedy pdf tags <file>
    remedy pdf info <file>
    remedy pdf reading-order <file>
    remedy pdf alt-text <file>
    remedy pdf artifacts <file>
    remedy pdf vision <file>
    remedy pdf screen-reader <file>
    remedy pdf report <file>
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from project_remedy.pdf_acceptance import (
    PDFAcceptanceResult,
    evaluate_pdf_acceptance,
)
from project_remedy.pdf_benchmark import (
    build_representative_corpus_manifest,
    default_manifest_path,
    load_corpus_manifest,
    run_gemini_tier_benchmark,
    run_acceptance_sweep,
    run_ocr_benchmark,
    save_corpus_manifest,
)
from project_remedy.serving_benchmark import (
    default_serving_benchmark_config,
    default_serving_benchmark_config_path,
    load_serving_benchmark_config,
    run_serving_bakeoff,
    run_serving_compatibility_probe,
    run_serving_compatibility_probe_for_manifest,
    save_serving_benchmark_config,
    save_serving_benchmark_run,
)
from project_remedy.pdf_checker import (
    CATEGORIES,
    CATEGORY_ALIASES,
    CheckReport,
    PDFAccessibilityChecker,
    walk_structure_tree,
    _get_struct_type,
    _node_has_direct_content,
)
from project_remedy.pdf_fixer import ALL_FIXES, FixReport, fix_all, fix_and_verify
from project_remedy.token_tracker import tracker

console = Console()


def _print_token_usage() -> None:
    """Print token usage summary if any API calls were made."""
    if tracker.total_calls == 0:
        return
    s = tracker.summary()
    thought_suffix = (
        f" + {s['thought_tokens']:,} thoughts"
        if s.get("thought_tokens", 0)
        else ""
    )
    parts = [
        f"[dim]Tokens: {s['input_tokens']:,} in + {s['output_tokens']:,} out"
        f"{thought_suffix} = {s['billed_total_tokens']:,} billed",
        f" | {s['api_calls']} API call(s)",
        f" | {s['elapsed_seconds']}s",
    ]
    by_prov = s.get("by_provider", {})
    if len(by_prov) > 1:
        prov_parts = [
            f"{name}: {d['total_tokens']:,}" for name, d in by_prov.items()
        ]
        parts.append(f" | {', '.join(prov_parts)}")
    console.print("".join(parts) + "[/dim]")


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _status_icon(status: str) -> str:
    if status == "Passed":
        return "[green]PASS[/green]"
    if status == "Failed":
        return "[red]FAIL[/red]"
    return "[yellow]MANUAL[/yellow]"


def _print_pdf_acceptance(result: PDFAcceptanceResult) -> None:
    """Print the shared composite PDF acceptance gate."""
    verdict = "[bold green]PASS[/bold green]" if result.passed else "[bold red]FAIL[/bold red]"
    console.print(f"\n{verdict} [bold]Composite PDF Acceptance[/bold]")
    console.print(f"  Checker failures: {len(result.checker_failures)}")
    console.print(f"  Screen reader errors: {len(result.screen_reader_errors)}")
    if result.verapdf_result.checked:
        status = "[green]PASS[/green]" if result.verapdf_result.passed else "[red]FAIL[/red]"
        console.print(
            f"  veraPDF: {status} "
            f"({len(result.verapdf_result.violations)} violation(s))"
        )
    else:
        console.print("  veraPDF: [dim]unavailable[/dim]")

    if not result.passed:
        for reason in result.failure_reasons():
            console.print(f"  [yellow]-[/yellow] {reason}")


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("pdf")
def pdf_group() -> None:
    """PDF accessibility tools — check, fix, inspect."""


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _load_pipeline_config(env: str | None, config: str | None):
    """Load PipelineConfig from the standard .env / config.yaml paths."""
    from project_remedy.config import load_config

    env_path = Path(env) if env else None
    yaml_path = Path(config) if config else None
    return load_config(env_path=env_path, yaml_path=yaml_path)


def _get_vision_result(cfg, pdf_path: Path, no_vision: bool):
    """Run vision analysis using pipeline config. Returns None if unavailable."""
    if no_vision:
        return None

    from project_remedy.pdf_vision import VisionAnalyzer, create_provider_from_config

    provider = create_provider_from_config(cfg)
    if provider is None:
        return None

    analyzer = VisionAnalyzer(provider)
    try:
        return asyncio.run(analyzer.analyze_all(pdf_path))
    except Exception as exc:
        console.print(f"[dim]Vision analysis unavailable: {exc}[/dim]")
        return None


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option(
    "--category", "-c",
    type=click.Choice(
        list(CATEGORY_ALIASES.keys()),
        case_sensitive=False,
    ),
    default=None,
    help="Run checks for one category only.",
)
@click.option("--no-vision", is_flag=True, help="Skip vision model analysis.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def check(
    file: str,
    env: str | None,
    config: str | None,
    category: str | None,
    no_vision: bool,
    as_json: bool,
) -> None:
    """Run all 32 accessibility checks on a PDF.

    Automatically uses the vision model from config.yaml / .env for
    reading order (check #4) and color contrast (check #8) analysis.
    Use --no-vision to skip.
    """
    tracker.reset()
    pdf_path = Path(file)

    try:
        cfg = _load_pipeline_config(env, config)
    except Exception:
        cfg = None

    vision_result = None
    if cfg and not no_vision:
        vision_result = _get_vision_result(cfg, pdf_path, no_vision)

    checker = PDFAccessibilityChecker(pdf_path, vision_result=vision_result)

    if category:
        report = checker.run_category(category)
    else:
        report = checker.run_all()

    if as_json:
        _print_report_json(report)
    else:
        _print_report_rich(report)

    _print_token_usage()


def _print_report_rich(report: CheckReport) -> None:
    header = (
        f"PDF Accessibility Report — {report.file_path.name} "
        f"({_human_size(report.file_size)}, {report.page_count} pages)"
    )

    lines: list[str] = []
    by_cat = report.results_by_category()

    for cat in CATEGORIES:
        results = by_cat.get(cat, [])
        if not results:
            continue
        passed = sum(1 for r in results if r.status == "Passed")
        lines.append(f"\n[bold]{cat}[/bold] ({passed}/{len(results)} passed)")

        for r in results:
            icon = _status_icon(r.status)
            fixable_tag = "  [dim]\\[fixable][/dim]" if r.status == "Failed" and r.fixable else ""
            lines.append(f"  {icon} {r.description}{fixable_tag}")
            for detail in r.details[:3]:
                lines.append(f"         [dim]{detail}[/dim]")

    # Summary line.
    lines.append("")
    lines.append(
        f"Summary: [green]{report.passed_count} passed[/green], "
        f"[red]{report.failed_count} failed[/red] "
        f"([cyan]{report.fixable_count} auto-fixable[/cyan]), "
        f"[yellow]{report.manual_count} manual[/yellow]"
    )
    if report.failed_count > 0:
        lines.append(
            f"Run: [bold]remedy pdf fix {report.file_path}[/bold]"
        )

    # Screen reader readability score.
    try:
        from project_remedy.compliance_report import calculate_screen_reader_readability
        from project_remedy.tag_tree_reader import validate_tag_tree

        sr_result = validate_tag_tree(report.file_path)
        score, _details = calculate_screen_reader_readability(
            report.file_path, sr_result, report,
        )
        if score >= 90:
            color, label = "green", "Excellent"
        elif score >= 70:
            color, label = "yellow", "Good"
        else:
            color, label = "red", "Needs Improvement"
        lines.append("")
        lines.append(
            f"Screen Reader Readability: [{color}]{score:.1f}/100 ({label})[/{color}]"
        )
    except Exception:
        pass

    body = "\n".join(lines)
    console.print(Panel(body, title=header, border_style="blue"))


def _print_report_json(report: CheckReport) -> None:
    import json

    data = {
        "file": str(report.file_path),
        "file_size": report.file_size,
        "page_count": report.page_count,
        "summary": {
            "passed": report.passed_count,
            "failed": report.failed_count,
            "fixable": report.fixable_count,
            "manual": report.manual_count,
        },
        "results": [
            {
                "rule_id": r.rule_id,
                "category": r.category,
                "description": r.description,
                "status": r.status,
                "details": r.details,
                "fixable": r.fixable,
            }
            for r in report.results
        ],
    }
    console.print_json(json.dumps(data))


# ---------------------------------------------------------------------------
# fix
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", "config_path", default=None, help="Path to config.yaml file.")
@click.option("-o", "--output", default=None, help="Output path (default: <file>_fixed.pdf).")
@click.option("--dry-run", is_flag=True, help="Show what would be fixed without writing.")
@click.option("--only", default=None, help="Fix only a specific rule_id.")
@click.option("--no-vision", is_flag=True, help="Skip vision model for alt text generation.")
@click.option("--thorough", is_flag=True, help="Send every page to vision model (skip heuristic pre-filter).")
def fix(
    file: str,
    env: str | None,
    config_path: str | None,
    output: str | None,
    dry_run: bool,
    only: str | None,
    no_vision: bool,
    thorough: bool,
) -> None:
    """Auto-fix all fixable accessibility issues.

    Uses the configured vision model to generate figure alt text
    automatically.  Use --no-vision to skip.

    Use --thorough to skip the heuristic pre-filter and analyze every
    page with the vision model for reading order and contrast.
    """
    tracker.reset()
    pdf_path = Path(file)
    output_path = Path(output) if output else None

    cfg = None
    if not no_vision:
        try:
            cfg = _load_pipeline_config(env, config_path)
        except Exception:
            pass

    use_verify_flow = not dry_run and only is None
    if use_verify_flow:
        report = fix_and_verify(
            pdf_path,
            output_path,
            config=cfg,
            thorough=thorough,
            max_cycles=3,
            conformance_repair=True,
            original_path=pdf_path,
        )
    else:
        report = fix_all(
            pdf_path,
            output_path,
            only=only,
            dry_run=dry_run,
            config=cfg,
            thorough=thorough,
        )

    _print_fix_report(report, dry_run)
    if use_verify_flow:
        acceptance_path = (
            report.output_path if report.output_path.exists() else report.input_path
        )
        acceptance = evaluate_pdf_acceptance(acceptance_path, config=cfg)
        _print_pdf_acceptance(acceptance)
    _print_token_usage()


def _print_fix_report(report: FixReport, dry_run: bool) -> None:
    verb = "Would fix" if dry_run else "Fixing"
    console.print(
        f"\n{verb} [bold]{report.input_path.name}[/bold] → "
        f"[bold]{report.output_path.name}[/bold]\n"
    )

    for change in report.changes:
        console.print(f"  [green]\\[FIXED][/green] {change}")

    for skip in report.skipped:
        console.print(f"  [yellow]\\[SKIP][/yellow]  {skip}")

    console.print("")
    if dry_run:
        console.print(
            f"  {report.fixed_count} issues would be fixed, "
            f"{report.skipped_count} skipped (manual)."
        )
    else:
        if report.changes:
            console.print(
                f"  {report.fixed_count} issues fixed, "
                f"{report.skipped_count} skipped (manual). "
                f"Saved to [bold]{report.output_path}[/bold]"
            )
        else:
            console.print("  No fixable issues found.")


def _acceptance_is_clean(acceptance) -> bool:
    retry_reasons = list(getattr(acceptance, "retry_reasons", []) or [])
    return bool(getattr(acceptance, "openable", getattr(acceptance, "passed", False))) and not retry_reasons


# ---------------------------------------------------------------------------
# escalate
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None, help="Output path (default: <file>_fixed.pdf).")
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", "config_path", default=None, help="Path to config.yaml file.")
@click.option("--thorough", is_flag=True, help="Send every page to vision model (skip heuristic pre-filter).")
def escalate(
    file: str,
    output: str | None,
    env: str | None,
    config_path: str | None,
    thorough: bool,
) -> None:
    """Fix PDF, then escalate to stronger model if checks still fail.

    Tier 1: fix_and_verify() with the default vision model.
    Tier 2: if composite acceptance still fails, re-run fix_and_verify()
    with the escalation model from config.yaml.
    """
    tracker.reset()
    pdf_path = Path(file)
    output_path = (
        Path(output)
        if output
        else pdf_path.parent / f"{pdf_path.stem}_fixed.pdf"
    )

    try:
        cfg = _load_pipeline_config(env, config_path)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    # Tier 1: fix with default model.
    console.print(f"\n[bold]Tier 1:[/bold] {pdf_path.name}...")
    tier1_report = fix_and_verify(
        pdf_path,
        output_path,
        config=cfg,
        thorough=thorough,
        max_cycles=3,
        conformance_repair=True,
        original_path=pdf_path,
    )
    _print_fix_report(tier1_report, dry_run=False)

    tier1_acceptance = evaluate_pdf_acceptance(output_path, original_path=pdf_path, config=cfg)
    _print_report_rich(tier1_acceptance.checker_report)
    _print_pdf_acceptance(tier1_acceptance)

    if _acceptance_is_clean(tier1_acceptance):
        console.print("[bold green]Composite PDF acceptance passed.[/bold green]")
        _print_token_usage()
        return

    # Tier 2: escalate to stronger model.
    from project_remedy.pdf_vision import create_escalation_provider

    esc_provider = create_escalation_provider(cfg)
    esc_model = cfg.api.escalation_model

    console.print(f"\n[bold]Tier 2:[/bold] Escalating to [cyan]{esc_model}[/cyan]...")
    tier2_report = fix_and_verify(
        output_path,
        output_path,
        config=cfg,
        thorough=True,
        vision_provider_override=esc_provider,
        max_cycles=3,
        conformance_repair=True,
        original_path=pdf_path,
    )
    _print_fix_report(tier2_report, dry_run=False)

    final = evaluate_pdf_acceptance(output_path, original_path=pdf_path, config=cfg)
    _print_report_rich(final.checker_report)
    _print_pdf_acceptance(final)

    if _acceptance_is_clean(final):
        console.print("[bold green]Composite PDF acceptance passed after escalation.[/bold green]")
    elif getattr(final, "openable", getattr(final, "passed", False)):
        console.print(
            "[bold yellow]Composite PDF acceptance has non-blocking warnings "
            "after escalation[/bold yellow]"
        )
    else:
        console.print(
            "[bold yellow]Composite PDF acceptance still failing "
            "— needs manual review[/bold yellow]"
        )

    _print_token_usage()


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--max-depth", "-d", default=0, type=int, help="Max tree depth (0=unlimited).")
def tags(file: str, max_depth: int) -> None:
    """Show the PDF structure tree."""
    import pikepdf

    pdf_path = Path(file)
    tree = Tree(f"[bold]{pdf_path.name}[/bold] — Structure Tree")

    with pikepdf.open(pdf_path) as pdf:
        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root is None:
            console.print("[yellow]No structure tree found.[/yellow]")
            return

        # Build tree nodes keyed by id(node).
        tree_nodes: dict[int, Tree] = {id(struct_root): tree}

        for node, depth, parent in walk_structure_tree(pdf):
            if node is struct_root:
                continue
            if max_depth and depth > max_depth:
                continue

            stype = _get_struct_type(node)
            alt = node.get("/Alt")
            label = f"[cyan]/{stype}[/cyan]"

            if alt and str(alt).strip():
                label += f'  [dim]alt="{str(alt)[:50]}"[/dim]'

            has_content = _node_has_direct_content(node)
            if has_content:
                label += "  [green]●[/green]"

            parent_tree = tree_nodes.get(id(parent), tree) if parent is not None else tree
            child_tree = parent_tree.add(label)
            tree_nodes[id(node)] = child_tree

    console.print(tree)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def info(file: str) -> None:
    """Show PDF metadata, language, mark info, and viewer preferences."""
    import pikepdf

    pdf_path = Path(file)
    table = Table(title=f"{pdf_path.name} — PDF Info", show_lines=True)
    table.add_column("Property", style="bold", min_width=25)
    table.add_column("Value")

    with pikepdf.open(pdf_path) as pdf:
        stat = pdf_path.stat()
        table.add_row("File size", _human_size(stat.st_size))
        table.add_row("Pages", str(len(pdf.pages)))

        # PDF version.
        table.add_row("PDF version", str(pdf.pdf_version))

        # Encryption.
        table.add_row("Encrypted", str(pdf.is_encrypted))

        # MarkInfo.
        mark_info = pdf.Root.get("/MarkInfo")
        if mark_info:
            marked = bool(mark_info.get("/Marked"))
            table.add_row("/MarkInfo/Marked", str(marked))
        else:
            table.add_row("/MarkInfo/Marked", "[red]Not set[/red]")

        # Language.
        lang = pdf.Root.get("/Lang")
        table.add_row("/Lang", str(lang) if lang else "[red]Not set[/red]")

        # ViewerPreferences.
        vp = pdf.Root.get("/ViewerPreferences")
        if vp:
            display = bool(vp.get("/DisplayDocTitle"))
            table.add_row("/DisplayDocTitle", str(display))
        else:
            table.add_row("/DisplayDocTitle", "[red]Not set[/red]")

        # Structure tree.
        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root:
            # Count elements.
            count = sum(1 for _ in walk_structure_tree(pdf))
            table.add_row("Structure tree elements", str(count))

            role_map = struct_root.get("/RoleMap")
            if role_map:
                mappings = [
                    f"{k} → {v}" for k, v in role_map.items()
                ]
                table.add_row("/RoleMap", ", ".join(mappings[:10]))
        else:
            table.add_row("Structure tree", "[red]Not found[/red]")

        # Outlines.
        outlines = pdf.Root.get("/Outlines")
        if outlines:
            outline_count = int(outlines.get("/Count", 0))
            table.add_row("Bookmarks", str(outline_count))
        else:
            table.add_row("Bookmarks", "None")

        # XMP metadata.
        try:
            with pdf.open_metadata() as meta:
                table.add_row("dc:title", str(meta.get("dc:title", "")))
                table.add_row("dc:language", str(meta.get("dc:language", "")))
                table.add_row("pdfuaid:part", str(meta.get("pdfuaid:part", "[red]Not set[/red]")))
                table.add_row("pdf:Producer", str(meta.get("pdf:Producer", "")))
        except Exception:
            table.add_row("XMP metadata", "[red]Error reading[/red]")

    console.print(table)


# ---------------------------------------------------------------------------
# reading-order
# ---------------------------------------------------------------------------


@pdf_group.command("reading-order")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--page", "-p", default=None, type=int, help="Show specific page only (1-based).")
def reading_order(file: str, page: int | None) -> None:
    """Show content reading order per page."""
    import pikepdf

    pdf_path = Path(file)

    with pikepdf.open(pdf_path) as pdf:
        pages_to_show = range(len(pdf.pages))
        if page:
            if page < 1 or page > len(pdf.pages):
                console.print(f"[red]Page {page} out of range (1-{len(pdf.pages)})[/red]")
                sys.exit(1)
            pages_to_show = [page - 1]

        for idx in pages_to_show:
            pg = pdf.pages[idx]
            console.print(f"\n[bold]Page {idx + 1}[/bold]")

            contents = pg.get("/Contents")
            if contents is None:
                console.print("  [dim](empty page)[/dim]")
                continue

            raw = b""
            if isinstance(contents, pikepdf.Array):
                for stream in contents:
                    try:
                        raw += stream.read_bytes()
                    except Exception:
                        pass
            else:
                try:
                    raw = contents.read_bytes()
                except Exception:
                    pass

            text = raw.decode("latin-1", errors="replace")

            # Find marked content blocks.
            order_num = 0
            for match in re.finditer(
                r"/(\w+)\s*(?:<<(.*?)>>)?\s*(BDC|BMC)", text
            ):
                tag = match.group(1)
                props = match.group(2) or ""
                order_num += 1
                mcid_match = re.search(r"/MCID\s+(\d+)", props)
                mcid = mcid_match.group(1) if mcid_match else "?"
                console.print(
                    f"  {order_num:3d}. /{tag}  MCID={mcid}"
                )

            if order_num == 0:
                console.print("  [dim](no marked content blocks)[/dim]")


# ---------------------------------------------------------------------------
# alt-text
# ---------------------------------------------------------------------------


@pdf_group.command("alt-text")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--missing-only", is_flag=True, help="Show only elements missing alt text.")
def alt_text(file: str, missing_only: bool) -> None:
    """List all alt text in the structure tree."""
    import pikepdf

    pdf_path = Path(file)
    table = Table(title=f"{pdf_path.name} — Alt Text", show_lines=True)
    table.add_column("#", justify="right", style="dim", max_width=5)
    table.add_column("Type", max_width=15)
    table.add_column("Alt Text", max_width=60)
    table.add_column("Has Content", justify="center", max_width=12)

    with pikepdf.open(pdf_path) as pdf:
        count = 0
        for node, _depth, _parent in walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if not stype:
                continue

            alt = node.get("/Alt")
            has_content = _node_has_direct_content(node)
            alt_str = str(alt) if alt is not None else ""

            if missing_only and alt is not None:
                continue

            # For --missing-only, only show elements that need alt.
            if missing_only and stype not in ("Figure", "Formula", "Form"):
                continue

            count += 1
            content_mark = "[green]Yes[/green]" if has_content else "[dim]No[/dim]"
            alt_display = alt_str[:60] if alt_str else "[red](none)[/red]"

            table.add_row(str(count), f"/{stype}", alt_display, content_mark)

            if count >= 200:
                table.add_row("...", "", f"({count}+ elements)", "")
                break

    console.print(table)


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def artifacts(file: str) -> None:
    """Show artifact vs tagged content breakdown per page."""
    import pikepdf

    pdf_path = Path(file)
    table = Table(title=f"{pdf_path.name} — Artifact vs Tagged Content", show_lines=True)
    table.add_column("Page", justify="right", max_width=6)
    table.add_column("Artifact Blocks", justify="right")
    table.add_column("Tagged Blocks", justify="right")
    table.add_column("Untagged Bytes", justify="right")

    with pikepdf.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, 1):
            contents = page.get("/Contents")
            if contents is None:
                table.add_row(str(idx), "0", "0", "0")
                continue

            raw = b""
            if isinstance(contents, pikepdf.Array):
                for stream in contents:
                    try:
                        raw += stream.read_bytes()
                    except Exception:
                        pass
            else:
                try:
                    raw = contents.read_bytes()
                except Exception:
                    pass

            text = raw.decode("latin-1", errors="replace")

            artifact_count = len(re.findall(r"/Artifact\s*(<<.*?>>)?\s*(BDC|BMC)", text))
            tagged_count = len(re.findall(r"/\w+\s*<<.*?>>\s*BDC", text)) - artifact_count

            # Calculate untagged bytes: content before first BDC/BMC.
            first_marked = re.search(r"/\w+\s*(<<.*?>>)?\s*(BDC|BMC)", text)
            untagged = len(text[: first_marked.start()].strip()) if first_marked else 0

            table.add_row(
                str(idx),
                str(artifact_count),
                str(max(0, tagged_count)),
                str(untagged),
            )

    console.print(table)


# ---------------------------------------------------------------------------
# vision — standalone vision analysis
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option(
    "--analyze", "-a",
    type=click.Choice(["reading-order", "contrast", "all"], case_sensitive=False),
    default="all",
    help="What to analyze (default: all).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def vision(
    file: str,
    env: str | None,
    config: str | None,
    analyze: str,
    as_json: bool,
) -> None:
    """Analyze PDF with a vision model for reading order and contrast.

    Uses the vision provider configured in config.yaml / .env
    (api.llm_backend, GEMINI_API_KEY, etc.).
    """
    from project_remedy.pdf_vision import VisionAnalyzer, create_provider_from_config

    tracker.reset()
    pdf_path = Path(file)

    try:
        cfg = _load_pipeline_config(env, config)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    provider = create_provider_from_config(cfg)
    if provider is None:
        console.print(
            "[bold red]Error:[/bold red] No vision provider available. "
            "Set GEMINI_API_KEY or configure api.llm_backend in config.yaml."
        )
        sys.exit(1)

    backend = cfg.api.llm_backend
    model = cfg.api.gemini_model if backend == "gemini" else cfg.api.vision_model
    console.print(
        f"[bold cyan]Vision analysis[/bold cyan] of [bold]{pdf_path.name}[/bold]\n"
        f"  Provider: {backend}  Model: {model}\n"
    )

    analyzer = VisionAnalyzer(provider)

    try:
        if analyze == "reading-order":
            result = asyncio.run(analyzer.analyze_reading_order(pdf_path))
        elif analyze == "contrast":
            result = asyncio.run(analyzer.analyze_contrast(pdf_path))
        else:
            result = asyncio.run(analyzer.analyze_all(pdf_path))
    except Exception as exc:
        console.print(f"[bold red]Vision analysis failed:[/bold red] {exc}")
        sys.exit(1)

    if as_json:
        import json as json_mod

        data = {
            "file": str(pdf_path),
            "provider": backend,
            "model": model,
            "reading_order_issues": [
                {
                    "page": i.page,
                    "severity": i.severity,
                    "description": i.description,
                    "suggestion": i.suggestion,
                }
                for i in result.reading_order_issues
            ],
            "contrast_issues": [
                {
                    "page": i.page,
                    "description": i.description,
                    "location": i.location,
                }
                for i in result.contrast_issues
            ],
        }
        console.print_json(json_mod.dumps(data))
        return

    # Reading order results.
    if analyze in ("reading-order", "all") and hasattr(result, "reading_order_issues"):
        ro_issues = result.reading_order_issues
        if ro_issues:
            console.print(f"[bold]Reading Order[/bold] — {len(ro_issues)} issue(s)\n")
            for issue in ro_issues:
                sev_style = "red" if issue.severity == "error" else "yellow"
                console.print(
                    f"  [{sev_style}]{issue.severity.upper()}[/{sev_style}] "
                    f"Page {issue.page}: {issue.description}"
                )
                if issue.suggestion:
                    console.print(f"         [dim]Fix: {issue.suggestion}[/dim]")
        else:
            console.print("[bold]Reading Order[/bold] — [green]No issues found[/green]\n")

    # Contrast results.
    if analyze in ("contrast", "all") and hasattr(result, "contrast_issues"):
        contrast_issues = result.contrast_issues
        if contrast_issues:
            console.print(f"\n[bold]Color Contrast[/bold] — {len(contrast_issues)} issue(s)\n")
            for issue in contrast_issues:
                loc = f" ({issue.location})" if issue.location else ""
                console.print(
                    f"  [red]FAIL[/red] Page {issue.page}{loc}: {issue.description}"
                )
        else:
            console.print("\n[bold]Color Contrast[/bold] — [green]No issues found[/green]")

    _print_token_usage()


# ---------------------------------------------------------------------------
# contrast
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None, help="Output path (default: <file>_contrast.pdf).")
@click.option("--level", type=click.Choice(["AA", "AAA"], case_sensitive=False), default="AA")
@click.option("--scan-only", is_flag=True, help="Detect issues without fixing.")
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", "config_path", default=None, help="Path to config.yaml file.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def contrast(
    file: str,
    output: str | None,
    level: str,
    scan_only: bool,
    env: str | None,
    config_path: str | None,
    as_json: bool,
) -> None:
    """Scan and fix color contrast issues in a PDF (WCAG 1.4.3 / 1.4.6 / 1.4.11).

    Uses AI vision to detect contrast violations, then programmatically
    fixes text colors, image contrast, and graphic elements.
    """
    import shutil

    tracker.reset()
    pdf_path = Path(file)
    output_path = Path(output) if output else pdf_path.parent / f"{pdf_path.stem}_contrast.pdf"

    try:
        cfg = _load_pipeline_config(env, config_path)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    # Create LLM client
    llm = _create_llm_client(cfg)

    async def _run():
        await llm.start()
        try:
            from project_remedy.contrast import ContrastDetector, ContrastRemediator

            if scan_only:
                detector = ContrastDetector(llm, dpi=150)
                issues = await detector.detect_document(str(pdf_path), level=level)

                if as_json:
                    import json as json_mod
                    data = {
                        "file": str(pdf_path),
                        "level": level,
                        "total_issues": len(issues),
                        "issues": [i.model_dump() for i in issues],
                    }
                    console.print_json(json_mod.dumps(data, default=str))
                else:
                    if issues:
                        console.print(f"\n[bold]Contrast Scan[/bold] — {len(issues)} issue(s)\n")
                        for issue in issues:
                            sev = "[red]FAIL[/red]"
                            loc = f" ({issue.text_content[:40]})" if issue.text_content else ""
                            console.print(
                                f"  {sev} Page {issue.page_index + 1}: "
                                f"{issue.issue_type.value}{loc} — "
                                f"ratio {issue.contrast_ratio}:1 "
                                f"(needs {issue.required_ratio}:1)"
                            )
                    else:
                        console.print("[bold green]No contrast issues found.[/bold green]")
            else:
                # Copy input to output before modifying
                shutil.copy2(pdf_path, output_path)

                remediator = ContrastRemediator(llm, dpi=150)
                analysis = await remediator.remediate_document(
                    str(pdf_path), str(output_path), level=level,
                )

                if as_json:
                    import json as json_mod
                    console.print_json(json_mod.dumps(analysis.model_dump(), default=str))
                else:
                    console.print(
                        f"\n[bold]Contrast Remediation[/bold] — {pdf_path.name}\n"
                        f"  Total issues: {analysis.total_issues}\n"
                        f"  Fixed: [green]{analysis.issues_fixed}[/green]\n"
                        f"  Remaining: [yellow]{analysis.issues_remaining}[/yellow]\n"
                        f"  Output: [bold]{output_path}[/bold]"
                    )
        finally:
            await llm.close()

    asyncio.run(_run())
    _print_token_usage()


# ---------------------------------------------------------------------------
# screen-reader
# ---------------------------------------------------------------------------


@pdf_group.command("screen-reader")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--annotated", "-a", is_flag=True, help="Show tag types and depth alongside text.")
@click.option("--validate", "-v", is_flag=True, help="Run structural validation checks.")
@click.option("--json", "as_json", is_flag=True, help="Output validation results as JSON.")
@click.option("--page", "-p", default=None, type=int, help="Show specific page only (1-based).")
def screen_reader(
    file: str,
    annotated: bool,
    validate: bool,
    as_json: bool,
    page: int | None,
) -> None:
    """Simulate screen reader output from the PDF tag tree.

    Shows what NVDA/VoiceOver would read, in order, by walking the
    /StructTreeRoot. With --validate, checks for structural issues
    that degrade the screen reader experience.
    """
    from project_remedy.tag_tree_reader import (
        read_tag_tree,
        validate_tag_tree,
    )

    pdf_path = Path(file)

    if validate or as_json:
        result = validate_tag_tree(pdf_path)

        if as_json:
            import json as json_mod

            data = {
                "file": str(result.file_path),
                "passed": result.passed,
                "errors": result.error_count,
                "warnings": result.warning_count,
                "tag_count": len(result.tag_tree.nodes),
                "page_count": result.tag_tree.page_count,
                "issues": [
                    {
                        "rule_id": i.rule_id,
                        "severity": i.severity.value,
                        "page": i.page,
                        "element": i.element,
                        "description": i.description,
                        "suggestion": i.suggestion,
                    }
                    for i in result.issues
                ],
            }
            console.print_json(json_mod.dumps(data))
            return

        # Rich formatted output.
        status = "[bold green]PASS[/bold green]" if result.passed else "[bold red]FAIL[/bold red]"
        console.print(
            f"\n{status} [bold]{pdf_path.name}[/bold]  "
            f"({len(result.tag_tree.nodes)} tags, {result.tag_tree.page_count} pages)"
        )
        console.print(
            f"  Errors: [red]{result.error_count}[/red]  "
            f"Warnings: [yellow]{result.warning_count}[/yellow]"
        )

        if result.issues:
            console.print()
            issue_table = Table(show_header=True, header_style="bold")
            issue_table.add_column("Sev", width=3)
            issue_table.add_column("Page", width=5)
            issue_table.add_column("Element", width=10)
            issue_table.add_column("Issue")
            issue_table.add_column("Fix", style="dim")

            for issue in result.issues:
                sev_style = {"error": "red", "warning": "yellow", "info": "dim"}
                marker = {"error": "E", "warning": "W", "info": "I"}
                page_str = str(issue.page + 1) if issue.page >= 0 else "doc"
                issue_table.add_row(
                    f"[{sev_style[issue.severity.value]}]{marker[issue.severity.value]}[/{sev_style[issue.severity.value]}]",
                    page_str,
                    issue.element,
                    issue.description,
                    issue.suggestion,
                )
            console.print(issue_table)
        return

    # Reading order output.
    report = read_tag_tree(pdf_path)
    if not report.has_structure_tree:
        console.print("[bold red]No structure tree — PDF is invisible to screen readers[/bold red]")
        sys.exit(1)

    if page is not None:
        by_page = report.nodes_by_page()
        page_idx = page - 1
        if page_idx not in by_page:
            console.print(f"[red]Page {page} has no tagged content[/red]")
            sys.exit(1)
        nodes = by_page[page_idx]
        console.print(f"\n[bold]Page {page} — Screen Reader Output[/bold]\n")
        for node in nodes:
            indent = "  " * (node.depth - 1)
            if annotated:
                content = node.alt_text or node.text or ""
                preview = content[:100].replace("\n", " ")
                if preview:
                    console.print(f"{indent}[cyan]<{node.tag}>[/cyan] {preview}")
                else:
                    console.print(f"{indent}[cyan]<{node.tag}>[/cyan]")
            else:
                if node.alt_text:
                    console.print(f"{indent}[{node.tag}: {node.alt_text}]")
                elif node.text:
                    console.print(f"{indent}{node.text}")
    else:
        if annotated:
            console.print(Panel(report.reading_order_annotated, title="Annotated Reading Order"))
        else:
            console.print(Panel(report.reading_order_text, title="Screen Reader Output"))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@pdf_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None, help="Output directory (default: ./compliance/).")
@click.option("--original", default=None, help="Path to original source file (for before/after).")
@click.option("--campus", default=None, help="Campus name for branding.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON report to stdout.")
def report(
    file: str,
    output: str | None,
    original: str | None,
    campus: str | None,
    as_json: bool,
) -> None:
    """Generate an accessibility compliance report for a PDF.

    Shows before/after comparison, WCAG 2.1 AA mapping, screen reader
    validation, and overall conformance determination.
    """
    from project_remedy.compliance_report import generate_document_report

    pdf_path = Path(file)
    output_dir = Path(output) if output else Path("compliance")
    original_path = Path(original) if original else pdf_path

    doc_report = generate_document_report(
        original_path=original_path,
        remediated_path=pdf_path,
        output_dir=output_dir,
        campus_name=campus or "",
    )

    if as_json:
        import json as json_mod
        console.print_json(json_mod.dumps(doc_report.to_dict(), default=str))
    else:
        conf_color = {
            "Conformant": "green",
            "Partially Conformant": "yellow",
            "Not Conformant": "red",
        }.get(doc_report.conformance, "white")

        console.print(f"\n[bold {conf_color}]{doc_report.conformance}[/bold {conf_color}]"
                       f" — {pdf_path.name}")
        console.print(f"  Checks: {doc_report.passed_checks}/{len(doc_report.check_results)} passed")
        console.print(f"  WCAG:   {doc_report.wcag_pass_count}/{len(doc_report.wcag_results)} criteria met")
        console.print(f"  SR:     {doc_report.sr_error_count} errors, {doc_report.sr_warning_count} warnings")
        if doc_report.verapdf_checked:
            verapdf_status = "PASS" if doc_report.verapdf_passed else "FAIL"
            console.print(
                f"  veraPDF: {verapdf_status} ({doc_report.verapdf_violation_count} violations)"
            )
        else:
            console.print("  veraPDF: unavailable")
        console.print(f"  Tags:   {doc_report.tag_count}")


# ---------------------------------------------------------------------------
# benchmark-manifest
# ---------------------------------------------------------------------------


@pdf_group.command("benchmark-manifest")
@click.option(
    "--exports-root",
    default="exports",
    show_default=True,
    help="Root directory containing <campus>/remediated-pdfs folders.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output JSON path for the manifest.",
)
def benchmark_manifest(exports_root: str, output: str | None) -> None:
    """Build a representative benchmark manifest from remediated export PDFs."""
    root = Path(exports_root)
    manifest = build_representative_corpus_manifest(root)
    output_path = Path(output) if output else default_manifest_path(Path.cwd())
    save_corpus_manifest(manifest, output_path)

    console.print(f"[bold green]Saved benchmark manifest[/bold green] to {output_path}")
    by_layout: dict[str, int] = {}
    for sample in manifest.samples:
        by_layout[sample.layout_class] = by_layout.get(sample.layout_class, 0) + 1

    table = Table(title="Representative Benchmark Manifest")
    table.add_column("Layout")
    table.add_column("Samples", justify="right")
    for layout_class, count in sorted(by_layout.items()):
        table.add_row(layout_class, str(count))
    console.print(table)


# ---------------------------------------------------------------------------
# benchmark-ocr
# ---------------------------------------------------------------------------


@pdf_group.command("benchmark-ocr")
@click.option(
    "--manifest",
    default=None,
    help="Path to a benchmark manifest JSON. Defaults to benchmarks/representative_pdf_manifest.json.",
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON to stdout.")
def benchmark_ocr(manifest: str | None, as_json: bool) -> None:
    """Run OCR benchmarks on the representative PDF corpus manifest."""
    manifest_path = Path(manifest) if manifest else default_manifest_path(Path.cwd())
    corpus_manifest = load_corpus_manifest(manifest_path)
    run = run_ocr_benchmark(corpus_manifest)

    if as_json:
        import json as json_mod

        console.print_json(
            json_mod.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "sample_count": run.sample_count,
                    "providers": run.providers,
                    "summaries": [asdict(summary) for summary in run.summaries],
                },
                default=str,
            )
        )
        return

    console.print(f"[bold]OCR benchmark manifest:[/bold] {manifest_path}")
    console.print(f"[bold]Samples:[/bold] {run.sample_count}")
    table = Table(title="OCR Benchmark Summary")
    table.add_column("Provider")
    table.add_column("Samples", justify="right")
    table.add_column("Mean Blocks", justify="right")
    table.add_column("Mean Chars", justify="right")
    table.add_column("Token Overlap", justify="right")
    table.add_column("Headings", justify="right")
    table.add_column("Table Lines", justify="right")
    table.add_column("Empty", justify="right")
    table.add_column("Failures", justify="right")
    for summary in run.summaries:
        table.add_row(
            summary.provider,
            str(summary.sample_count),
            f"{summary.mean_block_count:.1f}",
            f"{summary.mean_markdown_length:.1f}",
            f"{summary.mean_token_overlap:.2f}",
            f"{summary.mean_heading_count:.1f}",
            f"{summary.mean_table_markers:.1f}",
            str(summary.empty_output_count),
            str(len(summary.failures)),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# benchmark-gemini-tier
# ---------------------------------------------------------------------------


@pdf_group.command("benchmark-gemini-tier")
@click.option(
    "--manifest",
    default=None,
    help="Path to a benchmark manifest JSON. Defaults to benchmarks/representative_pdf_manifest.json.",
)
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option(
    "--sample-count",
    default=5,
    show_default=True,
    help="Maximum number of representative documents to run per model.",
)
@click.option(
    "--file",
    "files",
    multiple=True,
    help="Explicit PDF file(s) to include first in the benchmark sample set.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output JSON path for the Gemini tier benchmark artifact.",
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON to stdout.")
def benchmark_gemini_tier(
    manifest: str | None,
    env: str | None,
    config: str | None,
    sample_count: int,
    files: tuple[str, ...],
    output: str | None,
    as_json: bool,
) -> None:
    """Run the Gemini 3 Flash vs Flash-Lite benchmark with Pro escalation."""
    manifest_path = Path(manifest) if manifest else default_manifest_path(Path.cwd())
    corpus_manifest = load_corpus_manifest(manifest_path)
    cfg = _load_pipeline_config(env, config)
    if not cfg.api.gemini_api_key:
        raise click.ClickException(
            "Gemini benchmarking requires GEMINI_API_KEY in config or environment."
        )

    timestamp = re.sub(r"[^0-9]", "", datetime.now().isoformat())[:14]
    output_path = Path(output) if output else (
        Path.cwd() / "benchmarks" / f"gemini_tier_benchmark_{timestamp}.json"
    )
    run = run_gemini_tier_benchmark(
        corpus_manifest,
        config=cfg,
        manifest_path=manifest_path,
        output_root=output_path.with_suffix(""),
        sample_count=sample_count,
        explicit_files=[Path(file) for file in files],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")

    if as_json:
        console.print_json(json.dumps(run.to_dict(), default=str))
        return

    console.print(f"[bold green]Saved Gemini tier benchmark[/bold green] to {output_path}")
    console.print(f"[bold]Representative samples:[/bold] {run.sample_count}")
    table = Table(title="Gemini Tier Benchmark Summary")
    table.add_column("Model")
    table.add_column("HTML Pass", justify="right")
    table.add_column("PDF Pass", justify="right")
    table.add_column("Severe", justify="right")
    table.add_column("Escalations", justify="right")
    table.add_column("Empty", justify="right")
    table.add_column("Median Latency", justify="right")
    table.add_column("Cost / Doc", justify="right")
    for summary in run.summaries:
        table.add_row(
            summary.model,
            f"{summary.passed_count}/{summary.document_count}",
            f"{summary.pdf_passed_count}/{summary.document_count}",
            str(summary.severe_regressions),
            str(summary.escalation_count),
            str(summary.empty_output_count),
            f"{summary.median_latency_seconds:.1f}s",
            f"${summary.cost_per_document:.2f}",
        )
    console.print(table)

    detail = Table(title="Gemini Tier Benchmark Samples")
    detail.add_column("Model")
    detail.add_column("Layout")
    detail.add_column("File")
    detail.add_column("HTML")
    detail.add_column("PDF")
    detail.add_column("Esc")
    detail.add_column("Latency", justify="right")
    for entry in run.entries:
        pdf_status = "n/a" if entry.pdf_passed is None else ("pass" if entry.pdf_passed else "fail")
        detail.add_row(
            entry.model,
            entry.layout_class,
            entry.source_pdf_path.name,
            "pass" if entry.html_passed else "fail",
            pdf_status,
            str(entry.escalation_count),
            f"{entry.latency_seconds:.1f}s",
        )
    console.print(detail)
    console.print(
        Panel(
            (
                f"Recommended Tier 1 model: [bold]{run.decision.recommended_model}[/bold]\n"
                f"{run.decision.rationale}\n"
                f"Flash pass rate: {run.decision.flash_pass_rate:.1%}\n"
                f"Flash-Lite pass rate: {run.decision.flash_lite_pass_rate:.1%}"
            ),
            title="Decision",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# acceptance-sweep
# ---------------------------------------------------------------------------


@pdf_group.command("acceptance-sweep")
@click.option(
    "--manifest",
    default=None,
    help="Path to a benchmark manifest JSON. Defaults to benchmarks/representative_pdf_manifest.json.",
)
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON to stdout.")
def acceptance_sweep(
    manifest: str | None,
    env: str | None,
    config: str | None,
    as_json: bool,
) -> None:
    """Run the composite PDF acceptance gate over the representative manifest."""
    manifest_path = Path(manifest) if manifest else default_manifest_path(Path.cwd())
    corpus_manifest = load_corpus_manifest(manifest_path)

    try:
        cfg = _load_pipeline_config(env, config)
    except Exception:
        cfg = None

    sweep = run_acceptance_sweep(corpus_manifest, config=cfg)

    if as_json:
        import json as json_mod

        console.print_json(
            json_mod.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "document_count": sweep.document_count,
                    "passed_count": sweep.passed_count,
                    "entries": [entry.to_dict() for entry in sweep.entries],
                },
                default=str,
            )
        )
        return

    console.print(f"[bold]Acceptance sweep manifest:[/bold] {manifest_path}")
    console.print(
        f"[bold]{sweep.passed_count}/{sweep.document_count} representative documents passed composite acceptance[/bold]"
    )
    table = Table(title="Acceptance Sweep")
    table.add_column("Campus")
    table.add_column("Layout")
    table.add_column("File")
    table.add_column("Pass")
    table.add_column("Checker", justify="right")
    table.add_column("SR", justify="right")
    table.add_column("veraPDF")
    for entry in sweep.entries:
        verapdf_status = "n/a"
        if entry.verapdf_checked:
            verapdf_status = "pass" if entry.verapdf_passed else "fail"
        table.add_row(
            entry.campus,
            entry.layout_class,
            entry.pdf_path.name,
            "yes" if entry.passed else "no",
            str(entry.checker_failures),
            str(entry.screen_reader_errors),
            verapdf_status,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# serving-benchmark-init
# ---------------------------------------------------------------------------


@pdf_group.command("serving-benchmark-init")
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output JSON path for the serving benchmark candidate config.",
)
def serving_benchmark_init(output: str | None) -> None:
    """Write a default Apple-Silicon serving benchmark candidate config."""
    config = default_serving_benchmark_config()
    output_path = Path(output) if output else default_serving_benchmark_config_path(Path.cwd())
    save_serving_benchmark_config(config, output_path)

    console.print(f"[bold green]Saved serving benchmark config[/bold green] to {output_path}")
    console.print(
        "[dim]Candidates: "
        + ", ".join(candidate.name for candidate in config.candidates)
        + "[/dim]"
    )
    table = Table(title="Serving Benchmark Candidates")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Role")
    table.add_column("Base URL")
    table.add_column("Model")
    for candidate in config.candidates:
        table.add_row(
            candidate.name,
            candidate.kind,
            candidate.role,
            candidate.base_url,
            candidate.model,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# serving-benchmark-probe
# ---------------------------------------------------------------------------


@pdf_group.command("serving-benchmark-probe")
@click.option(
    "--config-path",
    "config_path",
    default=None,
    help="Path to the serving benchmark candidate JSON. Defaults to benchmarks/serving_stack_candidates.json.",
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON to stdout.")
def serving_benchmark_probe(config_path: str | None, as_json: bool) -> None:
    """Run compatibility probes for Apple-Silicon serving candidates."""
    path = Path(config_path) if config_path else default_serving_benchmark_config_path(Path.cwd())
    config = load_serving_benchmark_config(path)
    if config.manifest_path:
        manifest = load_corpus_manifest(config.manifest_path)
        results = run_serving_compatibility_probe_for_manifest(config, manifest)
    else:
        results = run_serving_compatibility_probe(config)

    if as_json:
        import json as json_mod

        console.print_json(
            json_mod.dumps(
                {
                    "config_path": str(path),
                    "results": [result.to_dict() for result in results],
                },
                default=str,
            )
        )
        return

    console.print(f"[bold]Serving benchmark config:[/bold] {path}")
    console.print(
        "[dim]Candidates: "
        + ", ".join(result.candidate for result in results)
        + "[/dim]"
    )
    table = Table(title="Serving Compatibility Probe")
    table.add_column("Candidate")
    table.add_column("Kind")
    table.add_column("Pass")
    table.add_column("Models")
    table.add_column("Expected")
    table.add_column("Multimodal")
    table.add_column("Response")
    table.add_column("Error")
    for result in results:
        table.add_row(
            result.candidate,
            result.kind,
            "yes" if result.passed else "no",
            "yes" if result.models_ok else "no",
            "yes" if result.expected_model_available else "no",
            "yes" if result.multimodal_ok else "no",
            "yes" if result.nonempty_response else "no",
            result.error[:60],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# serving-benchmark-run
# ---------------------------------------------------------------------------


@pdf_group.command("serving-benchmark-run")
@click.option(
    "--config-path",
    "config_path",
    default=None,
    help="Path to the serving benchmark candidate JSON. Defaults to benchmarks/serving_stack_candidates.json.",
)
@click.option(
    "--manifest",
    default=None,
    help="Path to a benchmark manifest JSON. Defaults to benchmarks/representative_pdf_manifest.json.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output JSON path for the full serving benchmark run artifact.",
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON to stdout.")
def serving_benchmark_run(
    config_path: str | None,
    manifest: str | None,
    output: str | None,
    as_json: bool,
) -> None:
    """Run the Apple-Silicon serving stack bake-off."""
    config_file = Path(config_path) if config_path else default_serving_benchmark_config_path(Path.cwd())
    manifest_path = Path(manifest) if manifest else default_manifest_path(Path.cwd())
    config = load_serving_benchmark_config(config_file)
    corpus_manifest = load_corpus_manifest(manifest_path)

    run = run_serving_bakeoff(
        config,
        corpus_manifest,
        config_path=config_file,
    )

    output_path = Path(output) if output else (
        Path.cwd()
        / "benchmarks"
        / f"serving_stack_benchmark_{run.generated_at.replace(':', '').replace('-', '')}.json"
    )
    save_serving_benchmark_run(run, output_path)

    if as_json:
        import json as json_mod

        console.print_json(json_mod.dumps(run.to_dict(), default=str))
        return

    console.print(f"[bold green]Saved serving benchmark run[/bold green] to {output_path}")

    compat = Table(title="Serving Compatibility")
    compat.add_column("Candidate")
    compat.add_column("Pass")
    compat.add_column("Error")
    for result in run.compatibility:
        compat.add_row(
            result.candidate,
            "yes" if result.passed else "no",
            result.error[:80],
        )
    console.print(compat)

    quality = Table(title="Serving Quality Summary")
    quality.add_column("Candidate")
    quality.add_column("Profile")
    quality.add_column("Success", justify="right")
    quality.add_column("Mean Latency", justify="right")
    quality.add_column("Mean Overlap", justify="right")
    for summary in run.quality_summaries:
        quality.add_row(
            summary.candidate,
            summary.profile,
            f"{summary.success_count}/{summary.sample_count}",
            f"{summary.mean_latency_ms:.1f} ms",
            f"{summary.mean_token_overlap_vs_baseline:.2f}",
        )
    console.print(quality)

    throughput = Table(title="Serving Throughput")
    throughput.add_column("Candidate")
    throughput.add_column("Profile")
    throughput.add_column("Scenario")
    throughput.add_column("Concurrency", justify="right")
    throughput.add_column("Req/s", justify="right")
    throughput.add_column("p50", justify="right")
    throughput.add_column("p95", justify="right")
    for point in run.throughput_points:
        throughput.add_row(
            point.candidate,
            point.profile,
            point.scenario,
            str(point.concurrency),
            f"{point.requests_per_second:.2f}",
            f"{point.p50_latency_ms:.1f}",
            f"{point.p95_latency_ms:.1f}",
        )
    console.print(throughput)


def _create_llm_client(cfg):
    """Create and return the appropriate LLM client based on config."""
    import os
    backend = os.environ.get("LLM_BACKEND", cfg.api.llm_backend)
    if backend == "zai":
        from project_remedy.zai_client import ZaiClient
        return ZaiClient()
    elif backend == "gemini":
        from project_remedy.gemini_client import GeminiClient
        return GeminiClient(cfg)
    else:
        from project_remedy.ollama_client import OllamaClient
        return OllamaClient(cfg)


# ---------------------------------------------------------------------------
# vision-plan command (Meta-Harness integration)
# ---------------------------------------------------------------------------


@pdf_group.command("vision-plan")
@click.option(
    "--pdf", "pdf_file", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input PDF file path.",
)
@click.option(
    "--harness", "harness_path", default=None,
    type=click.Path(exists=True),
    help="External harness.py (loads VisionPlannerHarness via importlib).",
)
@click.option("--output", default=None, type=click.Path(), help="Output trace JSON path.")
@click.option("--model", default=None, help="Override vision/planning model.")
@click.option(
    "--agent", "use_agent", is_flag=True, default=False,
    help="Use agentic loop (Tier 3) instead of grounder/planner/executor pipeline.",
)
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def vision_plan(
    pdf_file: str,
    harness_path: str | None,
    output: str | None,
    model: str | None,
    use_agent: bool,
    env: str | None,
    config_path: str | None,
) -> None:
    """Run vision-planner remediation on a PDF.

    Executes grounder (vision) -> planner (thinking) -> executor pipeline.
    Writes a JSON trace file with full diagnostics for Meta-Harness scoring.
    """
    import asyncio
    import importlib.util

    from project_remedy.vision_planner.pipeline import run_vision_plan

    pdf_path = Path(pdf_file)
    output_path = Path(output) if output else None

    # Load harness (skipped when --agent is set)
    harness = None
    if not use_agent:
        if harness_path:
            spec = importlib.util.spec_from_file_location("external_harness", harness_path)
            if spec is None or spec.loader is None:
                console.print(f"[bold red]Error:[/bold red] Cannot load harness from {harness_path}")
                sys.exit(1)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            harness_cls = getattr(module, "VisionPlannerHarness", None)
            if harness_cls is None:
                console.print("[bold red]Error:[/bold red] VisionPlannerHarness class not found in harness")
                sys.exit(1)
            harness = harness_cls()
        else:
            from project_remedy.vision_planner.harness import VisionPlannerHarness
            harness = VisionPlannerHarness()

    # Load config and create client
    try:
        cfg = _load_pipeline_config(env, config_path)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    if use_agent:
        from project_remedy.zai_client import ZaiClient
        client = ZaiClient()
    else:
        client = _create_llm_client(cfg)

    async def _run() -> dict:
        await client.start()
        try:
            if use_agent:
                from project_remedy.vision_planner.agent_loop import run_agent_loop, AgentLoopConfig
                import json as _json
                import os as _os

                # Load agent config from env var (set by meta-harness evaluator)
                agent_config_path = _os.environ.get("AGENT_CONFIG_PATH")
                loop_config = None
                system_prompt = None
                if agent_config_path and Path(agent_config_path).exists():
                    ac = _json.loads(Path(agent_config_path).read_text())
                    system_prompt = ac.get("system_prompt")
                    loop_config = AgentLoopConfig(
                        max_fix_attempts=ac.get("max_fix_attempts", 8),
                        max_fix_attempts_per_family=ac.get("max_fix_attempts_per_family", 4),
                        max_tool_rounds=ac.get("max_tool_rounds", 30),
                        allow_unsafe_artifactize=ac.get("allow_unsafe_artifactize", True),
                        temperature=ac.get("temperature", 0.0),
                    )

                kwargs: dict = {
                    "pdf_path": pdf_path,
                    "client": client,
                    "output_path": output_path,
                    "model": model,
                }
                if loop_config:
                    kwargs["loop_config"] = loop_config
                if system_prompt:
                    kwargs["system_prompt"] = system_prompt

                return await run_agent_loop(**kwargs)
            else:
                return await run_vision_plan(
                    pdf_path=pdf_path,
                    output_path=output_path,
                    harness=harness,
                    client=client,
                    model=model,
                    config=cfg,
                )
        finally:
            if hasattr(client, 'close'):
                import asyncio as _aio
                close_result = client.close()
                if _aio.iscoroutine(close_result):
                    await close_result

    trace = asyncio.run(_run())

    # Write trace JSON to output path (for meta-harness scoring)
    if output_path and isinstance(trace, dict):
        import json as _json2
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                _json2.dump(trace, f, indent=2, default=str)
        except Exception as e:
            console.print(f"  [yellow]Warning:[/yellow] Could not write trace: {e}")

    # Print summary
    console.print(f"\n[bold cyan]Vision-plan result[/bold cyan] for [bold]{pdf_path.name}[/bold]")
    console.print(f"  Passed:             {trace.get('passed', False)}")
    if use_agent:
        init = trace.get("initial_verapdf", {})
        final = trace.get("final_verapdf", {})
        console.print(f"  Status:             {trace.get('status', 'unknown')}")
        console.print(f"  Violations before:  {init.get('structural_count', init.get('total', '?'))}")
        console.print(f"  Violations after:   {final.get('structural_count', final.get('total', '?'))}")
        console.print(f"  Attempts used:      {trace.get('attempts_used', 0)}")
        console.print(f"  Reason:             {trace.get('reason', '')}")
    else:
        console.print(f"  Violations before:  {trace.get('violations_before', '?')}")
        console.print(f"  Violations after:   {trace.get('violations_after', '?')}")
        console.print(f"  Operations:         {len(trace.get('plan', {}).get('operations', []))}")
        console.print(f"  Elapsed:            {trace.get('elapsed_seconds', 0):.1f}s")
    if trace.get("error"):
        console.print(f"  [bold red]Error:[/bold red] {trace['error']}")
    if output_path:
        console.print(f"  Trace written to:   {output_path}")
