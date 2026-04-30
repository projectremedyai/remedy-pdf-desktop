"""Click CLI entry point for the LACCD ADA Document Remediation Pipeline."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from project_remedy import __version__
from project_remedy.cli_pdf import pdf_group
from project_remedy.config import CampusConfig, PipelineConfig, load_config
from project_remedy.logging_config import setup_logging
from project_remedy.models import JobStatus

console = Console()
logger = logging.getLogger(__name__)

# All 9 LACCD campus codes
ALL_CAMPUS_CODES = ["LACC", "ELAC", "LAMC", "LAPC", "LAHC", "LASC", "LATTC", "LAVC", "WLAC"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cfg(env: str | None, config: str | None) -> PipelineConfig:
    """Load configuration from env and yaml paths."""
    env_path = Path(env) if env else None
    yaml_path = Path(config) if config else None
    return load_config(env_path=env_path, yaml_path=yaml_path)


def _resolve_campus(cfg: PipelineConfig, campus_code: str | None) -> CampusConfig | None:
    """Resolve a campus code to a CampusConfig, or None for all."""
    if not campus_code or campus_code.lower() == "all":
        return None
    code = campus_code.upper()
    try:
        return cfg.get_campus(code)
    except ValueError:
        console.print(
            f"[bold red]Error:[/bold red] Unknown campus code '{code}'. "
            f"Available: {', '.join(ALL_CAMPUS_CODES)}"
        )
        sys.exit(1)


def _with_html_override(cfg: PipelineConfig, html: bool | None) -> PipelineConfig:
    if html is None:
        return cfg
    return replace(
        cfg,
        processing=replace(cfg.processing, html_workflow_enabled=html),
    )


def _get_campus_list(cfg: PipelineConfig, campus_code: str | None) -> list[CampusConfig]:
    """Return a list of campuses to process."""
    if not campus_code or campus_code.lower() == "all":
        return list(cfg.campuses) if cfg.campuses else []
    campus = _resolve_campus(cfg, campus_code)
    return [campus] if campus else []


def _run_async(coro) -> None:
    """Bridge a synchronous Click command to an async coroutine."""
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline error:[/bold red] {exc}")
        logger.exception("Pipeline failed with unhandled exception.")
        sys.exit(1)


def _print_report(report) -> None:
    """Print a PipelineReport as a Rich panel."""
    lines = report.summary_lines()
    body = "\n".join(lines)

    if report.errors:
        body += "\n\n[bold red]Errors:[/bold red]"
        for err in report.errors[:20]:
            body += f"\n  - {err[:120]}"
        if len(report.errors) > 20:
            body += f"\n  ... and {len(report.errors) - 20} more"

    console.print(Panel(body, title="Pipeline Report", border_style="green"))


def _print_job_table(jobs, title: str = "Jobs") -> None:
    """Print a summary table of document jobs."""
    table = Table(title=title, show_lines=True)
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Campus", justify="center", max_width=6)
    table.add_column("URL", max_width=50)
    table.add_column("Type", justify="center")
    table.add_column("Status", justify="center")

    status_styles = {
        JobStatus.VALIDATED: "green",
        JobStatus.FAILED: "red",
        JobStatus.FLAGGED: "yellow",
        JobStatus.CONVERTED: "cyan",
        JobStatus.DOWNLOADED: "blue",
    }

    for job in jobs[:50]:
        style = status_styles.get(job.status, "")
        status_text = f"[{style}]{job.status.value}[/{style}]" if style else job.status.value
        file_type = job.file_type.value.upper() if job.file_type else "?"
        url_short = job.url[-50:] if len(job.url) > 50 else job.url
        table.add_row(job.id[:12], job.campus_id or "-", url_short, file_type, status_text)

    if len(jobs) > 50:
        table.add_row("...", "", f"({len(jobs) - 50} more)", "", "")

    console.print(table)
    console.print(f"Total: {len(jobs)} job(s)")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="remedy")
def cli() -> None:
    """Remedy PDF Desktop — Document Accessibility Remediation Pipeline.

    Crawl, remediate, and validate inaccessible documents
    from LACCD campus websites, defaulting to direct document remediation.

    Full end-to-end workflow: `remedy run` (fallback:
    `python -m project_remedy.cli run`).

    PDF-only reruns: use `python scripts/batch_remediate.py` for
    downloaded PDFs on disk and `python scripts/run_pdf_remediation.py`
    for pipeline-tracked PDF jobs only.
    """


cli.add_command(pdf_group)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--campus", default=None, help="Campus code (e.g., LAMC) or 'all'.")
def crawl(env: str | None, config: str | None, campus: str | None) -> None:
    """Stage 1-2: Crawl the site, discover and download documents."""

    async def _crawl() -> None:
        from project_remedy.pipeline import Pipeline

        try:
            cfg = _load_cfg(env, config)
        except Exception as exc:
            console.print(f"[bold red]Config error:[/bold red] {exc}")
            sys.exit(1)

        setup_logging(cfg.output.log_dir)
        campus_cfg = _resolve_campus(cfg, campus)

        start_url = campus_cfg.start_url if campus_cfg else cfg.crawl.start_url
        if not start_url:
            console.print("[bold red]Error:[/bold red] No start_url configured.")
            sys.exit(1)

        console.print(
            f"[bold green]Starting crawl[/bold green] from {start_url}"
            + (f" (campus={campus_cfg.code})" if campus_cfg else "")
        )

        pipeline = Pipeline(cfg, campus=campus_cfg)
        try:
            await pipeline.start()
            jobs = await pipeline.crawl()
            _print_job_table(jobs, "Crawl Results")
        finally:
            await pipeline.close()

    _run_async(_crawl())


@cli.command()
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--campus", default=None, help="Campus code (e.g., LAMC) or 'all'.")
@click.option("--html/--no-html", default=None, help="Opt into the HTML remediation workflow.")
def process(env: str | None, config: str | None, campus: str | None, html: bool | None) -> None:
    """Process downloaded documents.

    Default mode runs direct PDF remediation only. Use --html to opt into the
    extraction + HTML conversion workflow.
    """

    async def _process() -> None:
        from project_remedy.pipeline import Pipeline

        try:
            cfg = _load_cfg(env, config)
            cfg = _with_html_override(cfg, html)
        except Exception as exc:
            console.print(f"[bold red]Config error:[/bold red] {exc}")
            sys.exit(1)

        setup_logging(cfg.output.log_dir)
        campus_cfg = _resolve_campus(cfg, campus)

        pipeline = Pipeline(cfg, campus=campus_cfg)
        try:
            await pipeline.start()

            downloaded = await pipeline._db.get_jobs_by_status(
                JobStatus.DOWNLOADED,
                campus_id=campus_cfg.code if campus_cfg else None,
            )
            mode_label = "HTML workflow" if cfg.processing.html_workflow_enabled else "document workflow"
            console.print(
                f"[bold green]Processing[/bold green] {len(downloaded)} downloaded document(s) via {mode_label}."
            )

            jobs = await pipeline.process()
            _print_job_table(jobs, "Processing Results")
        finally:
            await pipeline.close()

    _run_async(_process())


@cli.command()
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--campus", default=None, help="Campus code (e.g., LAMC) or 'all'.")
@click.option("--html/--no-html", default=None, help="Opt into the HTML validation workflow.")
def validate(env: str | None, config: str | None, campus: str | None, html: bool | None) -> None:
    """Validate processed documents.

    In document mode, PDF acceptance is enforced during remediation and this
    command reports current PDF remediation statuses.
    """

    async def _validate() -> None:
        from project_remedy.pipeline import Pipeline

        try:
            cfg = _load_cfg(env, config)
            cfg = _with_html_override(cfg, html)
        except Exception as exc:
            console.print(f"[bold red]Config error:[/bold red] {exc}")
            sys.exit(1)

        setup_logging(cfg.output.log_dir)
        campus_cfg = _resolve_campus(cfg, campus)

        pipeline = Pipeline(cfg, campus=campus_cfg)
        try:
            await pipeline.start()

            if cfg.processing.html_workflow_enabled:
                queued = await pipeline._db.get_jobs_by_status(
                    JobStatus.CONVERTED,
                    campus_id=campus_cfg.code if campus_cfg else None,
                )
            else:
                queued = await pipeline._db.get_jobs_by_status(
                    JobStatus.PDF_REMEDIATED,
                    JobStatus.FLAGGED,
                    campus_id=campus_cfg.code if campus_cfg else None,
                )
            console.print(
                f"[bold green]Validating[/bold green] {len(queued)} document(s)."
            )

            jobs = await pipeline.validate_all()

            passed = sum(1 for j in jobs if j.status == JobStatus.VALIDATED)
            flagged = sum(1 for j in jobs if j.status == JobStatus.FLAGGED)
            console.print(
                f"Results: [green]{passed} passed[/green], "
                f"[yellow]{flagged} flagged[/yellow]"
            )
            _print_job_table(jobs, "Validation Results")
        finally:
            await pipeline.close()

    _run_async(_validate())


@cli.command()
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--campus", default=None, help="Campus code (e.g., LAMC) or 'all'.")
@click.option("--html/--no-html", default=None, help="Opt into the HTML remediation workflow.")
def run(env: str | None, config: str | None, campus: str | None, html: bool | None) -> None:
    """Run the full end-to-end pipeline.

    Default mode is direct document remediation. Use --html to opt into the
    legacy HTML remediation workflow.
    """

    async def _run_all() -> None:
        from project_remedy.pipeline import Pipeline

        try:
            cfg = _load_cfg(env, config)
            cfg = _with_html_override(cfg, html)
        except Exception as exc:
            console.print(f"[bold red]Config error:[/bold red] {exc}")
            sys.exit(1)

        setup_logging(cfg.output.log_dir)
        campus_cfg = _resolve_campus(cfg, campus)
        console.print(
            "[bold green]Starting full pipeline run.[/bold green]"
            + (f" Campus: {campus_cfg.code}" if campus_cfg else " All campuses")
            + (
                " [HTML workflow]"
                if cfg.processing.html_workflow_enabled
                else " [document workflow]"
            )
        )

        pipeline = Pipeline(cfg, campus=campus_cfg)
        try:
            await pipeline.start()
            report = await pipeline.run()
            _print_report(report)
        finally:
            await pipeline.close()

    _run_async(_run_all())


@cli.command()
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--campus", default=None, help="Campus code (e.g., LAMC) or 'all'.")
def status(env: str | None, config: str | None, campus: str | None) -> None:
    """Show current pipeline progress."""

    async def _status() -> None:
        from project_remedy.database import DatabaseManager

        try:
            cfg = _load_cfg(env, config)
        except Exception as exc:
            console.print(f"[bold red]Config error:[/bold red] {exc}")
            sys.exit(1)

        setup_logging(cfg.output.log_dir)
        campus_cfg = _resolve_campus(cfg, campus)
        campus_id = campus_cfg.code if campus_cfg else None

        db = DatabaseManager(cfg.output.db_path)
        try:
            await db.connect()
            stats = await db.get_stats(campus_id=campus_id)
            total = sum(stats.values())

            title = f"Pipeline Status — {campus_cfg.code}" if campus_cfg else "Pipeline Status"
            table = Table(title=title, show_lines=True)
            table.add_column("Status", style="bold")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right")

            for member in JobStatus:
                count = stats.get(member.value, 0)
                pct = f"{count / total * 100:.1f}%" if total else "0.0%"
                style = ""
                if member == JobStatus.VALIDATED:
                    style = "green"
                elif member == JobStatus.FAILED:
                    style = "red"
                elif member == JobStatus.FLAGGED:
                    style = "yellow"
                table.add_row(
                    f"[{style}]{member.value}[/{style}]" if style else member.value,
                    str(count),
                    pct,
                )

            table.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]", "100.0%")
            console.print(table)
        finally:
            await db.close()

    _run_async(_status())


@cli.command()
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--campus", default=None, help="Campus code (e.g., LAMC) or 'all'.")
def wave(env: str | None, config: str | None, campus: str | None) -> None:
    """Run WAVE API validation against deployed pages."""

    async def _wave() -> None:
        from project_remedy.pipeline import Pipeline

        try:
            cfg = _load_cfg(env, config)
        except Exception as exc:
            console.print(f"[bold red]Config error:[/bold red] {exc}")
            sys.exit(1)

        setup_logging(cfg.output.log_dir)

        if not cfg.validation.wave_api_key:
            console.print(
                "[bold red]Error:[/bold red] WAVE_API_KEY is not configured. "
                "Set it in .env or config.yaml."
            )
            sys.exit(1)

        campus_cfg = _resolve_campus(cfg, campus)

        pipeline = Pipeline(cfg, campus=campus_cfg)
        try:
            await pipeline.start()

            campus_id = campus_cfg.code if campus_cfg else None
            jobs = await pipeline._db.get_jobs_by_status(
                JobStatus.VALIDATED, JobStatus.FLAGGED,
                campus_id=campus_id,
            )

            if not jobs:
                console.print("[yellow]No deployed jobs to validate with WAVE.[/yellow]")
                return

            console.print(
                f"[bold green]Running WAVE validation[/bold green] on {len(jobs)} page(s)."
            )

            for job in jobs:
                if not job.url:
                    continue
                result = await pipeline._validator.validate_with_wave(job.url)
                status_str = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
                console.print(
                    f"  {job.campus_id or '-':>5} | {status_str} | "
                    f"{len(result.violations)} issue(s) | {job.url[-60:]}"
                )
        finally:
            await pipeline.close()

    _run_async(_wave())


@cli.command("mcp-server")
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
def mcp_server(env: str | None, config: str | None) -> None:
    """Start the MCP server for Drupal deployment (STDIO transport)."""

    async def _run() -> None:
        try:
            from project_remedy.mcp_server import run_server
        except ImportError:
            console.print(
                "[bold red]Error:[/bold red] The 'mcp' package is not installed.\n"
                "Install it with: [bold]pip install project-remedy[mcp][/bold]"
            )
            sys.exit(1)
        await run_server(config_path=config, env_path=env)

    _run_async(_run())


@cli.command("retry-failed")
@click.option("--env", default=None, help="Path to .env file.")
@click.option("--config", default=None, help="Path to config.yaml file.")
@click.option("--campus", default=None, help="Campus code (e.g., LAMC) or 'all'.")
def retry_failed(env: str | None, config: str | None, campus: str | None) -> None:
    """Reset failed jobs and reprocess them."""

    async def _retry() -> None:
        from project_remedy.pipeline import Pipeline

        try:
            cfg = _load_cfg(env, config)
        except Exception as exc:
            console.print(f"[bold red]Config error:[/bold red] {exc}")
            sys.exit(1)

        setup_logging(cfg.output.log_dir)
        campus_cfg = _resolve_campus(cfg, campus)

        pipeline = Pipeline(cfg, campus=campus_cfg)
        try:
            await pipeline.start()
            reset_jobs = await pipeline.retry_failed()

            if not reset_jobs:
                console.print("[green]No failed jobs to retry.[/green]")
                return

            console.print(
                f"[bold green]Reset {len(reset_jobs)} failed job(s) "
                f"to 'discovered'.[/bold green]"
            )
            _print_job_table(reset_jobs, "Reset Jobs")
        finally:
            await pipeline.close()

    _run_async(_retry())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
