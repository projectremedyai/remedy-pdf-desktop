"""Campus and district compliance dashboards.

Generates:
- Per-campus index.html with stats, WCAG breakdown, document table
- District-wide dashboard with cross-campus overview

Usage::

    from project_remedy.dashboard_generator import (
        generate_campus_dashboard,
        generate_district_dashboard,
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from project_remedy.compliance_report import (
    Conformance,
    DocumentReport,
    _human_size,
)


# ---------------------------------------------------------------------------
# Shared HTML fragments (Tailwind + Material Icons)
# ---------------------------------------------------------------------------

_HEAD_COMMON = """\
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<script>
    tailwind.config = {
        darkMode: "class",
        theme: {
            extend: {
                colors: {
                    navy: "#003D66",
                    blue_accent: "#005A97",
                    success: "#10B981",
                    warning: "#F59E0B",
                    error: "#EF4444",
                    neutral_muted: "#94A3B8",
                    bg_muted: "#F8FAFC"
                },
                fontFamily: {
                    "headline": ["Plus Jakarta Sans"],
                    "body": ["Inter"],
                    "label": ["Inter"]
                },
                borderRadius: {"DEFAULT": "0.5rem", "lg": "1rem", "xl": "1.5rem", "full": "9999px"},
            },
        },
    }
</script>
<style>
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
    .sticky-header th {
        position: sticky;
        top: 0;
        z-index: 10;
    }
</style>"""


def _top_nav(active: str = "campus") -> str:
    """Return the top navigation bar HTML.

    *active* is one of ``"district"`` or ``"campus"``; the matching link
    receives the white-underline treatment.
    """
    def _cls(name: str) -> str:
        if name == active:
            return "text-white border-b-2 border-white pb-1 font-sans antialiased text-sm font-medium"
        return "text-emerald-100/70 hover:text-white transition-colors font-sans antialiased text-sm font-medium"

    district_href = "../../district-dashboard.html" if active == "campus" else "#"

    return f"""\
<header class="bg-emerald-900 shadow-md border-b border-emerald-800 sticky top-0 z-50">
<nav class="flex justify-between items-center px-6 py-3 w-full max-w-[1200px] mx-auto">
<div class="flex items-center gap-6">
<span class="text-xl font-bold text-white tracking-tight font-sans antialiased">LACCD Accessibility Dashboard</span>
<div class="hidden md:flex gap-6">
<a class="{_cls('district')}" href="{district_href}">District</a>
<a class="{_cls('campus')}" href="#">Campus</a>
<a class="{_cls('documents')}" href="#documents">Documents</a>
<a class="{_cls('criteria')}" href="#wcag-criteria">Criteria</a>
</div>
</div>
</nav>
</header>"""


# ---------------------------------------------------------------------------
# Campus dashboard
# ---------------------------------------------------------------------------


def generate_campus_dashboard(
    reports: list[DocumentReport],
    output_path: Path,
    *,
    campus_name: str = "Campus",
    campus_code: str = "",
    brand_color: str = "#003366",
    duplicate_entries: list[dict] | None = None,
) -> dict:
    """Generate a campus compliance dashboard (index.html).

    Returns summary dict for district-level aggregation.
    """
    now = datetime.now(timezone.utc)
    total = len(reports)
    conformant = sum(1 for r in reports if r.conformance == Conformance.CONFORMANT)
    partial = sum(1 for r in reports if r.conformance == Conformance.PARTIALLY)
    not_conf = sum(1 for r in reports if r.conformance == Conformance.NOT_CONFORMANT)
    pct = round(conformant / total * 100, 1) if total > 0 else 0

    # Screen reader readability aggregation.
    readability_scores = [
        getattr(r, "screen_reader_readability", 0.0) or 0.0 for r in reports
    ]
    avg_readability = round(sum(readability_scores) / len(readability_scores), 1) if readability_scores else 0.0
    min_readability = round(min(readability_scores), 1) if readability_scores else 0.0
    max_readability = round(max(readability_scores), 1) if readability_scores else 0.0

    # Detailed breakdown of partially conformant PDFs.
    _FONT_RULES = {"7.21.7-1", "7.21.7-2", "7.21.4.1-1", "7.21.4.2-2", "7.21.8-1"}
    partial_font_only = 0
    partial_structural = 0
    partial_checker = 0
    for r in reports:
        if r.conformance != Conformance.PARTIALLY:
            continue
        has_checker_fail = any(c["status"] == "Failed" for c in r.check_results)
        has_real_verapdf = False
        if r.verapdf_checked and not r.verapdf_passed:
            for v in r.verapdf_violations:
                rule_id = str(v.get("id", ""))
                if not any(fr in rule_id for fr in _FONT_RULES):
                    has_real_verapdf = True
                    break
        if has_checker_fail:
            partial_checker += 1
        elif has_real_verapdf:
            partial_structural += 1
        else:
            partial_font_only += 1

    # Issue frequency.
    issue_counts: dict[str, int] = {}
    for report in reports:
        for check in report.check_results:
            if check["status"] == "Failed":
                issue_counts[check["rule_id"]] = issue_counts.get(check["rule_id"], 0) + 1
        for sr in report.sr_issues:
            if sr["severity"] == "error":
                issue_counts[sr["rule_id"]] = issue_counts.get(sr["rule_id"], 0) + 1
    top_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:15]

    # WCAG breakdown.
    wcag_stats: dict[str, dict] = {}
    for report in reports:
        for w in report.wcag_results:
            if w.criterion_id not in wcag_stats:
                wcag_stats[w.criterion_id] = {
                    "name": w.criterion_name, "level": w.level,
                    "pass": 0, "fail": 0, "na": 0,
                }
            if w.status == "PASS":
                wcag_stats[w.criterion_id]["pass"] += 1
            elif w.status == "FAIL":
                wcag_stats[w.criterion_id]["fail"] += 1
            else:
                wcag_stats[w.criterion_id]["na"] += 1

    # Document table rows (Tailwind).
    doc_rows = []
    for report in sorted(reports, key=lambda r: r.conformance):
        conf_class = {
            Conformance.CONFORMANT: "text-success",
            Conformance.PARTIALLY: "text-warning",
            Conformance.NOT_CONFORMANT: "text-error",
        }.get(report.conformance, "text-slate-500")

        sr_icon = {
            Conformance.CONFORMANT: '<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-success">check_circle</span>',
            Conformance.PARTIALLY: '<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-warning">error_outline</span>',
            Conformance.NOT_CONFORMANT: '<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-error">cancel</span>',
        }.get(report.conformance, "")

        report_link = quote(report.report_filename)
        safe_name = report.document_name.replace('"', '&quot;').replace("'", "&#39;")
        source_link = ""
        if report.original.source_url:
            source_link = f'<a class="text-xs text-blue-600 hover:underline" href="{report.original.source_url}" target="_blank" rel="noopener" title="View original on website">Website</a>'
        original_name = Path(report.original.file_path).name
        original_link = f'<a class="text-xs text-slate-600 hover:underline" href="../downloads/pdf/{quote(original_name)}" target="_blank" title="Download original un-remediated PDF">Original</a>'
        remediated_name = Path(report.remediated_path).name
        remediated_link = f'<a class="text-xs text-emerald-800 hover:underline" href="../remediated-pdfs/{quote(remediated_name)}" target="_blank" title="Download remediated PDF">Remediated</a>'
        links_cell = " &middot; ".join(filter(None, [source_link, original_link, remediated_link]))
        sr_readability = getattr(report, "screen_reader_readability", 0.0) or 0.0
        if sr_readability >= 90:
            rd_badge = "bg-success/10 text-success"
        elif sr_readability >= 70:
            rd_badge = "bg-warning/10 text-warning"
        else:
            rd_badge = "bg-error/10 text-error"
        doc_rows.append(
            f'<tr class="hover:bg-slate-50 transition-colors doc-row" data-name="{safe_name.lower()}">'
            f'<td class="px-6 py-4"><a class="text-navy font-semibold hover:underline" href="documents/{report_link}">{_truncate(report.document_name, 60)}</a></td>'
            f'<td class="px-6 py-4 text-slate-600">{report.original.file_type.upper()}</td>'
            f'<td class="{conf_class} px-6 py-4 font-bold">{report.conformance}</td>'
            f'<td class="px-6 py-4"><span class="px-2 py-1 {rd_badge} rounded text-xs font-bold">{sr_readability:.0f}</span></td>'
            f'<td class="px-6 py-4">{links_cell}</td>'
            f'</tr>'
        )

    # Issue table rows (Tailwind).
    issue_rows = []
    for rule_id, count in top_issues:
        pct_docs = round(count / total * 100, 2) if total > 0 else 0
        muted_class = " text-slate-400" if count == 0 else ""
        issue_rows.append(
            f'<tr class="hover:bg-slate-50 transition-colors{muted_class}">'
            f'<td class="px-6 py-4 font-medium">{rule_id}</td>'
            f'<td class="px-6 py-4">{count}</td>'
            f'<td class="px-6 py-4 text-right font-semibold">{pct_docs}%</td>'
            f'</tr>'
        )

    # WCAG table rows (Tailwind with badge spans).
    wcag_rows = []
    for cid in sorted(wcag_stats.keys()):
        s = wcag_stats[cid]
        pass_pct = round(s["pass"] / total * 100, 1) if total > 0 else 0
        if pass_pct == 100:
            badge_class = "bg-success/10 text-success"
        elif pass_pct > 90:
            badge_class = "bg-warning/10 text-warning"
        else:
            badge_class = "bg-error/10 text-error"
        wcag_rows.append(
            f'<tr class="hover:bg-slate-50 transition-colors">'
            f'<td class="px-6 py-4 font-medium">{cid} {s["name"]} ({s["level"]})</td>'
            f'<td class="px-6 py-4"><span class="px-2 py-1 {badge_class} rounded text-xs font-bold">{pass_pct}%</span></td>'
            f'<td class="px-6 py-4">{s["pass"]}</td>'
            f'<td class="px-6 py-4">{s["fail"]}</td>'
            f'<td class="px-6 py-4">{s["na"]}</td>'
            f'</tr>'
        )

    html = _render_campus_html(
        campus_name=campus_name,
        brand_color=brand_color,
        total=total, conformant=conformant, partial=partial, not_conf=not_conf,
        pct=pct, doc_rows=doc_rows, issue_rows=issue_rows, wcag_rows=wcag_rows,
        date_str=now.strftime("%B %d, %Y"),
        duplicate_entries=duplicate_entries or [],
        avg_readability=avg_readability,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    # Aggregate issue stats from normalized issues arrays.
    all_issue_counts: dict[str, int] = {}
    fixable_issue_counts: dict[str, int] = {}
    source_limited_counts: dict[str, int] = {}
    for report in reports:
        for issue in getattr(report, "issues", []) or []:
            code = issue.get("code", "unknown")
            all_issue_counts[code] = all_issue_counts.get(code, 0) + 1
            if issue.get("fixable"):
                fixable_issue_counts[code] = fixable_issue_counts.get(code, 0) + 1
            else:
                source_limited_counts[code] = source_limited_counts.get(code, 0) + 1

    # Also write machine-readable summary.
    summary = {
        "campus": campus_code or campus_name,
        "campus_name": campus_name,
        "total": total,
        "conformant": conformant,
        "partially_conformant": partial,
        "not_conformant": not_conf,
        "conformance_rate": pct,
        "partial_font_only": partial_font_only,
        "partial_structural": partial_structural,
        "partial_checker": partial_checker,
        "top_issues": dict(top_issues),
        "issue_summary": dict(sorted(all_issue_counts.items(), key=lambda x: -x[1])),
        "fixable_issue_summary": dict(sorted(fixable_issue_counts.items(), key=lambda x: -x[1])),
        "source_limited_issue_summary": dict(sorted(source_limited_counts.items(), key=lambda x: -x[1])),
        "avg_sr_readability": avg_readability,
        "min_sr_readability": min_readability,
        "max_sr_readability": max_readability,
        "generated_at": now.isoformat(),
    }
    summary_path = output_path.parent / "campus-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


# ---------------------------------------------------------------------------
# District dashboard
# ---------------------------------------------------------------------------


def generate_district_dashboard(
    campus_summaries: list[dict],
    output_path: Path,
    *,
    brand_color: str = "#003366",
) -> None:
    """Generate the district-wide compliance dashboard."""
    now = datetime.now(timezone.utc)

    total_docs = sum(s["total"] for s in campus_summaries)
    total_conf = sum(s["conformant"] for s in campus_summaries)
    total_partial = sum(s["partially_conformant"] for s in campus_summaries)
    total_not = sum(s["not_conformant"] for s in campus_summaries)
    overall_pct = round(total_conf / total_docs * 100, 1) if total_docs > 0 else 0

    # District-wide SR readability average.
    _rd_vals = [s.get("avg_sr_readability", 0) or 0 for s in campus_summaries]
    dist_avg_readability = round(sum(_rd_vals) / len(_rd_vals), 1) if _rd_vals else 0.0

    # Campus rows (Tailwind — Stitch design).
    campus_rows = []
    for s in sorted(campus_summaries, key=lambda x: -x["total"]):
        code = s.get("campus", "")
        name = s.get("campus_name", code)
        top = ", ".join(list(s.get("top_issues", {}).keys())[:3])
        rate = s["conformance_rate"]
        if rate >= 95:
            badge_class = "bg-emerald-100 text-[#047857]"
        elif rate >= 80:
            badge_class = "bg-amber-100 text-[#92400E]"
        else:
            badge_class = "bg-red-100 text-[#B91C1C]"
        campus_rd = s.get("avg_sr_readability", 0) or 0
        if campus_rd >= 90:
            rd_badge = "bg-emerald-100 text-[#047857]"
        elif campus_rd >= 70:
            rd_badge = "bg-amber-100 text-[#92400E]"
        else:
            rd_badge = "bg-red-100 text-[#B91C1C]"
        campus_rows.append(
            f'<tr class="hover:bg-slate-50/80 transition-colors">'
            f'<td class="px-8 py-4 font-semibold text-[#005A97] hover:underline cursor-pointer"><a class="text-[#005A97] font-semibold hover:underline" href="{code}/compliance/index.html">{name}</a></td>'
            f'<td class="px-4 py-4 text-center">{s["total"]:,}</td>'
            f'<td class="px-4 py-4 text-center">{s["conformant"]:,}</td>'
            f'<td class="px-4 py-4 text-center">{s["partially_conformant"]:,}</td>'
            f'<td class="px-4 py-4 text-center">{s["not_conformant"]:,}</td>'
            f'<td class="px-4 py-4 text-center"><span class="px-3 py-1 rounded-full {badge_class} font-bold text-xs">{rate}%</span></td>'
            f'<td class="px-4 py-4 text-center"><span class="px-3 py-1 rounded-full {rd_badge} font-bold text-xs">{campus_rd:.0f}</span></td>'
            f'<td class="px-8 py-4 text-xs text-slate-600">{top}</td>'
            f'</tr>'
        )

    date_str = now.strftime("%B %d, %Y")

    # District-wide breakdown totals
    dist_font_only = sum(s.get("partial_font_only", 0) for s in campus_summaries)
    dist_structural = sum(s.get("partial_structural", 0) for s in campus_summaries)
    dist_checker = sum(s.get("partial_checker", 0) for s in campus_summaries)
    dist_wcag_compliant = total_conf + dist_font_only
    dist_wcag_pct = round(dist_wcag_compliant / total_docs * 100, 1) if total_docs > 0 else 0

    html = f"""<!DOCTYPE html>
<html class="light" lang="en">
<head>
{_HEAD_COMMON}
<title>LACCD Compliance Dashboard</title>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col">
<a href="#main" class="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:bg-navy focus:text-white focus:rounded">Skip to main content</a>

<!-- Top Navigation Bar (District — navy) -->
<header class="bg-[#003D66] text-white text-sm font-medium border-b border-white/10 shadow-md z-50 sticky top-0">
<div class="flex justify-between items-center px-6 py-3 w-full max-w-[1200px] mx-auto">
<span class="text-xl font-bold text-white tracking-tight">LACCD Compliance</span>
<nav class="hidden md:flex items-center gap-6 ml-8" aria-label="Main navigation">
<a class="text-white border-b-2 border-white pb-1" href="#">District</a>
<a class="text-white hover:text-white hover:bg-white/10 px-2 py-1 rounded transition-colors" href="#campus-grid">Campuses</a>
</nav>
</div>
</header>

<!-- Side Navigation Bar -->
<aside class="fixed left-0 top-0 h-full z-40 hidden md:flex flex-col bg-slate-50 h-screen w-64 border-r border-slate-200 pt-16">
<div class="px-6 py-4 border-b border-slate-200 mb-4">
<h2 class="text-lg font-bold text-[#003D66]">Compliance Portal</h2>
<p class="text-xs text-slate-600 uppercase font-semibold tracking-wider">District Administration</p>
</div>
<nav class="flex-1 px-4 space-y-1" aria-label="Sidebar navigation">
<a class="flex items-center gap-3 px-4 py-3 bg-blue-50 text-[#003D66] font-semibold border-r-4 border-[#003D66] translate-x-1 rounded-l transition-transform" href="#">
<span class="material-symbols-outlined" aria-hidden="true">dashboard</span>
<span>District Overview</span>
</a>
<a class="flex items-center gap-3 px-4 py-3 text-slate-700 hover:bg-slate-100 rounded transition-all" href="#campus-grid">
<span class="material-symbols-outlined" aria-hidden="true">school</span>
<span>Campuses</span>
</a>
</nav>
</aside>

<!-- Main Content Area -->
<main id="main" class="flex-1 md:ml-64 p-6 lg:p-10">
<div class="max-w-[1200px] mx-auto space-y-8">

<!-- Page Header -->
<div class="mb-8">
<h1 class="text-3xl font-extrabold text-[#003D66] tracking-tight">Los Angeles Community College District</h1>
<p class="text-slate-500 font-medium text-lg mt-1">ADA Document Accessibility Compliance Dashboard</p>
</div>

<!-- Hero Stat Bar -->
<section class="grid grid-cols-2 lg:grid-cols-6 gap-4">
<div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
<span class="text-3xl font-bold text-slate-900">{total_docs:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-widest mt-2">Total Documents</span>
</div>
<div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
<span class="text-3xl font-bold text-[#047857]">{total_conf:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-widest mt-2">Conformant</span>
</div>
<div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
<span class="text-3xl font-bold text-[#B45309]">{total_partial:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-widest mt-2">Partially</span>
</div>
<div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
<span class="text-3xl font-bold text-[#B91C1C]">{total_not:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-widest mt-2">Not Conformant</span>
</div>
<div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
<span class="text-3xl font-black text-[#003D66]">{overall_pct}%</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-widest mt-2">Overall Rate</span>
</div>
<div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
<span class="text-3xl font-bold text-[#6366f1]">{dist_avg_readability:.0f}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-widest mt-2">SR Readability</span>
</div>
</section>

<!-- ADA Title II & WCAG 2.1 AA Compliance -->
<section id="compliance-section" class="bg-white p-8 rounded-xl border border-slate-200 shadow-sm">
<div class="flex items-center gap-3 mb-6">
<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-[#003D66] text-3xl">verified</span>
<h3 class="text-xl font-bold text-[#003D66]">ADA Title II &amp; WCAG 2.1 AA Compliance</h3>
</div>
<p class="text-slate-600 leading-relaxed mb-6">
This dashboard measures document compliance against <strong>WCAG 2.1 Level AA</strong> — the standard required by the <strong>DOJ ADA Title II Final Rule</strong> (April 2024). Each document is validated through automated structure checks, screen reader testing, and veraPDF PDF/UA-1 conformance analysis. The compliance deadline for state and local government entities is <strong>April 24, 2026</strong>.
</p>
<div class="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
<div class="p-4 rounded-lg bg-emerald-50 border border-emerald-200">
<div class="flex items-center gap-2 mb-1">
<span class="h-3 w-3 rounded-full bg-[#10B981] shrink-0"></span>
<h4 class="font-bold text-emerald-800 text-sm">Conformant</h4>
</div>
<p class="text-2xl font-black text-emerald-700">{total_conf:,}</p>
<p class="text-xs text-emerald-800">{overall_pct}% of all documents</p>
<p class="text-xs text-slate-500 mt-1">Fully meet WCAG 2.1 AA and ADA Title II.</p>
</div>
<div class="p-4 rounded-lg {"bg-emerald-50 border border-emerald-200" if dist_font_only == 0 else "bg-sky-50 border border-sky-200"}">
<div class="flex items-center gap-2 mb-1">
<span class="h-3 w-3 rounded-full {"bg-[#10B981]" if dist_font_only == 0 else "bg-sky-400"} shrink-0"></span>
<h4 class="font-bold {"text-emerald-800" if dist_font_only == 0 else "text-sky-800"} text-sm">Font-Only Issues</h4>
</div>
<p class="text-2xl font-black {"text-emerald-700" if dist_font_only == 0 else "text-sky-700"}">{dist_font_only:,}</p>
<p class="text-xs text-emerald-800">Meet WCAG 2.1 AA (functionally accessible)</p>
<p class="text-xs text-slate-500 mt-1">Pass checker + screen reader. Only fail PDF/UA font rules.</p>
</div>
<div class="p-4 rounded-lg bg-slate-100 border border-slate-200">
<div class="flex items-center gap-2 mb-1">
<span class="h-3 w-3 rounded-full bg-slate-400 shrink-0"></span>
<h4 class="font-bold text-slate-700 text-sm">Structural Issues</h4>
</div>
<p class="text-2xl font-black text-slate-600">{dist_structural:,}</p>
<p class="text-xs text-slate-600">May not fully meet WCAG 2.1 AA</p>
<p class="text-xs text-slate-600 mt-1">Unmarked content (7.1-3) or artifact mixing (7.1-2).</p>
</div>
<div class="p-4 rounded-lg bg-slate-100 border border-slate-200">
<div class="flex items-center gap-2 mb-1">
<span class="h-3 w-3 rounded-full bg-slate-400 shrink-0"></span>
<h4 class="font-bold text-slate-700 text-sm">Checker Failures</h4>
</div>
<p class="text-2xl font-black text-slate-600">{dist_checker:,}</p>
<p class="text-xs text-slate-600">Needs further remediation</p>
<p class="text-xs text-slate-600 mt-1">Image-only PDFs or broken text encoding.</p>
</div>
</div>
<div class="p-4 rounded-lg bg-[#003D66]/5 border border-[#003D66]/10">
<div class="flex items-center gap-3">
<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-[#003D66] shrink-0" style="font-size: 20px;">shield</span>
<div>
<p class="text-sm text-slate-700"><strong class="text-[#003D66]">WCAG 2.1 AA Compliant: {dist_wcag_compliant:,} documents ({dist_wcag_pct}%)</strong> — includes Conformant + font-only documents that pass all functional accessibility tests. The DOJ ADA Title II requires WCAG compliance, not PDF/UA.</p>
</div>
</div>
</div>
<p class="text-slate-600 text-xs mt-4">Compliance auditing powered by Remedy PDF Desktop.</p>
</section>

<!-- Campus Breakdown Table -->
<section id="campus-grid" class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
<div class="px-8 py-6 border-b border-slate-100 flex justify-between items-center">
<h3 class="text-xl font-bold text-[#003D66]">Campus Breakdown</h3>
<button class="text-[#005A97] text-sm font-semibold hover:underline flex items-center gap-1">
<span class="material-symbols-outlined" aria-hidden="true" style="font-size: 18px;">download</span> Export CSV
</button>
</div>
<div class="overflow-x-auto">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-slate-50 text-slate-600 text-xs font-bold uppercase tracking-wider">
<th scope="col" class="px-8 py-4 sticky top-0 bg-slate-50">Campus</th>
<th scope="col" class="px-4 py-4 text-center">Total</th>
<th scope="col" class="px-4 py-4 text-center">Conformant</th>
<th scope="col" class="px-4 py-4 text-center">Partial</th>
<th scope="col" class="px-4 py-4 text-center">Not Conf.</th>
<th scope="col" class="px-4 py-4 text-center">Rate</th>
<th scope="col" class="px-4 py-4 text-center">SR</th>
<th scope="col" class="px-8 py-4">Top Issues</th>
</tr>
</thead>
<tbody class="divide-y divide-slate-100 text-sm">
{"".join(campus_rows)}
</tbody>
</table>
</div>
</section>

</div>
</main>

<!-- Footer -->
<footer class="bg-slate-100 text-slate-600 text-xs uppercase tracking-wider w-full border-t border-slate-200 mt-auto py-8 px-4">
<div class="max-w-[1200px] mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
<div class="flex flex-col gap-1 items-center md:items-start">
<p>&copy; {now.year} Los Angeles Community College District. All Rights Reserved.</p>
<p class="text-[10px] text-slate-600 normal-case">Generated {date_str} by Remedy PDF Desktop</p>
</div>
<div class="flex flex-col items-center md:items-end gap-3">
<div class="flex gap-6">
<span class="text-slate-600">WCAG 2.1 AA Validated</span>
<span class="text-slate-600">&middot;</span>
<span class="text-slate-600">PDF/UA-1 Conformance Tested</span>
</div>
<div class="bg-[#003D66] text-white px-3 py-1 rounded font-bold tracking-normal normal-case">
ADA compliance deadline: April 2026
</div>
</div>
</div>
</footer>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify_short(text: str) -> str:
    import re
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:50] or "report"


def _truncate(text: str, maxlen: int) -> str:
    return text if len(text) <= maxlen else text[:maxlen - 3] + "..."


def _render_campus_html(
    *,
    campus_name: str,
    brand_color: str,
    total: int,
    conformant: int,
    partial: int,
    not_conf: int,
    pct: float,
    doc_rows: list[str],
    issue_rows: list[str],
    wcag_rows: list[str],
    date_str: str,
    duplicate_entries: list[dict],
    avg_readability: float = 0.0,
) -> str:
    # Build the duplicates section HTML (only when entries exist).
    duplicates_section = ""
    if duplicate_entries:
        sorted_dupes = sorted(
            duplicate_entries,
            key=lambda e: e.get("generated_at", ""),
            reverse=True,
        )
        dupe_rows = []
        for entry in sorted_dupes:
            doc_name = _truncate(entry.get("document_name", ""), 60)
            source_run = entry.get("source_run", "")
            generated_at = entry.get("generated_at", "")
            reason = entry.get("reason", "")
            dupe_rows.append(
                f'<tr class="italic text-slate-600">'
                f'<td class="px-6 py-4">{doc_name}</td>'
                f'<td class="px-6 py-4 text-xs uppercase tracking-tight">{source_run}</td>'
                f'<td class="px-6 py-4">{generated_at}</td>'
                f'<td class="px-6 py-4"><span class="px-2 py-1 bg-neutral_muted text-white text-[10px] rounded uppercase font-bold tracking-widest">{reason or "Superseded"}</span></td>'
                f'</tr>'
            )
        duplicates_section = f"""
<!-- Superseded Duplicates (accordion) -->
<section id="superseded" class="bg-bg_muted rounded-lg border border-slate-200">
<button onclick="var c=document.getElementById('dup-content');var a=document.getElementById('dup-arrow');if(c.classList.contains('hidden')){{c.classList.remove('hidden');a.style.transform='rotate(180deg)';}}else{{c.classList.add('hidden');a.style.transform='';}}"
  class="w-full flex items-center justify-between p-6 text-left cursor-pointer hover:bg-slate-100/50 transition-colors rounded-lg">
<div>
<h2 class="text-xl font-bold font-headline text-slate-800">Superseded Duplicates ({len(sorted_dupes):,})</h2>
<p class="text-slate-600 text-sm mt-1 italic">Older pipeline runs — excluded from compliance metrics. Click to expand.</p>
</div>
<span id="dup-arrow" class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-slate-600 transition-transform duration-200">expand_more</span>
</button>
<div id="dup-content" class="hidden">
<div class="px-6 pb-6">
<div class="bg-white rounded border border-slate-200 overflow-hidden max-h-[500px] overflow-y-auto">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-slate-50 text-slate-600 text-[10px] font-bold uppercase tracking-wider sticky-header">
<th scope="col" class="px-6 py-3">Document</th>
<th scope="col" class="px-6 py-3">Source Run</th>
<th scope="col" class="px-6 py-3">Processed Date</th>
<th scope="col" class="px-6 py-3">Status</th>
</tr>
</thead>
<tbody class="divide-y divide-slate-100 text-sm">
{"".join(dupe_rows)}
</tbody>
</table>
</div>
</div>
</div>
</section>"""

    top_nav = _top_nav(active="campus")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
{_HEAD_COMMON}
<title>Accessibility Compliance — {campus_name}</title>
</head>
<body class="bg-slate-50 font-body text-slate-900 antialiased">
<a href="#main" class="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:bg-navy focus:text-white focus:rounded">Skip to main content</a>

{top_nav}

<!-- Side Navigation -->
<aside class="hidden lg:flex flex-col fixed left-0 top-[60px] h-[calc(100vh-60px)] z-40 bg-slate-50 border-r border-slate-200 w-64 p-4">
<div class="mb-8 px-4">
<h2 class="text-lg font-black text-emerald-900 font-headline">{campus_name}</h2>
<p class="text-xs font-semibold tracking-wide text-emerald-700">Compliance Division</p>
</div>
<nav class="space-y-1">
<a class="flex items-center gap-3 bg-emerald-50 text-emerald-700 px-4 py-3 rounded-lg border-r-4 border-emerald-700 font-semibold tracking-wide text-sm transition-transform" href="#">
<span class="material-symbols-outlined" aria-hidden="true">dashboard</span> Overview
</a>
<a class="flex items-center gap-3 text-slate-600 px-4 py-3 hover:bg-slate-100 font-semibold tracking-wide text-sm transition-transform" href="#wcag">
<span class="material-symbols-outlined" aria-hidden="true">fact_check</span> WCAG Compliance
</a>
<a class="flex items-center gap-3 text-slate-600 px-4 py-3 hover:bg-slate-100 font-semibold tracking-wide text-sm transition-transform" href="#issues">
<span class="material-symbols-outlined" aria-hidden="true">warning</span> Top Issues
</a>
<a class="flex items-center gap-3 text-slate-600 px-4 py-3 hover:bg-slate-100 font-semibold tracking-wide text-sm transition-transform" href="#documents">
<span class="material-symbols-outlined" aria-hidden="true">description</span> Document Library
</a>
<a class="flex items-center gap-3 text-slate-600 px-4 py-3 hover:bg-slate-100 font-semibold tracking-wide text-sm transition-transform" href="vpat-acr.html">
<span class="material-symbols-outlined" aria-hidden="true">policy</span> VPAT / ACR
</a>
{"" if not duplicate_entries else '<a class="flex items-center gap-3 text-slate-600 px-4 py-3 hover:bg-slate-100 font-semibold tracking-wide text-sm transition-transform" href="#superseded"><span class="material-symbols-outlined" aria-hidden=true>history</span> Superseded</a>'}
</nav>
</aside>

<!-- Main Content Area -->
<main id="main" class="lg:ml-64 p-6 min-h-screen">
<div class="max-w-[1200px] mx-auto space-y-8">

<!-- Campus Branding Header -->
<div class="bg-[{brand_color}] text-white p-8 rounded-lg shadow-sm">
<div class="flex flex-col md:flex-row md:items-end justify-between gap-4">
<div>
<h1 class="text-3xl font-extrabold font-headline tracking-tight">Accessibility Compliance Dashboard</h1>
<p class="text-emerald-100 font-medium mt-1">{campus_name} — Los Angeles Community College District</p>
</div>
<div class="flex gap-3">
<a href="vpat-acr.html" class="bg-navy hover:bg-blue_accent text-white px-5 py-2.5 rounded-md font-semibold text-sm transition-colors flex items-center gap-2 no-underline">
<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-[18px]">file_open</span> View VPAT / ACR
</a>
<a href="../../district-dashboard.html" class="bg-navy hover:bg-blue_accent text-white px-5 py-2.5 rounded-md font-semibold text-sm transition-colors flex items-center gap-2 no-underline">
<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-[18px]">hub</span> District Overview
</a>
</div>
</div>
</div>

<!-- Hero Stats Grid -->
<div class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
<div class="bg-white p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col items-center text-center">
<span class="text-3xl font-bold text-slate-900 font-headline">{total:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-wider mt-1">Total Documents</span>
</div>
<div class="bg-white p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col items-center text-center border-t-4 border-t-success">
<span class="text-3xl font-bold text-success font-headline">{conformant:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-wider mt-1">Conformant</span>
</div>
<div class="bg-white p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col items-center text-center border-t-4 border-t-warning">
<span class="text-3xl font-bold text-warning font-headline">{partial:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-wider mt-1">Partially</span>
</div>
<div class="bg-white p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col items-center text-center border-t-4 border-t-error">
<span class="text-3xl font-bold text-error font-headline">{not_conf:,}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-wider mt-1">Not Conformant</span>
</div>
<div class="bg-white p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col items-center text-center border-t-4 border-t-success">
<span class="text-3xl font-bold text-success font-headline">{pct}%</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-wider mt-1">Compliance Rate</span>
</div>
<div class="bg-white p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col items-center text-center border-t-4 border-t-[#6366f1]">
<span class="text-3xl font-bold text-[#6366f1] font-headline">{avg_readability:.0f}</span>
<span class="text-xs font-semibold text-slate-600 uppercase tracking-wider mt-1">SR Readability</span>
</div>
</div>

<!-- Understanding Section -->
<section class="bg-white p-6 rounded-lg shadow-sm border border-slate-100">
<div class="flex items-center gap-3 mb-4">
<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-navy">info</span>
<h2 class="text-xl font-bold font-headline text-navy">Understanding This Dashboard</h2>
</div>
<div class="text-slate-600 leading-relaxed">
<p class="mb-4">Each document on the {campus_name} website was tested against <strong>WCAG 2.1 Level AA</strong>
    — the accessibility standard required for ADA compliance. Click any document name to see its
    full compliance report with before/after comparison.</p>
<ul class="list-disc list-inside space-y-1 text-sm">
<li><strong>Conformant</strong> — no blocking accessibility failures remain after automated validation.</li>
<li><strong>Partially Conformant</strong> — some issues remain, but the document may still be usable in part.</li>
<li><strong>Not Conformant</strong> — significant accessibility barriers remain.</li>
<li><strong>Compliance Rate</strong> — percentage of documents that are fully Conformant.</li>
</ul>
<div class="mt-4 p-4 rounded-lg bg-navy/5 border border-navy/10">
<div class="flex items-start gap-3">
<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-navy shrink-0" style="font-size: 20px;">verified</span>
<div>
<h4 class="font-bold text-navy text-sm mb-1">ADA Title II &amp; WCAG 2.1 AA Compliance</h4>
<p class="text-slate-600 text-xs leading-relaxed mb-2">Documents marked <strong>Conformant</strong> meet the accessibility requirements of the <strong>DOJ ADA Title II Final Rule</strong> (April 2024) and <strong>WCAG 2.1 Level AA</strong>. Each document is validated through automated structure checks, screen reader testing, and veraPDF PDF/UA-1 conformance analysis.</p>
<details class="text-xs text-slate-600">
<summary class="cursor-pointer font-semibold text-navy hover:underline mb-1">Detailed compliance breakdown</summary>
<ul class="mt-2 space-y-1.5 leading-relaxed">
<li><strong class="text-emerald-700">Conformant</strong> — No blocking checker failures or screen reader errors remain. Minor source-font-only PDF/UA issues may still be noted separately.</li>
<li><strong class="text-amber-600">Partially Conformant (font-only)</strong> — Pass all checker and screen reader tests. Only fail on PDF/UA font-level rules (e.g., missing ToUnicode CMap). These are <strong>functionally accessible</strong> and meet WCAG 2.1 AA, as the DOJ requires WCAG compliance, not PDF/UA.</li>
<li><strong class="text-amber-600">Partially Conformant (structural)</strong> — Have unmarked content (7.1-3) or artifact mixing (7.1-2) that may affect WCAG 1.3.1 (Info and Relationships) or 1.3.2 (Meaningful Sequence).</li>
<li><strong class="text-red-600">Checker failures</strong> — Image-only PDFs without OCR text layers or broken text encoding. These do not meet WCAG 2.1 AA.</li>
</ul>
<p class="mt-2 text-slate-600">The DOJ compliance deadline for state and local government entities is <strong>April 24, 2026</strong>.</p>
</details>
</div>
</div>
</div>
</div>
</section>

<!-- WCAG 2.1 AA Conformance by Criterion -->
<section id="wcag" class="bg-white rounded-lg shadow-sm border border-slate-100 overflow-hidden">
<div class="p-6 border-b border-slate-100 bg-slate-50/50">
<h2 class="text-lg font-bold font-headline text-slate-800">WCAG 2.1 AA Conformance by Criterion</h2>
</div>
<div class="overflow-x-auto">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-slate-50 text-slate-600 text-xs font-bold uppercase tracking-wider sticky-header">
<th scope="col" class="px-6 py-4">Success Criterion</th>
<th scope="col" class="px-6 py-4">Pass Rate</th>
<th scope="col" class="px-6 py-4">Pass</th>
<th scope="col" class="px-6 py-4">Fail</th>
<th scope="col" class="px-6 py-4">N/A</th>
</tr>
</thead>
<tbody class="divide-y divide-slate-100 text-sm">
{"".join(wcag_rows)}
</tbody>
</table>
</div>
</section>

<!-- Top Issues -->
<section id="issues" class="bg-white rounded-lg shadow-sm border border-slate-100 overflow-hidden">
<div class="p-6 border-b border-slate-100 bg-slate-50/50">
<h2 class="text-lg font-bold font-headline text-slate-800">Top Issues</h2>
</div>
<div class="overflow-x-auto">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-slate-50 text-slate-600 text-xs font-bold uppercase tracking-wider sticky-header">
<th scope="col" class="px-6 py-4">Issue</th>
<th scope="col" class="px-6 py-4">Documents Affected</th>
<th scope="col" class="px-6 py-4 text-right">% of Total</th>
</tr>
</thead>
<tbody class="divide-y divide-slate-100 text-sm">
{"".join(issue_rows) if issue_rows else '<tr><td class="px-6 py-4 text-slate-400" colspan="3">No issues found.</td></tr>'}
</tbody>
</table>
</div>
</section>

<!-- All Documents -->
<section id="documents" class="bg-white rounded-lg shadow-sm border border-slate-100 overflow-hidden">
<div class="p-6 border-b border-slate-100 bg-slate-50/50">
<div class="flex justify-between items-center mb-4">
<h2 class="text-lg font-bold font-headline text-slate-800">All Documents ({total:,})</h2>
<span id="doc-match-count" class="text-xs text-slate-600 font-medium hidden"></span>
</div>
<div class="relative">
<span class="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 text-[20px]" aria-hidden="true">search</span>
<input id="doc-search" type="text" placeholder="Search documents..." aria-label="Search documents" autocomplete="off"
  class="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-navy/20 focus:border-navy placeholder:text-slate-400" />
</div>
</div>
<div class="overflow-x-auto" id="doc-table-container">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-slate-50 text-slate-600 text-xs font-bold uppercase tracking-wider sticky-header">
<th scope="col" class="px-6 py-4">Document</th>
<th scope="col" class="px-6 py-4">Type</th>
<th scope="col" class="px-6 py-4">Conformance</th>
<th scope="col" class="px-6 py-4">SR Readability</th>
<th scope="col" class="px-6 py-4">Links</th>
</tr>
</thead>
<tbody id="doc-tbody" class="divide-y divide-slate-100 text-sm">
{"".join(doc_rows)}
</tbody>
</table>
</div>
<p id="doc-no-results" class="hidden p-6 text-center text-slate-600 text-sm">No documents match your search.</p>
</section>

{duplicates_section}

</div>
</main>

<!-- Footer -->
<footer class="lg:ml-64 border-t border-slate-200 bg-white p-8">
<div class="max-w-[1200px] mx-auto flex flex-col md:flex-row justify-between items-center gap-4 text-slate-600 text-xs font-medium">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined" aria-hidden="true" style="color:inherit" class2="text-sm">verified_user</span>
Generated {date_str} by Remedy PDF Desktop
</div>
<div class="flex gap-6">
<a class="hover:text-navy transition-colors" href="#">Privacy Policy</a>
<a class="hover:text-navy transition-colors" href="#">Compliance Standards</a>
<a class="hover:text-navy transition-colors" href="#">Support Portal</a>
</div>
</div>
</footer>

<script>
(function() {{
  var input = document.getElementById('doc-search');
  var tbody = document.getElementById('doc-tbody');
  var noResults = document.getElementById('doc-no-results');
  var matchCount = document.getElementById('doc-match-count');
  if (!input || !tbody) return;
  var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr.doc-row'));
  var totalRows = rows.length;
  input.addEventListener('input', function() {{
    var q = this.value.toLowerCase().trim();
    var visible = 0;
    for (var i = 0; i < rows.length; i++) {{
      var name = rows[i].getAttribute('data-name') || '';
      var show = !q || name.indexOf(q) !== -1;
      rows[i].style.display = show ? '' : 'none';
      if (show) visible++;
    }}
    if (noResults) noResults.classList.toggle('hidden', visible > 0 || !q);
    if (matchCount) {{
      if (q) {{
        matchCount.textContent = visible + ' of ' + totalRows + ' documents';
        matchCount.classList.remove('hidden');
      }} else {{
        matchCount.classList.add('hidden');
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""
