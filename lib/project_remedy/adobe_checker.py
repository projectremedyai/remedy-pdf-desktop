"""Adobe PDF Accessibility Checker API integration.

Uses Adobe's PDF Services API to run the same accessibility checks
that Adobe Acrobat's built-in checker runs.  Validates remediated PDFs
against Adobe's exact checker logic — the gold standard for WCAG/PDF-UA.

Requires ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET in .env.
Budget: ~500 pages/month on the free tier — use sparingly (test harness only).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://ims-na1.adobelogin.com/ims/token/v3"
_API_BASE = "https://pdf-services.adobe.io"

# Cache the access token (24h expiry)
_cached_token: str = ""
_token_expiry: float = 0.0


@dataclass
class AdobeCheckResult:
    """Result from Adobe's PDF Accessibility Checker API."""

    checked: bool = False
    passed: bool = False
    report: dict = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    error: str = ""
    pages_consumed: int = 0

    def summary(self) -> str:
        if not self.checked:
            return f"Adobe check skipped: {self.error}"
        total = len(self.issues)
        if self.passed:
            return f"Adobe: PASS ({total} info items)"
        failed = [i for i in self.issues if i.get("status") == "Failed"]
        needs_check = [i for i in self.issues if "manual" in i.get("status", "").lower()]
        return f"Adobe: FAIL ({len(failed)} failed, {len(needs_check)} need manual check)"


def _get_access_token() -> str:
    """Get an OAuth access token using client credentials flow."""
    global _cached_token, _token_expiry

    if _cached_token and time.time() < _token_expiry:
        return _cached_token

    client_id = os.environ.get("ADOBE_CLIENT_ID", "")
    client_secret = os.environ.get("ADOBE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        raise ValueError("ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET must be set in .env")

    resp = httpx.post(
        _TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "openid,AdobeID,DCAPI",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    _cached_token = data["access_token"]
    _token_expiry = time.time() + int(data.get("expires_in", 86400)) - 300  # 5 min buffer
    return _cached_token


def _api_headers() -> dict:
    """Build headers for Adobe PDF Services API."""
    token = _get_access_token()
    client_id = os.environ.get("ADOBE_CLIENT_ID", "")
    return {
        "Authorization": f"Bearer {token}",
        "x-api-key": client_id,
        "Content-Type": "application/json",
    }


def check_accessibility(pdf_path: Path) -> AdobeCheckResult:
    """Run Adobe's PDF Accessibility Checker on a PDF file.

    Workflow:
    1. Get upload URI from Adobe
    2. Upload the PDF
    3. Submit accessibility check job
    4. Poll for completion
    5. Download and parse the report

    Returns an AdobeCheckResult with pass/fail and detailed issues.
    """
    if not pdf_path.exists():
        return AdobeCheckResult(error=f"File not found: {pdf_path}")

    try:
        headers = _api_headers()
    except Exception as e:
        return AdobeCheckResult(error=str(e))

    try:
        # Step 1: Get upload URI
        resp = httpx.post(
            f"{_API_BASE}/assets",
            headers=headers,
            json={
                "mediaType": "application/pdf",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        asset_data = resp.json()
        upload_uri = asset_data["uploadUri"]
        asset_id = asset_data["assetID"]

        # Step 2: Upload PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        upload_resp = httpx.put(
            upload_uri,
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=120.0,
        )
        upload_resp.raise_for_status()

        # Step 3: Submit accessibility check job
        job_resp = httpx.post(
            f"{_API_BASE}/operation/accessibilitychecker",
            headers=headers,
            json={
                "assetID": asset_id,
            },
            timeout=30.0,
        )
        job_resp.raise_for_status()

        # Get polling URL from Location header
        poll_url = job_resp.headers.get("location", "")
        if not poll_url:
            # Try x-request-id as fallback
            job_id = job_resp.headers.get("x-request-id", "")
            poll_url = f"{_API_BASE}/operation/accessibilitychecker/{job_id}/status"

        # Step 4: Poll for completion
        import fitz
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()

        max_polls = 60
        for i in range(max_polls):
            time.sleep(3)
            status_resp = httpx.get(poll_url, headers=headers, timeout=30.0)

            if status_resp.status_code == 200:
                result_data = status_resp.json()
                status = result_data.get("status", "")

                if status == "done":
                    # Step 5: Download report
                    report_asset = result_data.get("report", {}).get("assetID", "")
                    if report_asset:
                        report_resp = httpx.get(
                            f"{_API_BASE}/assets/{report_asset}",
                            headers=headers,
                            timeout=30.0,
                        )
                        report_resp.raise_for_status()
                        report = report_resp.json()
                    else:
                        report = result_data

                    issues = _parse_adobe_report(report)
                    passed = all(
                        i.get("status") != "Failed" for i in issues
                    )

                    return AdobeCheckResult(
                        checked=True,
                        passed=passed,
                        report=report,
                        issues=issues,
                        pages_consumed=page_count,
                    )

                elif status == "failed":
                    error = result_data.get("error", {}).get("message", "unknown error")
                    return AdobeCheckResult(error=f"Adobe job failed: {error}")

                # Still in progress — continue polling

            elif status_resp.status_code == 202:
                continue  # Still processing
            else:
                return AdobeCheckResult(
                    error=f"Polling failed: HTTP {status_resp.status_code}"
                )

        return AdobeCheckResult(error="Adobe job timed out after 3 minutes")

    except httpx.HTTPStatusError as e:
        return AdobeCheckResult(error=f"Adobe API error: {e.response.status_code} {e.response.text[:200]}")
    except Exception as e:
        return AdobeCheckResult(error=str(e)[:200])


def _parse_adobe_report(report: dict) -> list[dict]:
    """Parse Adobe's accessibility report into a flat list of issues."""
    issues = []

    # The report structure varies — handle both flat and nested formats
    if isinstance(report, dict):
        for category, checks in report.items():
            if isinstance(checks, dict):
                for check_name, check_data in checks.items():
                    if isinstance(check_data, dict) and "status" in check_data:
                        issues.append({
                            "category": category,
                            "check": check_name,
                            "status": check_data.get("status", ""),
                            "description": check_data.get("description", ""),
                            "details": check_data.get("details", []),
                        })
            elif isinstance(checks, list):
                for item in checks:
                    if isinstance(item, dict):
                        issues.append({
                            "category": category,
                            **item,
                        })

    return issues
