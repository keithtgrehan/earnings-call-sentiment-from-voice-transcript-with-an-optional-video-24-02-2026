#!/usr/bin/env python3
"""Acquire official public media for a narrow curated source slice."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import urlparse

import requests

from earnings_call_sentiment.downloaders.youtube import (
    download_youtube_audio,
    download_youtube_video,
)


DEFAULT_SOURCE_MANIFEST = Path("data/source_manifests/earnings_call_sources.csv")
DEFAULT_INPUTS_ROOT = Path("cache/curated_multimodal_slice")
DEFAULT_STATUS_CSV = DEFAULT_INPUTS_ROOT / "acquisition_status.csv"
DEFAULT_STATUS_JSON = DEFAULT_INPUTS_ROOT / "acquisition_status.json"
DEFAULT_SOURCE_IDS = [
    "msft_fy26_q2_example",
    "bac_q4_2025_example",
    "dis_q1_fy26_example",
    "goog_q1_2025_example",
    "sbux_prepared_remarks_example",
]

DIRECT_MEDIA_EXTENSIONS = {
    ".mp4": "video",
    ".m4v": "video",
    ".mkv": "video",
    ".mov": "video",
    ".webm": "video",
    ".mp3": "audio",
    ".m4a": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".ogg": "audio",
}

STATUS_COLUMNS = [
    "source_id",
    "manifest_row_found",
    "company",
    "event_title",
    "attempted_media_url",
    "attempted_url_label",
    "status",
    "video_path",
    "audio_path",
    "manual_file_placement_required",
    "reason",
]

# Keep this override list narrow and explicit. The values below are either the
# approved official IR page or the exact approved YouTube fallback supplied for
# this curated acquisition step.
VERIFIED_MEDIA_OVERRIDES = {
    "msft_fy26_q2_example": {
        "media_url": "https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q2",
        "url_label": "official_ir_page",
    },
    "bac_q4_2025_example": {
        "media_url": "https://www.youtube.com/watch?v=-4ztL9Bkb18",
        "url_label": "approved_youtube_fallback",
    },
    "dis_q1_fy26_example": {
        "media_url": "https://www.youtube.com/watch?v=kHplSjEUI4w",
        "url_label": "approved_youtube_fallback",
    },
    "goog_q1_2025_example": {
        "media_url": "https://www.youtube.com/watch?v=SySgINoaI9A",
        "url_label": "official_ir_linked_youtube",
    },
    "sbux_prepared_remarks_example": {
        "media_url": "https://www.youtube.com/watch?v=U8HT0PaymAA",
        "url_label": "approved_youtube_fallback",
    },
}

Q4_ATTENDEE_RE = re.compile(r"https://events\.q4inc\.com/attendee/(?P<meeting_id>\d+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attempt conservative public-media acquisition for curated source_ids "
            "without scraping or broad discovery."
        )
    )
    parser.add_argument(
        "--source-manifest",
        default=str(DEFAULT_SOURCE_MANIFEST),
        help="Curated source manifest path.",
    )
    parser.add_argument(
        "--inputs-root",
        default=str(DEFAULT_INPUTS_ROOT),
        help="Local acquisition root, defaulting to cache/curated_multimodal_slice.",
    )
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=DEFAULT_SOURCE_IDS,
        help="Curated source_ids to attempt.",
    )
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=["wav", "mp3", "m4a"],
        help="Audio format for official YouTube extraction.",
    )
    parser.add_argument(
        "--status-csv",
        default=str(DEFAULT_STATUS_CSV),
        help="Status CSV output path.",
    )
    parser.add_argument(
        "--status-json",
        default=str(DEFAULT_STATUS_JSON),
        help="Status JSON output path.",
    )
    return parser.parse_args()


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _write_status_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_status_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _looks_like_youtube(url: str) -> bool:
    host = urlparse(url).netloc.casefold()
    return host in {"youtube.com", "www.youtube.com", "youtu.be"} or host.endswith(".youtube.com")


def _existing_media_path(source_dir: Path, stem: str) -> Path | None:
    for extension in DIRECT_MEDIA_EXTENSIONS:
        candidate = source_dir / f"{stem}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _status_row(
    *,
    source_id: str,
    manifest_row_found: str,
    company: str,
    event_title: str,
    attempted_media_url: str,
    attempted_url_label: str,
    status: str,
    video_path: str,
    audio_path: str,
    manual_file_placement_required: str,
    reason: str,
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "manifest_row_found": manifest_row_found,
        "company": company,
        "event_title": event_title,
        "attempted_media_url": attempted_media_url,
        "attempted_url_label": attempted_url_label,
        "status": status,
        "video_path": video_path,
        "audio_path": audio_path,
        "manual_file_placement_required": manual_file_placement_required,
        "reason": reason,
    }


def _resolve_media_url(row: dict[str, str], source_id: str) -> tuple[str, str]:
    override = VERIFIED_MEDIA_OVERRIDES.get(source_id)
    if override:
        return override["media_url"], override["url_label"]
    return _clean_text(row.get("video_url")), "manifest_video_url"


def _probe_generic_page(url: str) -> tuple[str, str]:
    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        return "request_failed", str(exc)

    lowered = response.text.casefold()
    if response.status_code == 502:
        return "upstream_bad_gateway", "official webcast page returned HTTP 502"
    if response.status_code == 403:
        if "just a moment" in lowered or "_cf_chl_opt" in lowered:
            return "blocked_by_cloudflare", "official page is behind a Cloudflare challenge in this environment"
        return "blocked_by_source_site", f"official page returned HTTP {response.status_code}"
    if "webcast-registration-form" in lowered:
        return "registration_required", "official webcast requires registration form submission"
    if "provide the requested information below" in lowered and "submit" in lowered:
        return "registration_required", "official webcast requires registration form submission"
    if "earnings webcast" in lowered and "#register" in lowered:
        return "registration_required", "official event page points to a registration-gated webcast"
    if response.status_code >= 400:
        return "request_failed", f"official page returned HTTP {response.status_code}"
    return "page_loaded", ""


def _probe_q4_event(url: str) -> tuple[str, str]:
    match = Q4_ATTENDEE_RE.fullmatch(url)
    if match is None:
        return "not_q4_event_url", ""
    meeting_id = match.group("meeting_id")
    api_url = f"https://attendees.events.q4inc.com/rest/v1/event/{meeting_id}"
    try:
        response = requests.get(api_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    except requests.RequestException as exc:
        return "request_failed", str(exc)
    if response.status_code != 200:
        return "request_failed", f"Q4 event API returned HTTP {response.status_code}"
    payload = response.json().get("data", {})
    broadcast_url = _clean_text(payload.get("broadcastUrl"))
    backup_broadcast_url = _clean_text(payload.get("backupBroadcastUrl"))
    recordings = payload.get("broadcastRecordings") or payload.get("customRecordings") or []
    conference_recordings = (
        payload.get("conference", {}).get("eventRecordings")
        if isinstance(payload.get("conference"), dict)
        else []
    )
    if broadcast_url or backup_broadcast_url or recordings or conference_recordings:
        return "q4_event_has_media", ""
    post_event = payload.get("configuration", {}).get("postEvent", {})
    if isinstance(post_event, dict) and post_event.get("enableReplay") is False:
        return "no_public_replay_url", "official Q4 event API reports replay disabled and exposes no recording URL"
    return "no_public_media_url", "official Q4 event API exposes no public recording URL"


def _download_direct_media(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination.resolve()


def _attempt_generic_ytdlp(url: str, source_dir: Path) -> tuple[Path | None, str]:
    source_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "yt-dlp",
        "--no-warnings",
        "--no-playlist",
        "--output",
        str(source_dir / "video.%(ext)s"),
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = _clean_text(result.stderr) or _clean_text(result.stdout) or "yt-dlp failed"
        return None, message
    video_path = _existing_media_path(source_dir, "video")
    if video_path is None:
        return None, "yt-dlp completed without writing video.*"
    return video_path, ""


def _acquire_from_youtube(url: str, source_dir: Path, audio_format: str) -> tuple[Path | None, Path | None, str]:
    video_path = _existing_media_path(source_dir, "video")
    audio_path = _existing_media_path(source_dir, "audio")

    try:
        if video_path is None:
            video_path = download_youtube_video(url, source_dir)
        if audio_path is None:
            audio_path = download_youtube_audio(url, source_dir, audio_format=audio_format)
    except Exception as exc:
        return video_path, audio_path, str(exc)
    return video_path, audio_path, ""


def _acquire_source(
    *,
    row: dict[str, str] | None,
    source_id: str,
    inputs_root: Path,
    audio_format: str,
) -> dict[str, str]:
    if row is None:
        return _status_row(
            source_id=source_id,
            manifest_row_found="false",
            company="",
            event_title="",
            attempted_media_url="",
            attempted_url_label="",
            status="source_id_missing_from_manifest",
            video_path="",
            audio_path="",
            manual_file_placement_required="true",
            reason="source_id not present in curated source manifest",
        )

    company = _clean_text(row.get("company"))
    event_title = _clean_text(row.get("event_title"))
    source_dir = inputs_root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    media_url, url_label = _resolve_media_url(row, source_id)

    existing_video = _existing_media_path(source_dir, "video")
    existing_audio = _existing_media_path(source_dir, "audio")
    if existing_video and existing_audio:
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url=media_url,
            attempted_url_label=url_label,
            status="already_present",
            video_path=str(existing_video or ""),
            audio_path=str(existing_audio or ""),
            manual_file_placement_required="false",
            reason="",
        )

    if not media_url:
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url="",
            attempted_url_label=url_label,
            status="manual_required",
            video_path="",
            audio_path="",
            manual_file_placement_required="true",
            reason="manifest video_url is blank",
        )

    if _looks_like_youtube(media_url):
        video_path, audio_path, error = _acquire_from_youtube(media_url, source_dir, audio_format)
        if error:
            return _status_row(
                source_id=source_id,
                manifest_row_found="true",
                company=company,
                event_title=event_title,
                attempted_media_url=media_url,
                attempted_url_label=url_label,
                status="manual_required",
                video_path=str(video_path or ""),
                audio_path=str(audio_path or ""),
                manual_file_placement_required="true",
                reason=error,
            )
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url=media_url,
            attempted_url_label=url_label,
            status="acquired_video_and_audio",
            video_path=str(video_path or ""),
            audio_path=str(audio_path or ""),
            manual_file_placement_required="false",
            reason="",
        )

    q4_status, q4_reason = _probe_q4_event(media_url)
    if q4_status in {"no_public_replay_url", "no_public_media_url"}:
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url=media_url,
            attempted_url_label=url_label,
            status="manual_required",
            video_path="",
            audio_path="",
            manual_file_placement_required="true",
            reason=q4_reason,
        )

    direct_kind = DIRECT_MEDIA_EXTENSIONS.get(Path(urlparse(media_url).path).suffix.casefold())
    if direct_kind == "video":
        destination = source_dir / f"video{Path(urlparse(media_url).path).suffix}"
        video_path = _download_direct_media(media_url, destination)
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url=media_url,
            attempted_url_label=url_label,
            status="acquired_video_only",
            video_path=str(video_path),
            audio_path="",
            manual_file_placement_required="false",
            reason="",
        )
    if direct_kind == "audio":
        destination = source_dir / f"audio{Path(urlparse(media_url).path).suffix}"
        audio_path = _download_direct_media(media_url, destination)
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url=media_url,
            attempted_url_label=url_label,
            status="acquired_audio_only",
            video_path="",
            audio_path=str(audio_path),
            manual_file_placement_required="false",
            reason="",
        )

    page_status, page_reason = _probe_generic_page(media_url)
    if page_status in {
        "blocked_by_cloudflare",
        "blocked_by_source_site",
        "registration_required",
        "upstream_bad_gateway",
        "request_failed",
    }:
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url=media_url,
            attempted_url_label=url_label,
            status="manual_required",
            video_path="",
            audio_path="",
            manual_file_placement_required="true",
            reason=page_reason,
        )

    video_path, ytdlp_error = _attempt_generic_ytdlp(media_url, source_dir)
    if video_path is not None:
        return _status_row(
            source_id=source_id,
            manifest_row_found="true",
            company=company,
            event_title=event_title,
            attempted_media_url=media_url,
            attempted_url_label=url_label,
            status="acquired_video_only",
            video_path=str(video_path),
            audio_path="",
            manual_file_placement_required="false",
            reason="",
        )
    return _status_row(
        source_id=source_id,
        manifest_row_found="true",
        company=company,
        event_title=event_title,
        attempted_media_url=media_url,
        attempted_url_label=url_label,
        status="manual_required",
        video_path="",
        audio_path="",
        manual_file_placement_required="true",
        reason=ytdlp_error,
    )


def main() -> int:
    args = parse_args()
    repo_dir = repo_root()
    source_manifest_path = (repo_dir / args.source_manifest).resolve()
    inputs_root = (repo_dir / args.inputs_root).resolve()
    status_csv_path = (repo_dir / args.status_csv).resolve()
    status_json_path = (repo_dir / args.status_json).resolve()

    source_rows = _load_manifest_rows(source_manifest_path)
    rows_by_source_id = {_clean_text(row.get("source_id")): row for row in source_rows}
    source_ids = [source_id.strip() for source_id in args.source_ids if source_id.strip()]

    status_rows = [
        _acquire_source(
            row=rows_by_source_id.get(source_id),
            source_id=source_id,
            inputs_root=inputs_root,
            audio_format=args.audio_format,
        )
        for source_id in source_ids
    ]

    _write_status_csv(status_csv_path, status_rows)
    _write_status_json(
        status_json_path,
        {
            "source_ids": source_ids,
            "status_rows": status_rows,
        },
    )
    print(json.dumps({"status_rows": status_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
