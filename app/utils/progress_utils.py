# app/utils/progress_utils.py
"""
Utilities for tracking Keeling Schedule generation progress from log files.

Provides log parsing to extract the latest status of a schedule
generation process by reading debug logs from the end backwards.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from ..services.utils import eid_to_source

# Events for progress tracking
PROGRESS_EVENTS = {
    "SCHEDULE_START",
    "CANDIDATE_FOUND",
    "CANDIDATE_IDENTIFIED",
    "IDENTIFICATION_SUMMARY",
    "AMENDMENT_IDENTIFIED",
    "LLM_RESPONSE",
    "AMENDMENT_APPLYING",
    "AMENDMENT_APPLIED",
    "AMENDMENT_FAILED",
    "SCHEDULE_END",
}

# High-level stage messages
STAGE_MESSAGES = {
    "STARTING": "Starting Keeling Schedule generation...",
    "CANDIDATES": "Analysing bill for amendments...",
    "IDENTIFICATION": "Identifying amendments in provisions...",
    "APPLICATION": "Applying amendments to target act...",
    "FINALISING": "Creating working version in Lawmaker...",
}

# LLM prompt names for amendment application
APPLY_PROMPT_NAMES = {
    "ApplyInsertionAmendment",
    "ApplySubstitutionAmendment",
    "ApplyDeletionAmendment",
}


def get_progress_message(schedule_id: str) -> str:
    """
    Get a user-friendly progress message for a schedule.

    Args:
        schedule_id: The schedule ID to get progress for

    Returns:
        User-friendly progress message string
    """
    try:
        # Get latest event for schedule
        latest_event = _get_latest_event_for_schedule(schedule_id)

        if not latest_event:
            return STAGE_MESSAGES["STARTING"]

        # Generate message based on event type
        return _generate_message_for_event(latest_event)

    except Exception:
        # Return safe default on any error
        return "Processing schedule updates..."


def _read_file_backwards(filepath: Path, chunk_size: int = 4096) -> Iterator[str]:
    """
    Read a file backwards line by line efficiently.

    Args:
        filepath: Path to the file
        chunk_size: Size of chunks to read at a time

    Yields:
        Lines from the end of the file backwards
    """
    with open(filepath, "rb") as f:
        # Go to end of file
        f.seek(0, 2)
        file_size = f.tell()

        remainder = b""
        position = file_size

        while position > 0:
            # Calculate chunk size
            read_size = min(chunk_size, position)
            position -= read_size

            # Read chunk
            f.seek(position)
            chunk = f.read(read_size)

            # Process chunk
            lines = (chunk + remainder).split(b"\n")

            # First line might be partial, save for next iteration
            remainder = lines[0]

            # Yield complete lines in reverse
            for line in reversed(lines[1:]):
                if line:
                    yield line.decode("utf-8", errors="ignore")

        # Don't forget the first line
        if remainder:
            yield remainder.decode("utf-8", errors="ignore")


def _get_latest_event_for_schedule(schedule_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the latest relevant event for a schedule from today's log file.

    Reads backwards through the log file to find the most recent event.
    Also builds a lookup table of amendment details needed for certain events.

    Args:
        schedule_id: The schedule ID to search for

    Returns:
        Dict containing event info or None if no matching events found.
    """
    # Get today's log file
    log_dir = Path("logs") / "debug"
    log_filename = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file = log_dir / log_filename

    if not log_file.exists():
        return None

    # Amendment details are needed for LLM_RESPONSE events
    # Since reading backwards, these must be collected along the way
    amendment_details = {}
    latest_event = None

    # Read file backwards for efficiency
    for line in _read_file_backwards(log_file):
        # Check if this line contains our schedule_id
        if f"schedule_id={schedule_id}" not in line:
            continue

        # Check if it contains a relevant event
        event_match = None
        for event in PROGRESS_EVENTS:
            if event in line:
                event_match = event
                break

        # If latest event found and it's not an LLM_RESPONSE that needs amendment details,
        # return immediately
        if latest_event and event_match != "AMENDMENT_IDENTIFIED":
            # Check if we need amendment details
            if (
                latest_event.get("event") == "LLM_RESPONSE"
                and latest_event.get("phase") == "application"
                and latest_event.get("amendment_id") not in amendment_details
            ):
                # Keep searching for amendment details
                pass
            else:
                # All required data collected
                break

        if event_match == "AMENDMENT_IDENTIFIED":
            # Extract amendment details for lookup
            _extract_amendment_details(line, amendment_details)
            continue

        if not event_match or latest_event:
            continue

        # Extract timestamp
        timestamp_match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2})", line)
        if not timestamp_match:
            continue

        # Build event data
        event_data = {"event": event_match, "timestamp": timestamp_match.group(1), "schedule_id": schedule_id}

        # Extract event-specific data
        _extract_event_data(event_match, line, event_data, amendment_details)

        # This is our latest event (since we're reading backwards)
        latest_event = event_data

    # If LLM_RESPONSE event exists that needs amendment details, add them now
    if (
        latest_event
        and latest_event.get("event") == "LLM_RESPONSE"
        and latest_event.get("phase") == "application"
        and latest_event.get("amendment_id") in amendment_details
    ):
        latest_event.update(amendment_details[latest_event["amendment_id"]])

    return latest_event


def _extract_amendment_details(line: str, amendment_details: Dict[str, Dict[str, Any]]) -> None:
    """
    Extract amendment details from AMENDMENT_IDENTIFIED line.

    Parses a log line containing amendment information and adds the extracted
    details to the amendment_details lookup dictionary.

    Args:
        line: Log line containing AMENDMENT_IDENTIFIED event
        amendment_details: Dictionary to store amendment details keyed by amendment_id

    Returns:
        None
    """
    aid_match = re.search(r"amendment_id=(\S+)", line)
    if not aid_match:
        return

    aid = aid_match.group(1)
    details = {}

    # Extract all relevant fields
    for field, pattern in [
        ("affected_provision", r"affected_provision=(\S+)"),
        ("source_eid", r"source_eid=(\S+)"),
        ("location", r"location=(\S+)"),
        ("amendment_type", r"amendment_type=(\S+)"),
    ]:
        match = re.search(pattern, line)
        if match:
            details[field] = match.group(1)

    # Format source for display
    if "source_eid" in details:
        details["source"] = eid_to_source(details["source_eid"])

    amendment_details[aid] = details


def _extract_event_data(
    event_type: str, line: str, event_data: Dict[str, Any], amendment_details: Dict[str, Dict[str, Any]]
) -> None:
    """
    Extract event-specific data based on event type.

    Parses the log line to extract relevant data fields based on the type of event
    and updates the event_data dictionary with the extracted information.

    Args:
        event_type: Type of event from PROGRESS_EVENTS
        line: Log line containing the event
        event_data: Dictionary to update with extracted event information
        amendment_details: Lookup dictionary containing amendment details

    Returns:
        None
    """
    if event_type == "CANDIDATE_IDENTIFIED":
        # Extract candidate eid
        eid_match = re.search(r"candidate_eid=(\S+)", line)
        if eid_match:
            event_data["candidate_eid"] = eid_match.group(1)

    elif event_type == "LLM_RESPONSE":
        # Check prompt type
        prompt_name_match = re.search(r"prompt_name=(\S+)", line)
        if prompt_name_match:
            prompt_name = prompt_name_match.group(1)

            if prompt_name == "TableOfAmendments":
                # Identification phase
                candidate_eid_match = re.search(r"candidate_eid=(\S+)", line)
                if candidate_eid_match:
                    event_data["candidate_eid"] = candidate_eid_match.group(1)
                    event_data["phase"] = "identification"

            elif prompt_name in APPLY_PROMPT_NAMES:
                # Application phase
                aid_match = re.search(r"amendment_id=(\S+)", line)
                if aid_match:
                    event_data["amendment_id"] = aid_match.group(1)
                    event_data["phase"] = "application"

    elif event_type == "AMENDMENT_APPLIED":
        # Extract affected provision for display
        affected_match = re.search(r"affected_provision=(\S+)", line)
        if affected_match:
            affected_provision = affected_match.group(1)
            event_data["source"] = eid_to_source(affected_provision)

    elif event_type == "AMENDMENT_APPLYING":
        # Extract the provision being applied
        affected_match = re.search(r"affected_provision=(\S+)", line)
        if affected_match:
            event_data["affected_provision"] = affected_match.group(1)


def _generate_message_for_event(event_data: Dict[str, Any]) -> str:
    """
    Generate a user-friendly message based on the event type and data.

    Args:
        event_data: Event dictionary from _get_latest_event_for_schedule

    Returns:
        Human-readable progress message
    """
    event_type = event_data.get("event")

    if event_type == "SCHEDULE_START":
        return STAGE_MESSAGES["STARTING"]

    elif event_type == "CANDIDATE_FOUND":
        return STAGE_MESSAGES["CANDIDATES"]

    elif event_type == "CANDIDATE_IDENTIFIED":
        candidate_eid = event_data.get("candidate_eid", "")
        if candidate_eid:
            formatted_provision = eid_to_source(candidate_eid)
            return f"Identifying amendments in {formatted_provision}..."
        return STAGE_MESSAGES["IDENTIFICATION"]

    elif event_type == "LLM_RESPONSE":
        # Handle LLM responses for real-time progress
        phase = event_data.get("phase")

        if phase == "identification":
            candidate_eid = event_data.get("candidate_eid", "")
            if candidate_eid:
                formatted_provision = eid_to_source(candidate_eid)
                return f"Identifying amendments in {formatted_provision}..."
            return STAGE_MESSAGES["IDENTIFICATION"]

        elif phase == "application":
            # Show progress for amendment application
            affected_provision = event_data.get("affected_provision", "")
            amendment_type = event_data.get("amendment_type", "amendment")

            if affected_provision:
                formatted_provision = eid_to_source(affected_provision)
                # Use past tense to indicate completion of LLM processing
                return f"Applied {amendment_type.lower()} to {formatted_provision}..."
            return STAGE_MESSAGES["APPLICATION"]

    elif event_type == "AMENDMENT_APPLYING":
        affected_provision = event_data.get("affected_provision", "")
        if affected_provision:
            formatted_provision = eid_to_source(affected_provision)
            return f"Merging amendment into {formatted_provision}..."
        return STAGE_MESSAGES["APPLICATION"]

    elif event_type == "AMENDMENT_APPLIED":
        # This is the actual XML merge completion
        source = event_data.get("source", "")
        if source:
            return f"Merged amendment into {source}..."
        return STAGE_MESSAGES["APPLICATION"]

    elif event_type == "SCHEDULE_END":
        return STAGE_MESSAGES["FINALISING"]

    # Default fallback
    return "Processing schedule updates..."
