# app/routes.py
"""
API routes for the Keeling Schedule Service.

Provides endpoints for health checks and Keeling schedule generation.
"""
import os
import tempfile
import uuid
import logging as stdlib_logging
from typing import Dict, Any, Tuple
from datetime import datetime

from lxml import etree
from flask import Blueprint, jsonify, request, Response
from .kernel.llm_kernel import get_kernel
from .services.keeling_service import KeelingService
from .utils.progress_utils import get_progress_message
from .services.utils import sort_amendments_by_source_eid

logger = stdlib_logging.getLogger(__name__)

api = Blueprint("api", __name__)


# ==================== Request Handlers ====================


@api.before_request
def handle_proxy_headers() -> None:
    """
    Handle proxy headers for requests behind a reverse proxy.

    Ensures Flask correctly interprets the original request scheme
    when running behind an ingress controller or load balancer.
    """
    scheme = request.headers.get("X-Forwarded-Proto")
    if scheme:
        request.environ["wsgi.url_scheme"] = scheme


# ==================== Health Check Endpoints ====================


@api.route("/health", methods=["GET"])
def health_check() -> Tuple[Response, int]:
    """
    Health check endpoint to verify service status.

    Returns:
        JSON response with status "ok" and HTTP 200
    """
    return jsonify({"status": "ok"}), 200


@api.route("/")
def home() -> str:
    """
    Home endpoint returning service status.

    Returns:
        Simple text message confirming service is running
    """
    return "Keeling Schedule Flask App is running!"


# ==================== Main API Endpoints ====================


@api.route("/api/keeling-schedule/generate", methods=["POST"])
def generate_keeling_schedule() -> Tuple[Response, int]:
    """
    Generate a Keeling schedule from provided XML content.

    Expects JSON payload with:
        - bill_xml: XML content of the amending bill
        - act_xml: XML content of the act being amended
        - act_name: Name of the act being amended
        - schedule_id: (Optional) Unique identifier for tracking

    Returns:
        JSON response containing:
        - status: "success" or "failure"
        - amended_act: The amended XML (or original if failed)
        - table_of_amendments: List of identified amendments
        - message: Error message if failed
    """
    data = request.get_json() or {}
    schedule_id = data.get("schedule_id", str(uuid.uuid4()))
    logger.info(f"Starting Keeling schedule generation with ID: {schedule_id}")

    try:
        # Validate request
        data = _validate_request_data()

        # Process the request
        amended_act_xml, table_of_amendments = _process_keeling_schedule(
            data["bill_xml"], data["act_xml"], data["act_name"], schedule_id
        )

        response: dict[str, str | list] = {"status": "success", "amended_act": amended_act_xml}
        if data["include_table_of_amendments"]:
            response["table_of_amendments"] = table_of_amendments

        return jsonify(response), 200

    except ValueError as e:
        # Client error (bad request)
        logger.warning(f"Bad request for schedule {schedule_id}: {str(e)}")
        return (
            jsonify(
                {
                    "status": "failure",
                    "message": str(e),
                }
            ),
            400,
        )

    except Exception as e:
        # Server error
        logger.exception(f"Unexpected error in generate_keeling_schedule for {schedule_id}")

        # Graceful degradation: return the original act unmodified
        data = request.get_json() or {}
        return (
            jsonify(
                {
                    "status": "failure",
                    "amended_act": data.get("act_xml", ""),
                    "table_of_amendments": [],
                    "message": str(e),
                }
            ),
            500,
        )


@api.route("/api/keeling-schedule/progress/<schedule_id>", methods=["GET"])
def get_progress(schedule_id: str) -> Tuple[Response, int]:
    """
    Get progress information for a Keeling schedule generation.
    Args:
        schedule_id: The unique identifier for the schedule
    Returns:
        JSON response with progress information
    """
    try:
        # Get progress message
        message = get_progress_message(schedule_id)
        return (jsonify({"message": message}), 200)
    except Exception as e:
        logger.exception(f"Error in get_progress for schedule_id {schedule_id}: {str(e)}")
        # Return a fallback response so the frontend doesn't break
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return (
            jsonify(
                {
                    "message": f"Processing... (Last checked: {current_time})",
                }
            ),
            200,
        )  # Return 200 to avoid frontend errors


# ==================== Helper Functions ====================


def _get_keeling_service() -> KeelingService:
    """
    Get an instance of the Keeling service.

    Returns:
        Configured KeelingService instance
    """
    llm_kernel = get_kernel()
    return KeelingService(llm_kernel)


def _validate_request_data() -> Dict[str, Any]:
    """
    Validate the request data for Keeling schedule generation.

    Returns:
        Validated request data dictionary

    Raises:
        ValueError: If validation fails
    """
    data = request.get_json()
    if not data:
        raise ValueError("No JSON data provided")

    # Check required fields
    required_fields = ["bill_xml", "act_name", "act_xml", "include_table_of_amendments"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Validate XML content
    try:
        etree.fromstring(data["bill_xml"])
        etree.fromstring(data["act_xml"])
    except etree.ParseError as e:
        raise ValueError(f"Invalid XML provided: {str(e)}")

    return data


def _process_keeling_schedule(bill_xml: str, act_xml: str, act_name: str, schedule_id: str) -> Tuple[str, list]:
    """
    Process the Keeling schedule generation.

    Args:
        bill_xml: XML content of the amending bill
        act_xml: XML content of the act being amended
        act_name: Name of the act
        schedule_id: Unique schedule identifier

    Returns:
        Tuple of (amended XML, table of amendments)
    """
    # Create temporary files for processing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as bill_file:
        bill_file.write(bill_xml)
        bill_path = bill_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as act_file:
        act_file.write(act_xml)
        act_path = act_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as output_file:
        output_path = output_file.name

    try:
        keeling_service = _get_keeling_service()

        # Process amendments
        amendments = keeling_service.process_amending_bill(bill_path, act_path, act_name, schedule_id)

        # Apply amendments
        keeling_service.apply_amendments(act_path, amendments, output_path, schedule_id)

        # Evaluate accuracy if ground truth is available
        keeling_service.metrics_logger.evaluate_schedule_accuracy(schedule_id, act_name)

        # Build the amendments table
        table_of_amendments = [
            {
                "source": amendment.source,
                "source_eid": amendment.source_eid,
                "type_of_amendment": amendment.amendment_type.value,
                "affected_provision": amendment.affected_provision,
                "location": amendment.location.value,
                "whole_provision": amendment.whole_provision,
            }
            for amendment in sort_amendments_by_source_eid(amendments)
        ]

        # Read the result
        with open(output_path, "r", encoding="utf-8") as f:
            amended_act_xml = f.read()

        return amended_act_xml, table_of_amendments

    finally:
        # Clean up temporary files
        for file_path in [bill_path, act_path, output_path]:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {e}")
