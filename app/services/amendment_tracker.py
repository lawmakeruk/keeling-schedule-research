# app/services/amendment_tracker.py
"""
Tracks the lifecycle of amendments from identification through application,
ensuring complete accountability and no silent failures.
"""

import threading
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from ..logging.debug_logger import get_logger, event, bind, EventType as EVT

logger = get_logger(__name__)


class AmendmentStatus(Enum):
    """Status of an amendment in its lifecycle."""

    IDENTIFIED = "identified"
    APPLYING = "applying"
    APPLIED = "applied"
    FAILED = "failed"
    ERROR_COMMENTED = "error_commented"


@dataclass
class AmendmentRecord:
    """Record of a single amendment's journey through the system."""

    amendment_id: str
    schedule_id: str
    status: AmendmentStatus
    amendment_type: str
    affected_document: str
    affected_provision: str
    source_eid: str
    source: str
    location: str
    whole_provision: bool

    # Tracking fields
    identified_at: datetime = field(default_factory=datetime.now)
    status_history: List[tuple[AmendmentStatus, datetime]] = field(default_factory=list)
    error_message: Optional[str] = None
    error_location: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    comment_inserted: bool = False

    def __post_init__(self):
        """Initialise status history with initial status and validate location."""
        self.status_history.append((self.status, self.identified_at))

        # Validate location value
        valid_locations = {"Before", "After", "Replace", "Each_Place"}
        if self.location not in valid_locations:
            raise ValueError(
                f"Invalid location '{self.location}'. " f"Must be one of: {', '.join(sorted(valid_locations))}"
            )

    def update_status(self, new_status: AmendmentStatus, error_message: Optional[str] = None):
        """Update status and track history."""
        self.status = new_status
        self.status_history.append((new_status, datetime.now()))
        if error_message:
            self.error_message = error_message


class AmendmentTracker:
    """
    Tracks every amendment through its lifecycle, ensuring none are lost.
    Thread-safe implementation for concurrent amendment processing.

    This class is purely responsible for state tracking.
    """

    def __init__(self, metrics_logger=None):
        """
        Initialise the tracker.

        Args:
            metrics_logger: Optional MetricsLogger instance for database integration
        """
        self._records: Dict[str, AmendmentRecord] = {}
        self._lock = threading.Lock()
        self.metrics_logger = metrics_logger

        # Track amendments by status for quick queries
        self._by_status: Dict[AmendmentStatus, Set[str]] = {status: set() for status in AmendmentStatus}

    # ==================== Public Interface Methods ====================

    def register_amendment_from_object(self, amendment, schedule_id: str) -> None:
        """
        Register an amendment from an Amendment object.

        Args:
            amendment: Amendment object to register
            schedule_id: Parent schedule ID

        Raises:
            ValueError: If amendment has no ID
        """
        from .utils import get_amendment_id
        import uuid

        amendment_id = get_amendment_id(amendment)
        if not amendment_id:
            # Generate ID if missing and set it on the amendment
            amendment_id = str(uuid.uuid4())
            setattr(amendment, "amendment_id", amendment_id)
            logger.warning(f"Amendment missing ID, generated: {amendment_id}")

        self.register_amendment(
            amendment_id=amendment_id,
            schedule_id=schedule_id,
            amendment_type=amendment.amendment_type.value,
            affected_document=amendment.affected_document,
            affected_provision=amendment.affected_provision,
            source_eid=amendment.source_eid,
            source=amendment.source,
            location=amendment.location.value,
            whole_provision=amendment.whole_provision,
        )

    def register_amendment(
        self,
        amendment_id: str,
        schedule_id: str,
        amendment_type: str,
        affected_document: str,
        affected_provision: str,
        source_eid: str,
        source: str,
        location: str,
        whole_provision: bool,
    ) -> None:
        """
        Register a newly identified amendment.

        Args:
            amendment_id: Unique identifier for the amendment
            schedule_id: Parent schedule ID
            amendment_type: Type (insertion, deletion, substitution)
            affected_document: Target legislative document title

            affected_provision: Target provision eId
            source_eid: Source element eId
            source: Human-readable source reference
            location: Where to apply (Before, After, Replace)
            whole_provision: Whether it affects entire provision

        Raises:
            ValueError: If location is not valid
        """
        # Validate location before creating record
        valid_locations = {"Before", "After", "Replace", "Each_Place"}
        if location not in valid_locations:
            raise ValueError(
                f"Invalid location '{location}' for amendment {amendment_id}. "
                f"Must be one of: {', '.join(sorted(valid_locations))}"
            )

        with self._lock:
            if amendment_id in self._records:
                logger.warning(f"Amendment {amendment_id} already registered")
                return

            record = AmendmentRecord(
                amendment_id=amendment_id,
                schedule_id=schedule_id,
                status=AmendmentStatus.IDENTIFIED,
                amendment_type=amendment_type,
                affected_document=affected_document,
                affected_provision=affected_provision,
                source_eid=source_eid,
                source=source,
                location=location,
                whole_provision=whole_provision,
            )

            self._records[amendment_id] = record
            self._by_status[AmendmentStatus.IDENTIFIED].add(amendment_id)

            with bind(schedule_id=schedule_id, amendment_id=amendment_id):
                event(
                    logger,
                    EVT.AMENDMENT_IDENTIFIED,
                    f"Registered {amendment_type} amendment",
                    affected_provision=affected_provision,
                    source_eid=source_eid,
                    location=location,
                    whole_provision=whole_provision,
                )

            logger.debug(f"Registered amendment {amendment_id} for {affected_provision}")

    # ==================== Status Update Methods ====================

    def mark_applying(self, amendment_id: str) -> bool:
        """
        Mark amendment as currently being applied.

        Args:
            amendment_id: Amendment identifier

        Returns:
            True if status updated, False if amendment not found
        """
        if not self._update_status(amendment_id, AmendmentStatus.APPLYING):
            return False

        with self._lock:
            record = self._records[amendment_id]

        with bind(schedule_id=record.schedule_id, amendment_id=amendment_id):
            event(
                logger,
                EVT.AMENDMENT_APPLYING,
                f"Starting to apply {record.amendment_type} amendment",
                affected_provision=record.affected_provision,
            )

        return True

    def mark_applied(self, amendment_id: str, processing_time: Optional[float] = None) -> bool:
        """
        Mark amendment as successfully applied.

        Args:
            amendment_id: Amendment identifier
            processing_time: Time taken to apply in seconds

        Returns:
            True if status updated, False if amendment not found
        """
        if not self._update_status(amendment_id, AmendmentStatus.APPLIED):
            return False

        with self._lock:
            record = self._records[amendment_id]
            if processing_time:
                record.processing_time_seconds = processing_time

        with bind(schedule_id=record.schedule_id, amendment_id=amendment_id):
            event(
                logger,
                EVT.AMENDMENT_APPLIED,
                f"Successfully applied {record.amendment_type} amendment",
                affected_provision=record.affected_provision,
                processing_time_seconds=processing_time,
            )

        # Update MetricsLogger if available
        if self.metrics_logger and processing_time is not None:
            self.metrics_logger.update_amendment_application(
                amendment_id=amendment_id, application_time_seconds=processing_time, success_status=True
            )

        return True

    def mark_multiple_failed(
        self, amendment_ids: List[str], error_message: str, error_location: Optional[str] = None
    ) -> None:
        """
        Mark multiple amendments as failed.

        Args:
            amendment_ids: List of amendment IDs to mark as failed
            error_message: Error message to record
            error_location: Where the error occurred (optional)
        """
        for amendment_id in amendment_ids:
            if amendment_id:
                self.mark_failed(amendment_id, error_message, error_location=error_location)

    def mark_failed(
        self,
        amendment_id: str,
        error_message: str,
        error_location: Optional[str] = None,
        processing_time: Optional[float] = None,
    ) -> bool:
        """
        Mark amendment as failed.

        Args:
            amendment_id: Amendment identifier
            error_message: Description of the failure
            error_location: Where in the process it failed
            processing_time: Time spent before failure

        Returns:
            True if status updated, False if amendment not found
        """
        if not self._update_status(amendment_id, AmendmentStatus.FAILED, error_message):
            return False

        with self._lock:
            record = self._records[amendment_id]
            record.error_location = error_location
            if processing_time:
                record.processing_time_seconds = processing_time

        with bind(schedule_id=record.schedule_id, amendment_id=amendment_id):
            event(
                logger,
                EVT.AMENDMENT_FAILED,
                f"Failed to apply {record.amendment_type} amendment",
                affected_provision=record.affected_provision,
                error_message=error_message,
                error_location=error_location,
                processing_time_seconds=processing_time,
            )

        # Update MetricsLogger if available
        if self.metrics_logger and processing_time is not None:
            self.metrics_logger.update_amendment_application(
                amendment_id=amendment_id, application_time_seconds=processing_time, success_status=False
            )

        logger.error(f"Amendment {amendment_id} failed: {error_message}")
        return True

    def mark_error_commented(self, amendment_id: str) -> bool:
        """
        Mark that error comment was successfully inserted for non-applied amendment.

        Args:
            amendment_id: Amendment identifier

        Returns:
            True if status updated and comment marked, False otherwise
        """
        with self._lock:
            if amendment_id not in self._records:
                return False

            record = self._records[amendment_id]

            # Only mark comment inserted for amendments that need them
            if record.status not in {AmendmentStatus.APPLIED, AmendmentStatus.ERROR_COMMENTED}:
                record.comment_inserted = True
            else:
                return False  # Already applied or already has comment

        # Update status to ERROR_COMMENTED
        if not self._update_status(amendment_id, AmendmentStatus.ERROR_COMMENTED):
            return False

        with self._lock:
            record = self._records[amendment_id]

        with bind(schedule_id=record.schedule_id, amendment_id=amendment_id):
            event(
                logger,
                EVT.ERROR_COMMENT_INSERTED,
                f"Error comment inserted for {record.amendment_type} amendment",
                affected_provision=record.affected_provision,
            )

        return True

    # ==================== Query Methods ====================

    def get_amendment(self, amendment_id: str) -> Optional[AmendmentRecord]:
        """
        Retrieve a specific amendment record by its ID.

        Args:
            amendment_id: Unique identifier of the amendment to retrieve

        Returns:
            AmendmentRecord if found, None otherwise
        """
        with self._lock:
            return self._records.get(amendment_id)

    def get_all_amendments(self) -> List[AmendmentRecord]:
        """
        Retrieve all amendment records currently tracked.

        Returns:
            List of all AmendmentRecord objects in the tracker
        """
        with self._lock:
            return list(self._records.values())

    def get_amendments_by_status(self, status: AmendmentStatus) -> List[AmendmentRecord]:
        """
        Get all amendments with a specific status.

        Args:
            status: AmendmentStatus to filter by

        Returns:
            List of AmendmentRecord objects with the specified status
        """
        with self._lock:
            return [self._records[aid] for aid in self._by_status[status]]

    def get_all_requiring_comments(self) -> List[AmendmentRecord]:
        """
        Get all amendments that require error comments.

        This includes amendments with FAILED, IDENTIFIED, or APPLYING status
        that don't already have comments inserted.

        Returns:
            List of amendment records needing error comments
        """
        with self._lock:
            requiring_comments = []

            for aid, record in self._records.items():
                # Any non-applied status without a comment needs one
                if (
                    record.status
                    in {
                        AmendmentStatus.FAILED,
                        AmendmentStatus.IDENTIFIED,
                        AmendmentStatus.APPLYING,
                    }
                    and not record.comment_inserted
                ):
                    requiring_comments.append(record)

            return requiring_comments

    # ==================== Validation Methods ====================

    def ensure_all_amendments_resolved(self) -> Dict[str, Any]:
        """
        Final validation to ensure every amendment is either applied or has an error comment.

        Returns:
            Dictionary with resolution status and any unresolved amendments:
            - all_resolved: Boolean indicating if all amendments are resolved
            - stats: Dictionary with counts (total, applied, with_error_comments, unresolved)
            - unresolved_amendments: List of unresolved amendment details
        """
        with self._lock:
            unresolved = []
            stats = {"total": len(self._records), "applied": 0, "with_error_comments": 0, "unresolved": 0}

            for aid, record in self._records.items():
                if record.status == AmendmentStatus.APPLIED:
                    stats["applied"] += 1
                elif record.comment_inserted:
                    stats["with_error_comments"] += 1
                else:
                    stats["unresolved"] += 1
                    unresolved.append(
                        {"amendment_id": aid, "status": record.status.value, "error": record.error_message}
                    )

            return {"all_resolved": stats["unresolved"] == 0, "stats": stats, "unresolved_amendments": unresolved}

    # ==================== Internal Helper Methods ====================

    def _update_status(
        self, amendment_id: str, new_status: AmendmentStatus, error_message: Optional[str] = None
    ) -> bool:
        """
        Internal method to update status with proper tracking.

        Args:
            amendment_id: Amendment identifier
            new_status: New status to set
            error_message: Optional error message (typically for FAILED status)

        Returns:
            True if status updated, False if amendment not found
        """
        with self._lock:
            if amendment_id not in self._records:
                logger.error(f"Cannot update status for unknown amendment {amendment_id}")
                return False

            record = self._records[amendment_id]
            old_status = record.status

            # Update the record's status and history
            record.update_status(new_status, error_message)

            # Update the by-status index
            if old_status in self._by_status:
                self._by_status[old_status].discard(amendment_id)
            self._by_status[new_status].add(amendment_id)

            return True
