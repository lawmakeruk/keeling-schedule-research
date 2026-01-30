# tests/unit/services/test_amendment_tracker.py
"""
Unit tests for AmendmentTracker class that manages amendment lifecycle tracking.
"""
import threading
import time
from unittest import TestCase
from unittest.mock import Mock, patch
from app.services.amendment_tracker import AmendmentTracker, AmendmentStatus, AmendmentRecord


class TestAmendmentTracker(TestCase):
    """Test cases for AmendmentTracker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = AmendmentTracker()

        # Sample amendment data
        self.amendment_data = {
            "amendment_id": "test-001",
            "schedule_id": "schedule-001",
            "amendment_type": "insertion",
            "affected_document": "Test Act",
            "affected_provision": "sec_1__subsec_2",
            "source_eid": "sec_25__subsec_2",
            "source": "s. 25(2)",
            "location": "After",
            "whole_provision": False,
        }

    def test_init(self):
        """Test tracker initialisation."""
        tracker = AmendmentTracker()

        # Check initial state
        self.assertEqual(len(tracker._records), 0)
        self.assertIsNotNone(tracker._lock)
        self.assertIsNone(tracker.metrics_logger)

        # Check status tracking initialised
        for status in AmendmentStatus:
            self.assertIn(status, tracker._by_status)
            self.assertEqual(len(tracker._by_status[status]), 0)

    def test_init_with_logger(self):
        """Test tracker initialisation with MetricsLogger."""
        mock_logger = Mock()
        tracker = AmendmentTracker(metrics_logger=mock_logger)

        self.assertEqual(tracker.metrics_logger, mock_logger)

    def test_register_amendment_success(self):
        """Test successful amendment registration."""
        self.tracker.register_amendment(**self.amendment_data)

        # Check amendment was registered
        record = self.tracker.get_amendment("test-001")
        self.assertIsNotNone(record)
        self.assertEqual(record.amendment_id, "test-001")
        self.assertEqual(record.status, AmendmentStatus.IDENTIFIED)
        self.assertEqual(record.location, "After")

        # Check status tracking
        identified = self.tracker.get_amendments_by_status(AmendmentStatus.IDENTIFIED)
        self.assertEqual(len(identified), 1)

    def test_register_amendment_invalid_location(self):
        """Test registration fails with invalid location."""
        invalid_data = self.amendment_data.copy()
        invalid_data["location"] = "Within"  # Invalid location

        with self.assertRaises(ValueError) as context:
            self.tracker.register_amendment(**invalid_data)

        self.assertIn("Invalid location 'Within'", str(context.exception))
        self.assertIn("Must be one of: After, Before, Each_Place, Replace", str(context.exception))

        # Check nothing was registered
        self.assertEqual(len(self.tracker.get_all_amendments()), 0)

    def test_register_amendment_duplicate(self):
        """Test registering duplicate amendment."""
        # Register once
        self.tracker.register_amendment(**self.amendment_data)

        # Try to register again with same ID
        with patch("app.services.amendment_tracker.logger.warning") as mock_warning:
            self.tracker.register_amendment(**self.amendment_data)
            mock_warning.assert_called_with("Amendment test-001 already registered")

        # Should still only have one
        self.assertEqual(len(self.tracker.get_all_amendments()), 1)

    def test_location_validation_all_valid(self):
        """Test all valid location values."""
        valid_locations = ["Before", "After", "Replace"]

        for i, location in enumerate(valid_locations):
            data = self.amendment_data.copy()
            data["amendment_id"] = f"test-{i}"
            data["location"] = location

            # Should not raise
            self.tracker.register_amendment(**data)

            record = self.tracker.get_amendment(f"test-{i}")
            self.assertEqual(record.location, location)

    def test_mark_applying(self):
        """Test marking amendment as applying."""
        self.tracker.register_amendment(**self.amendment_data)

        result = self.tracker.mark_applying("test-001")
        self.assertTrue(result)

        record = self.tracker.get_amendment("test-001")
        self.assertEqual(record.status, AmendmentStatus.APPLYING)

    def test_mark_applied_without_logger(self):
        """Test marking amendment as applied without logger."""
        self.tracker.register_amendment(**self.amendment_data)

        result = self.tracker.mark_applied("test-001", processing_time=1.5)
        self.assertTrue(result)

        record = self.tracker.get_amendment("test-001")
        self.assertEqual(record.status, AmendmentStatus.APPLIED)
        self.assertEqual(record.processing_time_seconds, 1.5)

        # Check status tracking updated
        applied = self.tracker.get_amendments_by_status(AmendmentStatus.APPLIED)
        self.assertEqual(len(applied), 1)
        identified = self.tracker.get_amendments_by_status(AmendmentStatus.IDENTIFIED)
        self.assertEqual(len(identified), 0)

    def test_mark_applied_with_logger(self):
        """Test marking amendment as applied with logger."""
        mock_logger = Mock()
        tracker = AmendmentTracker(metrics_logger=mock_logger)

        tracker.register_amendment(**self.amendment_data)
        tracker.mark_applied("test-001", processing_time=2.0)

        # Check logger was called
        mock_logger.update_amendment_application.assert_called_once_with(
            amendment_id="test-001", application_time_seconds=2.0, application_status=True
        )

    def test_mark_failed(self):
        """Test marking amendment as failed."""
        self.tracker.register_amendment(**self.amendment_data)

        result = self.tracker.mark_failed(
            "test-001", "Element not found", error_location="xml_lookup", processing_time=0.5
        )
        self.assertTrue(result)

        record = self.tracker.get_amendment("test-001")
        self.assertEqual(record.status, AmendmentStatus.FAILED)
        self.assertEqual(record.error_message, "Element not found")
        self.assertEqual(record.error_location, "xml_lookup")
        self.assertEqual(record.processing_time_seconds, 0.5)

    def test_mark_failed_with_logger(self):
        """Test marking amendment as failed with logger."""
        mock_logger = Mock()
        tracker = AmendmentTracker(metrics_logger=mock_logger)

        tracker.register_amendment(**self.amendment_data)
        tracker.mark_failed("test-001", "Test failure", processing_time=1.5)

        # Check logger was called with failure
        mock_logger.update_amendment_application.assert_called_once_with(
            amendment_id="test-001", application_time_seconds=1.5, application_status=False
        )

    def test_mark_error_commented(self):
        """Test marking error comment inserted."""
        self.tracker.register_amendment(**self.amendment_data)
        self.tracker.mark_failed("test-001", "Test error")

        result = self.tracker.mark_error_commented("test-001")
        self.assertTrue(result)

        record = self.tracker.get_amendment("test-001")
        self.assertEqual(record.status, AmendmentStatus.ERROR_COMMENTED)
        self.assertTrue(record.comment_inserted)

    def test_mark_error_commented_unknown_amendment(self):
        """Test marking error comment for unknown amendment."""
        result = self.tracker.mark_error_commented("unknown-id")
        self.assertFalse(result)

    def test_mark_error_commented_already_processed(self):
        """Test marking error comment for already processed amendment."""
        # Register and apply an amendment
        self.tracker.register_amendment(**self.amendment_data)
        self.tracker.mark_applied("test-001")

        # Try to mark error comment on applied amendment
        result = self.tracker.mark_error_commented("test-001")
        self.assertFalse(result)

        # Also test for amendment that already has error comment
        data2 = self.amendment_data.copy()
        data2["amendment_id"] = "test-002"
        self.tracker.register_amendment(**data2)
        self.tracker.mark_failed("test-002", "Error")
        self.tracker.mark_error_commented("test-002")

        # Try to mark error comment again
        result = self.tracker.mark_error_commented("test-002")
        self.assertFalse(result)

    def test_thread_safety(self):
        """Test concurrent access is thread-safe."""
        amendments_per_thread = 100
        num_threads = 10

        def register_amendments(start_id):
            for i in range(amendments_per_thread):
                data = self.amendment_data.copy()
                data["amendment_id"] = f"thread-{start_id}-{i}"
                self.tracker.register_amendment(**data)
                # Simulate some processing
                self.tracker.mark_applying(data["amendment_id"])
                time.sleep(0.001)
                self.tracker.mark_applied(data["amendment_id"])

        # Create threads
        threads = []
        for t in range(num_threads):
            thread = threading.Thread(target=register_amendments, args=(t * amendments_per_thread,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all amendments registered and processed
        all_amendments = self.tracker.get_all_amendments()
        self.assertEqual(len(all_amendments), num_threads * amendments_per_thread)

        # All should be applied
        applied = self.tracker.get_amendments_by_status(AmendmentStatus.APPLIED)
        self.assertEqual(len(applied), num_threads * amendments_per_thread)

    def test_amendment_record_validation(self):
        """Test AmendmentRecord validates location on creation."""
        # Valid locations
        for location in ["Before", "After", "Replace"]:
            record = AmendmentRecord(
                amendment_id="test",
                schedule_id="sched",
                status=AmendmentStatus.IDENTIFIED,
                amendment_type="insertion",
                affected_document="Test Act",
                affected_provision="sec_1",
                source_eid="sec_2",
                source="s. 2",
                location=location,
                whole_provision=False,
            )
            self.assertEqual(record.location, location)

        # Invalid location
        with self.assertRaises(ValueError) as context:
            AmendmentRecord(
                amendment_id="test",
                schedule_id="sched",
                status=AmendmentStatus.IDENTIFIED,
                amendment_type="insertion",
                affected_document="Test Act",
                affected_provision="sec_1",
                source_eid="sec_2",
                source="s. 2",
                location="Invalid",
                whole_provision=False,
            )

        self.assertIn("Invalid location 'Invalid'", str(context.exception))

    def test_update_status_unknown_amendment(self):
        """Test updating status of unknown amendment."""
        result = self.tracker.mark_applied("unknown-id")
        self.assertFalse(result)

        result = self.tracker.mark_failed("unknown-id", "Error")
        self.assertFalse(result)

    def test_ensure_no_silent_failures(self):
        """Test that all amendments are either applied or have error comments."""
        # Register multiple amendments
        statuses = [
            ("test-applied", AmendmentStatus.APPLIED),
            ("test-failed", AmendmentStatus.FAILED),
            ("test-stuck", AmendmentStatus.APPLYING),  # Stuck in progress
        ]

        for aid, _ in statuses:
            data = self.amendment_data.copy()
            data["amendment_id"] = aid
            self.tracker.register_amendment(**data)

        # Set statuses
        self.tracker.mark_applied("test-applied")
        self.tracker.mark_failed("test-failed", "Error message")
        self.tracker.mark_applying("test-stuck")

        # Before adding comments, we should have unresolved amendments
        requiring_comments = self.tracker.get_all_requiring_comments()
        self.assertEqual(len(requiring_comments), 2)  # failed and stuck

        resolution = self.tracker.ensure_all_amendments_resolved()
        self.assertFalse(resolution["all_resolved"])
        self.assertEqual(resolution["stats"]["unresolved"], 2)

        # Add error comments for all non-applied
        for record in requiring_comments:
            self.tracker.mark_error_commented(record.amendment_id)

        # Now all should be resolved
        resolution = self.tracker.ensure_all_amendments_resolved()
        self.assertTrue(resolution["all_resolved"])
        self.assertEqual(resolution["stats"]["unresolved"], 0)
        self.assertEqual(resolution["stats"]["applied"], 1)
        self.assertEqual(resolution["stats"]["with_error_comments"], 2)

    def test_mark_applying_unknown_amendment(self):
        """Test marking unknown amendment as applying."""
        result = self.tracker.mark_applying("unknown-id")
        self.assertFalse(result)

    def test_register_amendment_from_object_with_id(self):
        """Test registering amendment from object that has an ID."""
        # Create mock amendment object
        mock_amendment = Mock()
        mock_amendment.amendment_id = "existing-id"
        mock_amendment.amendment_type = Mock(value="insertion")
        mock_amendment.affected_document = "Test Act"
        mock_amendment.affected_provision = "sec_1"
        mock_amendment.source_eid = "sec_2"
        mock_amendment.source = "s. 2"
        mock_amendment.location = Mock(value="After")
        mock_amendment.whole_provision = True

        # Mock get_amendment_id to return the existing ID
        with patch("app.services.utils.get_amendment_id", return_value="existing-id"):
            self.tracker.register_amendment_from_object(mock_amendment, "schedule-001")

        # Verify amendment was registered
        record = self.tracker.get_amendment("existing-id")
        self.assertIsNotNone(record)
        self.assertEqual(record.amendment_type, "insertion")
        self.assertEqual(record.location, "After")

    def test_register_amendment_from_object_without_id(self):
        """Test registering amendment from object that lacks an ID."""
        # Create mock amendment object without ID
        mock_amendment = Mock()
        mock_amendment.amendment_type = Mock(value="deletion")
        mock_amendment.affected_document = "Test Act"
        mock_amendment.affected_provision = "sec_3"
        mock_amendment.source_eid = "sec_4"
        mock_amendment.source = "s. 4"
        mock_amendment.location = Mock(value="Replace")
        mock_amendment.whole_provision = False

        # Mock get_amendment_id to return None (no ID)
        with patch("app.services.utils.get_amendment_id", return_value=None):
            with patch("app.services.amendment_tracker.logger.warning") as mock_warning:
                self.tracker.register_amendment_from_object(mock_amendment, "schedule-002")

                # Verify warning was logged
                mock_warning.assert_called_once()
                self.assertIn("Amendment missing ID, generated:", mock_warning.call_args[0][0])

        # Verify amendment was registered with generated ID
        all_amendments = self.tracker.get_all_amendments()
        self.assertEqual(len(all_amendments), 1)

        # Check that ID was set on the amendment object
        self.assertTrue(hasattr(mock_amendment, "amendment_id"))
        self.assertIsNotNone(mock_amendment.amendment_id)

        # Verify the amendment was properly registered
        record = all_amendments[0]
        self.assertEqual(record.amendment_type, "deletion")
        self.assertEqual(record.location, "Replace")

    def test_mark_multiple_failed_with_mixed_ids(self):
        """Test marking multiple amendments as failed with mix of valid and empty IDs."""
        # Register some amendments
        amendments = ["test-001", "test-002", "test-003"]
        for aid in amendments:
            data = self.amendment_data.copy()
            data["amendment_id"] = aid
            self.tracker.register_amendment(**data)

        # Mark multiple as failed, including None and empty string
        amendment_ids = ["test-001", None, "test-002", "", "test-003"]
        self.tracker.mark_multiple_failed(amendment_ids, "Batch failure", error_location="batch_processing")

        # Verify only valid IDs were marked as failed
        for aid in ["test-001", "test-002", "test-003"]:
            record = self.tracker.get_amendment(aid)
            self.assertEqual(record.status, AmendmentStatus.FAILED)
            self.assertEqual(record.error_message, "Batch failure")
            self.assertEqual(record.error_location, "batch_processing")

    def test_mark_multiple_failed_empty_list(self):
        """Test marking multiple failed with empty list."""
        # Should handle empty list gracefully
        self.tracker.mark_multiple_failed([], "No amendments")

        # Nothing should be registered or failed
        self.assertEqual(len(self.tracker.get_all_amendments()), 0)

    def test_mark_multiple_failed_all_empty_ids(self):
        """Test marking multiple failed with all empty/None IDs."""
        # Register one amendment to ensure nothing happens to it
        self.tracker.register_amendment(**self.amendment_data)

        # Try to mark multiple failed with only empty values
        self.tracker.mark_multiple_failed([None, "", None], "Should not apply")

        # Original amendment should still be in IDENTIFIED status
        record = self.tracker.get_amendment("test-001")
        self.assertEqual(record.status, AmendmentStatus.IDENTIFIED)
        self.assertIsNone(record.error_message)

    def test_mark_error_commented_concurrent_deletion(self):
        """Test mark_error_commented when amendment is deleted between checks."""
        # This is a bit contrived but tests the edge case
        self.tracker.register_amendment(**self.amendment_data)

        # Mock _update_status to return False
        with patch.object(self.tracker, "_update_status", return_value=False):
            result = self.tracker.mark_error_commented("test-001")
            self.assertFalse(result)
