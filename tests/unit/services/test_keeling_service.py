# tests/unit/services/test_keeling_service.py
"""
Unit tests for the KeelingService class.
"""
from unittest.mock import Mock, patch, ANY, MagicMock
from concurrent.futures import Future
import pytest
from lxml import etree

from app.services.keeling_service import KeelingService
from app.models.amendments import Amendment, AmendmentType, AmendmentLocation


# ==================== Concurrent Testing Helpers ====================


def create_completed_future(result):
    """Create a Future that's already completed with the given result."""
    future = Future()
    future.set_result(result)
    return future


def create_failed_future(exception):
    """Create a Future that's already failed with the given exception."""
    future = Future()
    future.set_exception(exception)
    return future


# ==================== Test Class ====================


class TestKeelingService:
    """Tests for KeelingService class."""

    @pytest.fixture
    def mock_llm_kernel(self):
        """Create mock LLM kernel."""
        mock_kernel = Mock()
        mock_kernel.llm_config.enable_aws_bedrock = False
        mock_kernel.llm_config.enable_azure_openai = True
        mock_kernel.llm_config.bedrock_service_id = "bedrock-service"
        mock_kernel.llm_config.bedrock_model_id = "bedrock-model"
        mock_kernel.llm_config.azure_model_deployment_name = "gpt-4"
        mock_kernel.llm_config.get_active_service_id.return_value = "gpt-4"
        return mock_kernel

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies."""
        with (
            patch("app.services.keeling_service.XMLHandler") as mock_xml_class,
            patch("app.services.keeling_service.AmendmentTracker") as mock_tracker_class,
            patch("app.services.keeling_service.AmendmentProcessor") as mock_processor_class,
            patch("app.services.keeling_service.MetricsLogger") as mock_logger_class,
        ):

            # Create mock instances
            mock_xml = Mock()
            mock_tracker = Mock()
            mock_processor = Mock()
            mock_logger = Mock()

            # Configure class constructors
            mock_xml_class.return_value = mock_xml
            mock_tracker_class.return_value = mock_tracker
            mock_processor_class.return_value = mock_processor
            mock_logger_class.return_value = mock_logger

            # Set up namespaces on xml_handler
            mock_xml.namespaces = {
                "akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0",
                "ukl": "https://www.legislation.gov.uk/namespaces/UK-AKN",
            }

            yield {
                "xml_handler": mock_xml,
                "tracker": mock_tracker,
                "processor": mock_processor,
                "logger": mock_logger,
                "xml_class": mock_xml_class,
                "tracker_class": mock_tracker_class,
                "processor_class": mock_processor_class,
                "logger_class": mock_logger_class,
            }

    def test_init(self, mock_llm_kernel, mock_dependencies):
        """Test service initialisation."""
        service = KeelingService(mock_llm_kernel)

        # Verify components created
        assert service.llm_kernel == mock_llm_kernel
        assert service.xml_handler == mock_dependencies["xml_handler"]
        assert service.amendment_tracker == mock_dependencies["tracker"]
        assert service.amendment_processor == mock_dependencies["processor"]
        assert service.metrics_logger == mock_dependencies["logger"]

        # Verify logger configured correctly
        mock_dependencies["logger_class"].assert_called_once_with(
            enable_aws_bedrock=False,
            bedrock_service_id="bedrock-service",
            bedrock_model_id="bedrock-model",
            enable_azure_openai=True,
            azure_model_deployment_name="gpt-4",
        )

        # Verify tracker updated with logger
        assert mock_dependencies["tracker"].metrics_logger == mock_dependencies["logger"]

    @patch("os.path.getsize")
    def test_process_amending_bill_success(self, mock_getsize, mock_llm_kernel, mock_dependencies):
        """Test successful processing of amending bill."""
        mock_getsize.return_value = 1000
        service = KeelingService(mock_llm_kernel)

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_target_act = Mock(spec=etree.ElementTree)
        mock_simplified_tree = Mock(spec=etree.ElementTree)

        # Mock load_xml to return different trees for bill and act
        mock_dependencies["xml_handler"].load_xml.side_effect = [mock_tree, mock_target_act]
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 5

        # Mock extract_eid_patterns
        mock_dependencies["xml_handler"].extract_eid_patterns.return_value = {
            "examples": {"sections": ["sec_1", "sec_2"]},
            "conventions": {"definition_suffix": "_"},
        }

        # Mock deep copy to return our controlled simplified tree
        with patch("copy.deepcopy") as mock_deepcopy:
            mock_deepcopy.return_value = mock_simplified_tree

            # Mock the preprocessing methods
            with patch.object(service, "_preprocess_for_identification") as mock_preprocess:
                # Mock _get_candidate_amendments to return candidates
                with patch.object(service, "_get_candidate_amendments") as mock_get_candidates:
                    mock_get_candidates.return_value = [
                        ("<section>content1</section>", "sec_1"),
                        ("<section>content2</section>", "sec_2"),
                    ]

                    # Mock _identify_amendments_parallel
                    with patch.object(service, "_identify_amendments_parallel") as mock_identify:
                        mock_amendments = [Mock(spec=Amendment), Mock(spec=Amendment)]
                        mock_amendments[0].affected_document = "Test Act 2020"
                        mock_amendments[1].affected_document = "Test Act 2020"
                        mock_identify.return_value = mock_amendments

                        # Call method
                        result = service.process_amending_bill(
                            "/path/to/bill.xml", "/path/to/act.xml", "Test Act 2020", "schedule-123"
                        )

        # Verify results
        assert result == mock_amendments

        # Verify logging
        mock_dependencies["logger"].log_schedule_start.assert_called_once_with(
            schedule_id="schedule-123",
            act_name="Test Act 2020",
            model_id="gpt-4",
            service_id="gpt-4",
            max_worker_threads=256,
            bill_xml_size=1000,
            act_xml_size=None,
        )

        # Verify XML operations
        assert mock_dependencies["xml_handler"].load_xml.call_count == 2
        mock_dependencies["xml_handler"].load_xml.assert_any_call("/path/to/bill.xml")
        mock_dependencies["xml_handler"].load_xml.assert_any_call("/path/to/act.xml")

        # Verify pattern extraction
        mock_dependencies["xml_handler"].extract_eid_patterns.assert_called_once_with(mock_target_act)
        mock_dependencies["xml_handler"].set_dnum_counter.assert_called_once_with(5)

        # Verify deep copy was made
        mock_deepcopy.assert_called_once_with(mock_tree)

        # Verify preprocessing was called
        mock_preprocess.assert_called_once_with(mock_simplified_tree, "Test Act 2020", "schedule-123")

        # Verify _get_candidate_amendments was called
        mock_get_candidates.assert_called_once_with(mock_simplified_tree, "Test Act 2020", "schedule-123")

        # Verify _identify_amendments_parallel was called with the correct candidates
        mock_identify.assert_called_once_with(
            [("<section>content1</section>", "sec_1"), ("<section>content2</section>", "sec_2")],
            "Test Act 2020",
            "schedule-123",
        )

    def test_identify_single_candidate_success(self, mock_llm_kernel, mock_dependencies):
        """Test successful identification of amendments in a single candidate."""
        service = KeelingService(mock_llm_kernel)

        # Mock LLM response
        csv_response = """
        source_eid,source,type_of_amendment,whole_provision,location,affected_document,affected_provision
        sec_1,s. 1,INSERTION,true,AFTER,Test Act,sec_3
        """

        mock_llm_kernel.run_inference.return_value = csv_response

        # Call method
        result = service._identify_single_candidate("<section>content</section>", "sec_1", "Test Act", "schedule-123")

        # Verify result
        assert len(result) == 1
        assert isinstance(result[0], Amendment)
        assert result[0].source_eid == "sec_1"
        assert result[0].amendment_type == AmendmentType.INSERTION

        # Verify LLM called
        mock_llm_kernel.run_inference.assert_called_once_with(
            "TableOfAmendments",
            "schedule-123",
            None,
            "sec_1",
            act_name="Test Act",
            xml_provision="<section>content</section>",
            eid_patterns="{}",
        )

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_identify_amendments_parallel(self, mock_executor_class, mock_llm_kernel, mock_dependencies):
        """Test parallel identification of amendments."""
        service = KeelingService(mock_llm_kernel)

        # Mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Mock amendments
        amendments1 = [Mock(spec=Amendment)]
        amendments2 = [Mock(spec=Amendment)]

        # Create completed futures with results
        future1 = create_completed_future(amendments1)
        future2 = create_completed_future(amendments2)

        mock_executor.submit.side_effect = [future1, future2]

        # Call method
        candidates = [("<section>1</section>", "sec_1"), ("<section>2</section>", "sec_2")]
        result = service._identify_amendments_parallel(candidates, "Test Act", "schedule-123")

        # Verify results
        assert len(result) == 2
        assert amendments1[0] in result
        assert amendments2[0] in result

        # Verify thread pool created with correct max workers
        mock_executor_class.assert_called_once_with(max_workers=256)

    @patch("os.path.getsize")
    def test_apply_amendments_with_prefetch(self, mock_getsize, mock_llm_kernel, mock_dependencies):
        """Test applying amendments with LLM response pre-fetching."""
        mock_getsize.return_value = 2000
        service = KeelingService(mock_llm_kernel)

        # Create test amendments
        whole_amendment1 = Mock(spec=Amendment)
        whole_amendment1.whole_provision = True
        whole_amendment1.amendment_id = "whole1"
        whole_amendment1.amendment_type = Mock(value="INSERTION")
        whole_amendment1.affected_provision = "sec_1"
        whole_amendment1.source_eid = "source_1"
        whole_amendment1.source = "s. 1"
        whole_amendment1.location = Mock(value="AFTER")

        partial_amendment1 = Mock(spec=Amendment)
        partial_amendment1.whole_provision = False
        partial_amendment1.affected_provision = "sec_2"
        partial_amendment1.amendment_id = "partial1"
        partial_amendment1.amendment_type = Mock(value="SUBSTITUTION")
        partial_amendment1.source_eid = "source_3"
        partial_amendment1.source = "s. 3"
        partial_amendment1.location = Mock(value="REPLACE")

        amendments = [whole_amendment1, partial_amendment1]

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 10

        # Mock tracker resolution with correct structure
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": True,
            "stats": {"total": 2, "applied": 2, "with_error_comments": 0, "unresolved": 0},
            "unresolved_amendments": [],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = amendments

        # Mock get_all_requiring_comments to return empty list (no failures)
        mock_dependencies["tracker"].get_all_requiring_comments.return_value = []

        # Mock processor apply_amendment
        mock_dependencies["processor"].apply_amendment.side_effect = [
            (True, None),  # whole amendment succeeds
            (True, None),  # partial amendment succeeds
        ]

        with patch.object(service, "_fetch_llm_responses_parallel") as mock_fetch:
            # Mock LLM responses for partial amendments
            mock_fetch.return_value = {"partial1": "<updated>content</updated>"}

            # Call method
            service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify pre-fetch called only for partial amendments
        mock_fetch.assert_called_once()
        partial_amendments_arg = mock_fetch.call_args[0][0]
        assert len(partial_amendments_arg) == 1
        assert partial_amendments_arg[0] == partial_amendment1

        # Verify amendments applied in correct order
        assert mock_dependencies["processor"].apply_amendment.call_count == 2

        # Check whole amendment applied without LLM response
        first_call = mock_dependencies["processor"].apply_amendment.call_args_list[0]
        assert first_call[0][0] == whole_amendment1
        assert len(first_call[0]) == 4  # No LLM response parameter

        # Check partial amendment applied with LLM response
        second_call = mock_dependencies["processor"].apply_amendment.call_args_list[1]
        assert second_call[0][0] == partial_amendment1
        assert second_call[0][4] == "<updated>content</updated>"  # LLM response

        # Verify cleanup operations
        mock_dependencies["xml_handler"].renumber_dnums.assert_called_once()
        mock_dependencies["xml_handler"].normalise_namespaces.assert_called_once()
        mock_dependencies["xml_handler"].normalise_eids.assert_called_once()
        mock_dependencies["xml_handler"].save_xml.assert_called_once_with(ANY, "/path/to/output.xml")

    def test_update_act_size_database_error(self, mock_llm_kernel, mock_dependencies):
        """Test handling of database errors when updating act size."""
        service = KeelingService(mock_llm_kernel)

        # Mock database connection to raise exception
        mock_conn = Mock()
        mock_conn.cursor.side_effect = Exception("Database error")
        mock_dependencies["logger"]._get_db_connection.return_value = mock_conn
        amendments = []

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 0

        # Mock tracker resolution
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": True,
            "stats": {"total": 0, "applied": 0, "with_error_comments": 0, "unresolved": 0},
            "unresolved_amendments": [],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = []

        with patch("os.path.getsize", return_value=1000):
            # Should not raise exception even if database update fails
            service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify update was attempted
        mock_dependencies["logger"].update_schedule_act_size.assert_called_once_with("schedule-123", 1000)

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_fetch_llm_responses_parallel(self, mock_executor_class, mock_llm_kernel, mock_dependencies):
        """Test parallel fetching of LLM responses with grouping."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Create test amendments targeting different provisions
        amendment1 = Mock()
        amendment1.amendment_id = "id1"
        amendment1.affected_provision = "sec_1"
        amendment1.source_eid = "source_1"

        amendment2 = Mock()
        amendment2.amendment_id = "id2"
        amendment2.affected_provision = "sec_2"
        amendment2.source_eid = "source_2"

        amendments = [amendment1, amendment2]

        # Set up return values for each group
        group_results = [
            {"responses": {"id1": "response1"}, "failures": {}},  # First group result
            {"responses": {"id2": "response2"}, "failures": {}},  # Second group result
        ]

        # Create completed futures for each group result
        futures = [create_completed_future(result) for result in group_results]
        mock_executor.submit.side_effect = futures

        # Call method
        result = service._fetch_llm_responses_parallel(amendments, Mock(), "schedule-123")

        # Verify results
        assert result == {"id1": "response1", "id2": "response2"}

        # Verify thread pool created
        mock_executor_class.assert_called_once_with(max_workers=256)

    @patch("os.path.getsize")
    def test_apply_amendments_partial_no_llm_response(self, mock_getsize, mock_llm_kernel, mock_dependencies):
        """Test handling when LLM response is not available for partial amendment."""
        mock_getsize.return_value = 1000
        service = KeelingService(mock_llm_kernel)

        # Create partial amendment
        partial_amendment = Mock(spec=Amendment)
        partial_amendment.whole_provision = False
        partial_amendment.amendment_id = "partial1"
        partial_amendment.affected_provision = "sec_1"
        partial_amendment.amendment_type = Mock(value="INSERTION")
        partial_amendment.source_eid = "source_1"
        partial_amendment.source = "s. 1"
        partial_amendment.location = Mock(value="AFTER")

        amendments = [partial_amendment]

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 0

        # Mock processor to fail when no LLM response
        mock_dependencies["processor"].apply_amendment.return_value = (False, "No LLM response available")

        # Mock tracker with correct resolution structure
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": False,
            "stats": {"total": 1, "applied": 0, "with_error_comments": 0, "unresolved": 1},
            "unresolved_amendments": [
                {"amendment_id": "partial1", "status": "failed", "error": "No LLM response available"}
            ],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = []
        mock_dependencies["tracker"].get_all_requiring_comments.return_value = []

        with patch.object(service, "_fetch_llm_responses_parallel") as mock_fetch:
            # Return empty dict - no LLM response for the amendment
            mock_fetch.return_value = {}

            service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify amendment marked as failed
        mock_dependencies["tracker"].mark_failed.assert_called_once()

    @patch("os.path.getsize")
    def test_apply_amendments_processor_failure(self, mock_getsize, mock_llm_kernel, mock_dependencies):
        """Test handling when amendment processor returns failure."""
        mock_getsize.return_value = 1000
        service = KeelingService(mock_llm_kernel)

        # Create amendment
        amendment = Mock(spec=Amendment)
        amendment.whole_provision = True
        amendment.amendment_id = "id1"
        amendment.affected_provision = "sec_1"
        amendment.amendment_type = Mock(value="DELETION")
        amendment.source_eid = "source_1"
        amendment.source = "s. 1"
        amendment.location = Mock(value="REPLACE")

        amendments = [amendment]

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 0

        # Mock processor to return failure
        mock_dependencies["processor"].apply_amendment.return_value = (False, "Processing failed")

        # Mock tracker with correct resolution structure
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": False,
            "stats": {"total": 1, "applied": 0, "with_error_comments": 0, "unresolved": 1},
            "unresolved_amendments": [{"amendment_id": "id1", "status": "failed", "error": "Processing failed"}],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = []
        mock_dependencies["tracker"].get_all_requiring_comments.return_value = []

        service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify failure handling
        mock_dependencies["tracker"].mark_failed.assert_called_once()

    @patch("os.path.getsize")
    def test_apply_amendments_exception_during_application(self, mock_getsize, mock_llm_kernel, mock_dependencies):
        """Test exception handling during amendment application."""
        mock_getsize.return_value = 1000
        service = KeelingService(mock_llm_kernel)

        # Create amendment
        amendment = Mock(spec=Amendment)
        amendment.whole_provision = True
        amendment.amendment_id = "id1"
        amendment.affected_provision = "sec_1"
        amendment.amendment_type = Mock(value="INSERTION")
        amendment.source_eid = "source_1"
        amendment.source = "s. 1"
        amendment.location = Mock(value="AFTER")

        amendments = [amendment]

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 0

        # Mock processor to raise exception
        mock_dependencies["processor"].apply_amendment.side_effect = RuntimeError("Unexpected error")

        # Mock tracker with correct resolution structure
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": False,
            "stats": {"total": 1, "applied": 0, "with_error_comments": 0, "unresolved": 1},
            "unresolved_amendments": [{"amendment_id": "id1", "status": "failed", "error": "Unexpected error"}],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = []
        mock_dependencies["tracker"].get_all_requiring_comments.return_value = []

        service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify exception handling
        mock_dependencies["tracker"].mark_failed.assert_called_once_with("id1", "Unexpected error")

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_fetch_llm_responses_parallel_no_amendment_id(
        self, mock_executor_class, mock_llm_kernel, mock_dependencies
    ):
        """Test handling amendments without IDs in parallel fetch."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Create amendments - one with ID, one without
        amendment1 = Mock()
        amendment1.amendment_id = "id1"
        amendment1.affected_provision = "sec_1"
        amendment1.source_eid = "source_1"

        amendment2 = Mock()
        amendment2.amendment_id = None  # No ID
        amendment2.affected_provision = "sec_2"
        amendment2.source_eid = "source_2"

        # Mock get_amendment_id to return appropriate values
        with patch("app.services.keeling_service.get_amendment_id") as mock_get_id:
            mock_get_id.side_effect = lambda a: a.amendment_id

            # Create completed future with result
            future = create_completed_future({"id1": "response1"})
            mock_executor.submit.return_value = future

            amendments = [amendment1, amendment2]
            result = service._fetch_llm_responses_parallel(amendments, Mock(), "schedule-123")

            # Should only process amendment with ID
            assert result == {"id1": "response1"}

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_fetch_llm_responses_parallel_missing_elements(
        self, mock_executor_class, mock_llm_kernel, mock_dependencies
    ):
        """Test handling when target or source elements are missing."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Create amendments
        amendment1 = Mock()
        amendment1.amendment_id = "id1"
        amendment1.affected_provision = "sec_1"
        amendment1.source_eid = "source_1"

        amendment2 = Mock()
        amendment2.amendment_id = "id2"
        amendment2.affected_provision = "sec_2"
        amendment2.source_eid = "source_2"

        amendments = [amendment1, amendment2]

        # Create futures - first group fails, second succeeds
        future1 = create_completed_future({})  # Empty dict for failed group
        future2 = create_completed_future({"id2": "response2"})  # Success for second group

        mock_executor.submit.side_effect = [future1, future2]

        result = service._fetch_llm_responses_parallel(amendments, Mock(), "schedule-123")

        # Should only have response for second amendment
        assert result == {"id2": "response2"}

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_fetch_llm_responses_parallel_future_exception(
        self, mock_executor_class, mock_llm_kernel, mock_dependencies
    ):
        """Test exception handling when processing groups fails."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Create amendment
        amendment = Mock()
        amendment.amendment_id = "id1"
        amendment.affected_document = "Test Act"
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"

        amendments = [amendment]

        # Create future that raises exception
        future = create_failed_future(RuntimeError("Processing error"))
        mock_executor.submit.return_value = future

        result = service._fetch_llm_responses_parallel(amendments, Mock(), "schedule-123")

        # Should return empty dict on error
        assert result == {}

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_identify_amendments_parallel_with_exception(self, mock_executor_class, mock_llm_kernel, mock_dependencies):
        """Test error handling when a candidate processing fails."""
        service = KeelingService(mock_llm_kernel)

        # Mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Create futures - one succeeds, one fails
        future1 = create_completed_future([Mock(spec=Amendment)])
        future2 = create_failed_future(Exception("Processing failed"))

        mock_executor.submit.side_effect = [future1, future2]

        candidates = [("<section>1</section>", "sec_1"), ("<section>2</section>", "sec_2")]

        # Call method
        result = service._identify_amendments_parallel(candidates, "Test Act", "schedule-123")

        # Should still return the successful result
        assert len(result) == 1

    def test_get_candidate_amendments(self, mock_llm_kernel, mock_dependencies):
        """Test finding candidate provisions containing amendments."""
        service = KeelingService(mock_llm_kernel)

        # Mock XML elements
        elem1 = Mock()
        elem2 = Mock()
        elem3 = Mock()
        elem4 = Mock()

        # Set up eId attributes for debugging
        elem1.get.return_value = "sec_1"
        elem2.get.return_value = "sec_2"
        elem3.get.return_value = "sec_3"
        elem4.get.return_value = "sec_4"

        # Mock sibling element that contains act name and context pattern
        sibling = Mock()
        sibling.get.side_effect = lambda attr, default=None: {"class": "prov1"}.get(attr, default)

        # Mock parent elements with proper get() method that returns default values
        parent1 = Mock()
        parent1.get.side_effect = lambda attr, default=None: default
        parent1.getparent.return_value = None
        parent1.__iter__ = Mock(return_value=iter([elem1]))  # elem1 has no siblings

        parent2 = Mock()
        parent2.get.side_effect = lambda attr, default=None: default
        parent2.getparent.return_value = None
        parent2.__iter__ = Mock(return_value=iter([elem2]))  # elem2 has no siblings

        parent3 = Mock()
        parent3.get.side_effect = lambda attr, default=None: default
        parent3.getparent.return_value = None
        parent3.__iter__ = Mock(return_value=iter([elem3]))  # elem3 has no siblings

        parent4 = Mock()
        parent4.get.side_effect = lambda attr, default=None: default
        parent4.getparent.return_value = None
        parent4.__iter__ = Mock(return_value=iter([sibling, elem4]))  # elem4 has a preceding sibling

        # Set up parent relationships
        elem1.getparent.return_value = parent1
        elem2.getparent.return_value = parent2
        elem3.getparent.return_value = parent3
        elem4.getparent.return_value = parent4

        # Mock find_provisions_containing_text
        mock_dependencies["xml_handler"].find_provisions_containing_text.return_value = [
            (elem1, "sec_1"),
            (elem2, "sec_2"),
            (elem3, "sec_3"),
            (elem4, "sec_4"),
        ]

        # Track calls to get_text_content and return appropriate values
        text_content_values = {
            elem1: "This inserts new section in the Test Act",  # Contains act name
            elem2: "This inserts new section in another act",  # Wrong act
            elem3: "This omits section from the Test Act",  # Contains act name
            elem4: "This inserts a new provision",  # No act name
            sibling: "The Test Act is amended as follows",  # Has act name and context pattern
        }

        def get_text_content_side_effect(element, exclude_quoted=True):
            return text_content_values.get(element, "")

        mock_dependencies["xml_handler"].get_text_content.side_effect = get_text_content_side_effect

        # Mock get_comment_content - all return empty strings
        mock_dependencies["xml_handler"].get_comment_content.return_value = ""

        # Mock element_to_string
        mock_dependencies["xml_handler"].element_to_string.side_effect = [
            "<section>1</section>",
            "<section>3</section>",
            "<section>4</section>",  # elem4 should be included due to sibling context
        ]

        # Call method with schedule_id
        mock_tree = Mock()
        schedule_id = "test-schedule-123"
        result = service._get_candidate_amendments(mock_tree, "Test Act", schedule_id)

        # Should return elem1, elem3 (contain "Test Act") and elem4 (sibling context)
        assert len(result) == 3
        assert result[0] == ("<section>1</section>", "sec_1")
        assert result[1] == ("<section>3</section>", "sec_3")
        assert result[2] == ("<section>4</section>", "sec_4")

    def test_fetch_single_amendment_response(self, mock_llm_kernel, mock_dependencies):
        """Test the _fetch_single_amendment_response method."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create amendment
        amendment = Mock()
        amendment.amendment_id = "id1"
        amendment.affected_document = "Test Act"
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"

        # Mock XML elements
        mock_target = Mock()
        mock_source = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = [mock_target, mock_source]

        # Mock processor prepare
        mock_dependencies["processor"].prepare_llm_amendment.return_value = {
            "prompt_name": "TestPrompt",
            "schedule_id": "schedule-123",
            "amendment_id": "id1",
            "kwargs": {"param1": "value1"},
        }

        # Mock check_token_limits to return True (within limits)
        mock_dependencies["processor"].check_token_limits.return_value = (True, 1000)

        # Mock LLM response
        mock_llm_kernel.run_inference.return_value = "<amended>content</amended>"

        # Call method
        result = service._fetch_single_amendment_response(amendment, Mock(), "schedule-123")

        # Verify result
        assert result == "<amended>content</amended>"

        # Verify check_token_limits was called
        mock_dependencies["processor"].check_token_limits.assert_called_once_with(amendment, mock_target, mock_source)

    def test_register_amendment_without_id(self, mock_llm_kernel, mock_dependencies):
        """Test registering amendment that has no amendment_id attribute."""
        service = KeelingService(mock_llm_kernel)

        # Create amendment without amendment_id attribute
        amendment = Mock(spec=Amendment)
        amendment.amendment_type = Mock(value="INSERTION")
        amendment.affected_document = "Test Act"
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"
        amendment.source = "s. 1"
        amendment.location = Mock(value="AFTER")
        amendment.whole_provision = True

        # Remove amendment_id attribute to simulate getattr returning None
        delattr(amendment, "amendment_id")

        # Test registration through tracker
        service.amendment_tracker.register_amendment_from_object(amendment, "schedule-123")

        # Verify tracker method called
        mock_dependencies["tracker"].register_amendment_from_object.assert_called_once_with(amendment, "schedule-123")

    def test_process_amendment_group_empty_list(self, mock_llm_kernel, mock_dependencies):
        """Test _process_amendment_group with empty amendment list."""
        service = KeelingService(mock_llm_kernel)

        result = service._process_amendment_group([], Mock(), "schedule-123", "sec_1")

        assert result == {"responses": {}, "failures": {}}

    def test_process_amendment_group_no_amendment_id(self, mock_llm_kernel, mock_dependencies):
        """Test _process_amendment_group when amendment has no ID."""
        service = KeelingService(mock_llm_kernel)

        # Create amendment without ID
        amendment = Mock()
        amendment.amendment_id = None

        with patch("app.services.keeling_service.get_amendment_id", return_value=None):
            result = service._process_amendment_group([amendment], Mock(), "schedule-123", "sec_1")

        assert result == {"responses": {}, "failures": {}}

    def test_process_amendment_group_with_previous_response_no_parent(self, mock_llm_kernel, mock_dependencies):
        """Test _process_amendment_group when target has no parent in working tree."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create two amendments
        amendment1 = Mock()
        amendment1.amendment_id = "id1"
        amendment1.affected_document = "Test Act"
        amendment1.affected_provision = "sec_1"
        amendment1.source_eid = "source_1"

        amendment2 = Mock()
        amendment2.amendment_id = "id2"
        amendment2.affected_document = "Test Act"
        amendment2.affected_provision = "sec_1"
        amendment2.source_eid = "source_2"

        # Mock first successful response
        first_response = "<section>amended1</section>"

        # Mock target element without parent
        mock_target = Mock()
        mock_target.getparent.return_value = None

        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = [
            mock_target,  # First find for working tree
            Mock(),  # Second find for fetch_single_amendment_response
            Mock(),  # Source element
        ]

        # Mock parse_xml_string
        mock_dependencies["xml_handler"].parse_xml_string.return_value = MagicMock()

        with patch.object(service, "_fetch_single_amendment_response") as mock_fetch:
            mock_fetch.side_effect = [first_response, "<section>amended2</section>"]

            result = service._process_amendment_group([amendment1, amendment2], Mock(), "schedule-123", "sec_1")

        assert "responses" in result
        assert "failures" in result
        assert len(result["responses"]) == 2
        assert "id2" in result["responses"]

    def test_process_amendment_group_parse_error(self, mock_llm_kernel, mock_dependencies):
        """Test _process_amendment_group when parsing previous response fails."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create two amendments
        amendment1 = Mock()
        amendment1.amendment_id = "id1"

        amendment2 = Mock()
        amendment2.amendment_id = "id2"

        # Mock first successful response
        first_response = "<section>amended1</section>"

        # Mock parse_xml_string to fail
        mock_dependencies["xml_handler"].parse_xml_string.side_effect = Exception("Parse error")

        # Mock find_element_by_eid
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = Mock()

        with patch.object(service, "_fetch_single_amendment_response") as mock_fetch:
            mock_fetch.side_effect = [first_response, "<section>amended2</section>"]

            result = service._process_amendment_group([amendment1, amendment2], Mock(), "schedule-123", "sec_1")

        # Should still process both amendments
        assert len(result) == 2

    def test_process_amendment_group_target_not_found_in_working_tree(self, mock_llm_kernel, mock_dependencies):
        """Test _process_amendment_group when target not found in working tree."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create two amendments
        amendment1 = Mock()
        amendment1.amendment_id = "id1"

        amendment2 = Mock()
        amendment2.amendment_id = "id2"

        # Mock first successful response
        first_response = "<section>amended1</section>"

        # Mock find_element_by_eid to return None for working tree
        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = [
            None,  # Target not found in working tree
            Mock(),  # Target for fetch_single_amendment_response
            Mock(),  # Source element
        ]

        with patch.object(service, "_fetch_single_amendment_response") as mock_fetch:
            mock_fetch.side_effect = [first_response, "<section>amended2</section>"]

            result = service._process_amendment_group([amendment1, amendment2], Mock(), "schedule-123", "sec_1")

        # Should still process both amendments
        assert len(result) == 2

    def test_process_amendment_group_failed_amendment(self, mock_llm_kernel, mock_dependencies):
        """Test _process_amendment_group when one amendment fails."""
        service = KeelingService(mock_llm_kernel)

        # Create three amendments
        amendments = [Mock(amendment_id="id1"), Mock(amendment_id="id2"), Mock(amendment_id="id3")]

        # Import AmendmentStatus for proper mocking
        from app.services.amendment_tracker import AmendmentStatus

        # Need to mock the tracker for the failed amendment
        mock_failed_record = Mock()
        mock_failed_record.status = AmendmentStatus.FAILED
        mock_failed_record.error_message = "Test failure"

        # The get_amendment method will be called once per amendment when checking for failures
        # We want it to return the failed record only for id2
        def get_amendment_side_effect(amendment_id):
            if amendment_id == "id2":
                return mock_failed_record
            return None

        mock_dependencies["tracker"].get_amendment.side_effect = get_amendment_side_effect

        with patch.object(service, "_fetch_single_amendment_response") as mock_fetch:
            # Second amendment fails
            mock_fetch.side_effect = ["<response1>", None, "<response3>"]

            result = service._process_amendment_group(amendments, Mock(), "schedule-123", "sec_1")

        assert "responses" in result
        assert "failures" in result

        # Should have responses for first and third
        assert len(result["responses"]) == 2
        assert "id1" in result["responses"]
        assert "id3" in result["responses"]

        # Should have failure for second
        assert len(result["failures"]) == 1
        assert "id2" in result["failures"]
        assert result["failures"]["id2"] == "Test failure"

    def test_fetch_single_amendment_response_target_not_found(self, mock_llm_kernel, mock_dependencies):
        """Test _fetch_single_amendment_response when target element not found."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "test-id"
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"

        # Mock find_element_by_eid to return None for target
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = None

        result = service._fetch_single_amendment_response(amendment, Mock(), "schedule-123")

        assert result is None
        mock_dependencies["tracker"].mark_failed.assert_called_once_with(
            "test-id", "Target element sec_1 not found", error_location="llm_fetch"
        )

    def test_fetch_single_amendment_response_source_not_found(self, mock_llm_kernel, mock_dependencies):
        """Test _fetch_single_amendment_response when source element not found."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "test-id"
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"

        # Mock find_element_by_eid to return target but not source
        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = [
            Mock(),  # Target found
            None,  # Source not found
        ]

        result = service._fetch_single_amendment_response(amendment, Mock(), "schedule-123")

        assert result is None
        mock_dependencies["tracker"].mark_failed.assert_called_once_with(
            "test-id", "Source element source_1 not found", error_location="llm_fetch"
        )

    def test_fetch_single_amendment_response_llm_exception(self, mock_llm_kernel, mock_dependencies):
        """Test _fetch_single_amendment_response when LLM throws exception."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "test-id"
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"

        # Mock successful element finding
        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = [Mock(), Mock()]  # Target  # Source

        # Mock processor prepare
        mock_dependencies["processor"].prepare_llm_amendment.return_value = {
            "prompt_name": "TestPrompt",
            "schedule_id": "schedule-123",
            "amendment_id": "test-id",
            "kwargs": {"param1": "value1"},
        }

        # Mock check_token_limits to return True (within limits)
        mock_dependencies["processor"].check_token_limits.return_value = (True, 1000)

        # Mock LLM to throw exception
        mock_llm_kernel.run_inference.side_effect = RuntimeError("LLM service unavailable")

        result = service._fetch_single_amendment_response(amendment, Mock(), "schedule-123")

        assert result is None
        mock_dependencies["tracker"].mark_failed.assert_called_once_with(
            "test-id", "LLM call failed: LLM service unavailable", error_location="llm_fetch"
        )

    def test_fetch_single_amendment_response_no_amendment_id(self, mock_llm_kernel, mock_dependencies):
        """Test _fetch_single_amendment_response when amendment has no ID."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        # No amendment_id attribute
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"

        # Mock get_amendment_id to return None
        with patch("app.services.keeling_service.get_amendment_id", return_value=None):
            # Mock find_element_by_eid to return None for target
            mock_dependencies["xml_handler"].find_element_by_eid.return_value = None

            result = service._fetch_single_amendment_response(amendment, Mock(), "schedule-123")

        assert result is None
        # mark_failed should not be called when there's no amendment ID
        mock_dependencies["tracker"].mark_failed.assert_not_called()

    def test_fetch_single_amendment_response_exceeds_token_limit(self, mock_llm_kernel, mock_dependencies):
        """Test _fetch_single_amendment_response when token limit is exceeded."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "test-id"
        amendment.affected_provision = "sec_1"
        amendment.source_eid = "source_1"
        amendment.source = "s. 25(2)"

        # Mock successful element finding
        mock_target = Mock()
        mock_source = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = [mock_target, mock_source]

        # Mock check_token_limits to return False (exceeds limits)
        mock_dependencies["processor"].check_token_limits.return_value = (False, 20000)

        # Mock LLM config for debug logging
        mock_llm_kernel.llm_config.get_max_completion_tokens.return_value = 16384

        result = service._fetch_single_amendment_response(amendment, Mock(), "schedule-123")

        assert result is None

        # Verify token check was called
        mock_dependencies["processor"].check_token_limits.assert_called_once_with(amendment, mock_target, mock_source)

        # Verify amendment was marked as failed with correct error
        mock_dependencies["tracker"].mark_failed.assert_called_once_with(
            "test-id",
            "Amendment in s. 25(2) was not applied because the affected provision is too large to process.",
            error_location="token_limit_check",
        )

        # Verify LLM was NOT called
        mock_llm_kernel.run_inference.assert_not_called()

    def test_identify_single_candidate_error(self, mock_llm_kernel, mock_dependencies):
        """Test error handling in amendment identification (LLM_ERROR case)."""
        service = KeelingService(mock_llm_kernel)

        # Mock LLM error (not a ValueError)
        mock_llm_kernel.run_inference.side_effect = RuntimeError("Network timeout")

        with patch("app.services.keeling_service.event") as mock_event:
            result = service._identify_single_candidate(
                "<section>content</section>", "sec_1", "Test Act", "schedule-123"
            )

        # Should return empty list on error
        assert result == []

        # Verify CANDIDATE_SKIPPED event with LLM_ERROR reason
        skipped_calls = [
            call
            for call in mock_event.call_args_list
            if len(call[0]) >= 2 and hasattr(call[0][1], "value") and call[0][1].value == "CANDIDATE_SKIPPED"
        ]
        assert len(skipped_calls) == 1
        assert skipped_calls[0][1]["reason"] == "LLM_ERROR"
        assert skipped_calls[0][1]["error"] == "Network timeout"

    def test_identify_single_candidate_no_amendments_found(self, mock_llm_kernel, mock_dependencies):
        """Test when CSV has valid format but no valid amendment rows."""
        service = KeelingService(mock_llm_kernel)

        # Mock LLM response with valid headers but no data rows
        csv_response = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision"""
        mock_llm_kernel.run_inference.return_value = csv_response

        with patch("app.services.keeling_service.event") as mock_event:
            result = service._identify_single_candidate(
                "<section>content</section>", "sec_1", "Test Act", "schedule-123"
            )

        # Should return empty list
        assert result == []

        # Verify CANDIDATE_SKIPPED event with NO_AMENDMENTS_FOUND reason
        skipped_calls = [
            call
            for call in mock_event.call_args_list
            if len(call[0]) >= 2 and hasattr(call[0][1], "value") and call[0][1].value == "CANDIDATE_SKIPPED"
        ]
        assert len(skipped_calls) == 1
        assert skipped_calls[0][1]["reason"] == "NO_AMENDMENTS_FOUND"
        assert "No valid amendments found" in skipped_calls[0][1]["error"]

    def test_identify_single_candidate_empty_response(self, mock_llm_kernel, mock_dependencies):
        """Test when LLM returns empty response."""
        service = KeelingService(mock_llm_kernel)

        # Mock LLM to return empty string
        mock_llm_kernel.run_inference.return_value = ""

        with patch("app.services.keeling_service.event") as mock_event:
            result = service._identify_single_candidate(
                "<section>content</section>", "sec_1", "Test Act", "schedule-123"
            )

        # Should return empty list
        assert result == []

        # Verify CANDIDATE_SKIPPED event with EMPTY_RESPONSE reason
        skipped_calls = [
            call
            for call in mock_event.call_args_list
            if len(call[0]) >= 2 and hasattr(call[0][1], "value") and call[0][1].value == "CANDIDATE_SKIPPED"
        ]
        assert len(skipped_calls) == 1
        assert skipped_calls[0][1]["reason"] == "EMPTY_RESPONSE"
        assert "no headers" in skipped_calls[0][1]["error"].lower()

    def test_identify_single_candidate_invalid_csv(self, mock_llm_kernel, mock_dependencies):
        """Test when LLM returns invalid CSV format."""
        service = KeelingService(mock_llm_kernel)

        # Mock LLM response with missing required columns
        csv_response = """wrong_column1,wrong_column2
        value1,value2"""

        mock_llm_kernel.run_inference.return_value = csv_response

        with patch("app.services.keeling_service.event") as mock_event:
            result = service._identify_single_candidate(
                "<section>content</section>", "sec_1", "Test Act", "schedule-123"
            )

        # Should return empty list
        assert result == []

        # Verify CANDIDATE_SKIPPED event with INVALID_CSV reason
        skipped_calls = [
            call
            for call in mock_event.call_args_list
            if len(call[0]) >= 2 and hasattr(call[0][1], "value") and call[0][1].value == "CANDIDATE_SKIPPED"
        ]
        assert len(skipped_calls) == 1
        assert skipped_calls[0][1]["reason"] == "INVALID_CSV"
        # The error message should mention missing columns
        assert "missing required columns" in skipped_calls[0][1]["error"].lower()

    def test_identify_single_candidate_all_rows_invalid(self, mock_llm_kernel, mock_dependencies):
        """Test when CSV has valid headers but all data rows fail validation."""
        service = KeelingService(mock_llm_kernel)

        # Mock LLM response with valid headers but invalid data (missing required fields)
        csv_response = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
    ,,,,,
    sec_1,,INVALID_TYPE,maybe,NOWHERE,"""

        mock_llm_kernel.run_inference.return_value = csv_response

        # Mock the logger in utils to suppress error output during test
        with patch("app.services.utils.logger"):
            with patch("app.services.keeling_service.event") as mock_event:
                result = service._identify_single_candidate(
                    "<section>content</section>", "sec_1", "Test Act", "schedule-123"
                )

        # Should return empty list
        assert result == []

        # Verify CANDIDATE_SKIPPED event with NO_AMENDMENTS_FOUND reason
        skipped_calls = [
            call
            for call in mock_event.call_args_list
            if len(call[0]) >= 2 and hasattr(call[0][1], "value") and call[0][1].value == "CANDIDATE_SKIPPED"
        ]
        assert len(skipped_calls) == 1
        assert skipped_calls[0][1]["reason"] == "NO_AMENDMENTS_FOUND"

    def test_identify_single_candidate_filters_other_acts(self, mock_llm_kernel, mock_dependencies):
        """Test that amendments to other acts are filtered out and logged."""
        service = KeelingService(mock_llm_kernel)

        # Mock LLM response with amendments to multiple acts
        csv_response = (
            "source_eid,source,type_of_amendment,whole_provision,location,"
            "affected_document,affected_provision\n"
            "sec_1,s. 1,INSERTION,true,AFTER,Test Act,sec_10\n"
            "sec_1,s. 1,DELETION,false,REPLACE,Test Act,sec_11\n"
            "sec_1,s. 1,SUBSTITUTION,true,REPLACE,Other Act 2000,sec_20\n"
            "sec_1,s. 1,INSERTION,false,BEFORE,Different Act 1999,sec_30\n"
            "sec_1,s. 1,DELETION,true,REPLACE,Another Act 2010,sec_40"
        )

        mock_llm_kernel.run_inference.return_value = csv_response

        # Mock the metrics logger to avoid database operations
        with patch("app.services.keeling_service.logger") as mock_logger:
            with patch("app.services.keeling_service.event") as mock_event:
                result = service._identify_single_candidate(
                    "<section>content</section>", "sec_1", "Test Act", "schedule-123"
                )

        # Should return only amendments for "Test Act"
        assert len(result) == 2
        assert all(a.affected_document == "Test Act" for a in result)

        # Verify the INFO log about filtering was called
        info_calls = [call for call in mock_logger.info.call_args_list]
        filtering_log = None
        for call in info_calls:
            if "filtered out" in str(call):
                filtering_log = call
                break

        assert filtering_log is not None
        log_message = str(filtering_log[0][0])

        # Check the log message contains expected information
        assert "Candidate sec_1" in log_message
        assert "identified 2 amendments to Test Act" in log_message
        assert "filtered out 3 amendments to other acts" in log_message
        assert "Other Act 2000" in log_message
        assert "Different Act 1999" in log_message
        assert "Another Act 2010" in log_message

        # Verify CANDIDATE_IDENTIFIED event shows correct counts
        identified_calls = [
            call
            for call in mock_event.call_args_list
            if len(call[0]) >= 2 and hasattr(call[0][1], "value") and call[0][1].value == "CANDIDATE_IDENTIFIED"
        ]
        assert len(identified_calls) == 1
        assert identified_calls[0][1]["amendments"] == 2  # Only target act amendments
        assert identified_calls[0][1]["total_identified"] == 5  # All amendments found
        assert identified_calls[0][1]["filtered_out"] == 3  # Non-target amendments

        # Verify only target act amendments were logged to metrics
        # The metrics_logger is mocked via mock_dependencies
        assert service.metrics_logger.log_amendment.call_count == 2

        # Verify the calls were for the correct amendments (only Test Act ones)
        calls = service.metrics_logger.log_amendment.call_args_list
        for call in calls:
            # Each call should have affected_provision for Test Act amendments
            kwargs = call[1] if len(call) > 1 else call[0][0] if call[0] else {}
            if isinstance(kwargs, dict) and "affected_provision" in kwargs:
                assert kwargs["affected_provision"] in ["sec_10", "sec_11"]

    def test_fetch_llm_responses_parallel_group_with_only_failures(self, mock_llm_kernel, mock_dependencies):
        """Test when a group returns failures but no successful responses."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Mock executor
        mock_executor = Mock()
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create amendment
            amendment = Mock()
            amendment.amendment_id = "id1"
            amendment.affected_provision = "sec_1"
            amendment.source_eid = "source_1"

            # Group returns only failures, no responses
            group_result = {"responses": {}, "failures": {"id1": "Some error"}}
            future = create_completed_future(group_result)
            mock_executor.submit.return_value = future

            result = service._fetch_llm_responses_parallel([amendment], Mock(), "schedule-123")

            # Should have empty responses but failures should be tracked
            assert result == {}
            assert hasattr(service, "_fetch_phase_failures")
            assert service._fetch_phase_failures == {"id1": "Some error"}

    def test_process_amendment_group_failed_without_tracker_record(self, mock_llm_kernel, mock_dependencies):
        """Test when amendment fails but tracker doesn't have a failed record."""
        service = KeelingService(mock_llm_kernel)

        amendment = Mock(amendment_id="id1")

        # Mock tracker to return None (no record)
        mock_dependencies["tracker"].get_amendment.return_value = None

        with patch.object(service, "_fetch_single_amendment_response") as mock_fetch:
            # Amendment fails
            mock_fetch.return_value = None

            result = service._process_amendment_group([amendment], Mock(), "schedule-123", "sec_1")

        assert "responses" in result
        assert "failures" in result
        assert len(result["failures"]) == 1
        assert result["failures"]["id1"] == "Failed to fetch LLM response"  # Fallback message

    @patch("os.path.getsize")
    def test_apply_amendments_with_fetch_phase_failures(self, mock_getsize, mock_llm_kernel, mock_dependencies):
        """Test that fetch phase failures are properly propagated to apply phase."""
        mock_getsize.return_value = 1000
        service = KeelingService(mock_llm_kernel)

        # Create partial amendment
        amendment = Mock(spec=Amendment)
        amendment.whole_provision = False
        amendment.amendment_id = "id1"
        amendment.affected_provision = "sec_1"
        amendment.amendment_type = Mock(value="INSERTION")
        amendment.source_eid = "source_1"
        amendment.source = "s. 1"
        amendment.location = Mock(value="AFTER")

        amendments = [amendment]

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 0

        # Mock tracker resolution
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": False,
            "stats": {"total": 1, "applied": 0, "with_error_comments": 1, "unresolved": 0},
            "unresolved_amendments": [],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = []
        mock_dependencies["tracker"].get_all_requiring_comments.return_value = []

        # Mock processor to fail
        mock_dependencies["processor"].apply_amendment.return_value = (False, "No LLM response available")

        with patch.object(service, "_fetch_llm_responses_parallel") as mock_fetch:
            # Simulate that fetch phase detected a token limit error
            mock_fetch.return_value = {}  # No responses
            # Manually set the fetch phase failures
            service._fetch_phase_failures = {
                "id1": "Amendment in s. 1 was not applied because the affected provision is too large to process."
            }

            service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify the correct error message was used
        mock_dependencies["tracker"].mark_failed.assert_called_once_with(
            "id1",
            "Amendment in s. 1 was not applied because the affected provision is too large to process.",
            processing_time=ANY,
        )

    def test_preprocess_for_identification(self, mock_llm_kernel, mock_dependencies):
        """Test the _preprocess_for_identification method orchestration."""
        service = KeelingService(mock_llm_kernel)

        # Create mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock the preprocessing methods
        with patch.object(service, "_inject_crossheading_context") as mock_inject_crossheading:
            with patch.object(service, "_inject_document_context") as mock_inject_document:
                mock_inject_crossheading.return_value = 5  # 5 provisions injected
                mock_inject_document.return_value = 3  # 3 provisions injected

                # Call the method
                service._preprocess_for_identification(mock_tree, "Test Act 2020", "schedule-123")

        # Verify all three stages were called
        mock_dependencies["xml_handler"].simplify_amending_bill.assert_called_once_with(mock_tree)
        mock_inject_crossheading.assert_called_once_with(mock_tree)
        mock_inject_document.assert_called_once_with(mock_tree, "Test Act 2020", "schedule-123")

    def test_inject_crossheading_context_no_schedules(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_crossheading_context when no schedules found."""
        service = KeelingService(mock_llm_kernel)

        # Create mock tree
        mock_tree = Mock()
        mock_tree.xpath.return_value = []  # No schedules found

        # Call the method
        result = service._inject_crossheading_context(mock_tree)

        # Should return 0 as no schedules found
        assert result == 0
        mock_tree.xpath.assert_called_once_with(
            ".//akn:hcontainer[@name='schedule']", namespaces=mock_dependencies["xml_handler"].namespaces
        )

    def test_inject_crossheading_context_with_schedules(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_crossheading_context with schedules containing crossheadings."""
        service = KeelingService(mock_llm_kernel)

        # Create mock schedule
        mock_schedule = Mock()
        mock_schedule.get.side_effect = lambda key, default=None: "sched_1" if key == "eId" else default

        # Create mock crossheading
        mock_crossheading = Mock()
        mock_crossheading.get.side_effect = lambda key, default=None: "xhdg_1" if key == "eId" else default

        # Create mock heading element
        mock_heading_elem = Mock()
        mock_crossheading.find.return_value = mock_heading_elem

        # Create mock child provisions
        mock_child1 = Mock()
        mock_child1.get.side_effect = lambda key, default=None: "para_1" if key == "eId" else default
        mock_child2 = Mock()
        mock_child2.get.side_effect = lambda key, default=None: "para_2" if key == "eId" else default

        # Setup mock tree
        mock_tree = Mock()
        mock_tree.xpath.return_value = [mock_schedule]

        # Setup mock XML handler methods
        mock_dependencies["xml_handler"].get_schedule_heading_text.return_value = "Schedule 1 heading"
        mock_dependencies["xml_handler"].get_crossheadings_in_schedule.return_value = [mock_crossheading]
        mock_dependencies["xml_handler"].get_text_content.return_value = "Crossheading text"
        mock_dependencies["xml_handler"].get_ancestor_crossheading_contexts.return_value = []
        mock_dependencies["xml_handler"].get_crossheading_child_provisions.return_value = [mock_child1, mock_child2]

        # Call the method
        result = service._inject_crossheading_context(mock_tree)

        # Should have injected context into 2 child provisions
        assert result == 2

        # Verify injections
        assert mock_dependencies["xml_handler"].inject_xml_comment.call_count == 2
        mock_dependencies["xml_handler"].inject_xml_comment.assert_any_call(
            mock_child1, " Crossheading context: Schedule 1 heading > Crossheading text "
        )
        mock_dependencies["xml_handler"].inject_xml_comment.assert_any_call(
            mock_child2, " Crossheading context: Schedule 1 heading > Crossheading text "
        )

    def test_inject_crossheading_context_no_heading(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_crossheading_context when crossheading has no heading element."""
        service = KeelingService(mock_llm_kernel)

        # Create mock schedule
        mock_schedule = Mock()
        mock_schedule.get.return_value = "sched_1"

        # Create mock crossheading without heading
        mock_crossheading = Mock()
        mock_crossheading.get.return_value = "xhdg_1"
        mock_crossheading.find.return_value = None  # No heading element

        # Setup mock tree
        mock_tree = Mock()
        mock_tree.xpath.return_value = [mock_schedule]

        # Setup mock XML handler
        mock_dependencies["xml_handler"].get_schedule_heading_text.return_value = ""
        mock_dependencies["xml_handler"].get_crossheadings_in_schedule.return_value = [mock_crossheading]

        # Call the method
        result = service._inject_crossheading_context(mock_tree)

        # Should not inject anything as no heading found
        assert result == 0
        mock_dependencies["xml_handler"].inject_xml_comment.assert_not_called()

    def test_inject_document_context_no_context_provisions(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_document_context when no context provisions found."""
        service = KeelingService(mock_llm_kernel)

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock _find_context_provisions to return empty list
        with patch.object(service, "_find_context_provisions") as mock_find:
            mock_find.return_value = []

            # Call the method
            result = service._inject_document_context(mock_tree, "Test Act", "schedule-123")

        # Should return 0 as no context provisions found
        assert result == 0
        mock_find.assert_called_once_with(mock_tree, "Test Act")

    def test_inject_document_context_with_references(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_document_context with context provisions containing references."""
        service = KeelingService(mock_llm_kernel)

        # Create mock elements
        mock_context_elem = Mock()
        mock_context_elem.get.return_value = "context_1"

        mock_target_elem = Mock()
        mock_target_elem.get.return_value = "sec_3"

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock context provisions
        context_provisions = [(mock_context_elem, "context_1")]

        # Mock references found by LLM
        all_references = {(mock_context_elem, "context_1"): [("section", "3", "3")]}

        # Setup mocks
        with patch.object(service, "_find_context_provisions") as mock_find:
            with patch.object(service, "_identify_all_references_parallel") as mock_identify:
                with patch.object(service, "_should_inject_context") as mock_should_inject:
                    with patch.object(service, "_inject_act_reference") as mock_inject:
                        mock_find.return_value = context_provisions
                        mock_identify.return_value = all_references
                        mock_should_inject.return_value = True

                        # Mock XML handler to find the target element
                        mock_dependencies["xml_handler"].find_provision_by_type_and_number.return_value = (
                            mock_target_elem
                        )

                        # Call the method
                        result = service._inject_document_context(mock_tree, "Test Act", "schedule-123")

        # Should have injected context into 1 provision
        assert result == 1

        # Verify injection was called
        mock_inject.assert_called_once_with(mock_target_elem, "Test Act", "from context_1")

    def test_inject_document_context_with_range_references(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_document_context with range references (e.g., sections 3-5)."""
        service = KeelingService(mock_llm_kernel)

        # Create mock elements
        mock_context_elem = Mock()
        mock_target1 = Mock()
        mock_target1.get.return_value = "sec_3"
        mock_target2 = Mock()
        mock_target2.get.return_value = "sec_4"

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock context provisions with range reference
        context_provisions = [(mock_context_elem, "context_1")]
        all_references = {(mock_context_elem, "context_1"): [("section", "3", "5")]}  # Range from 3 to 5

        with patch.object(service, "_find_context_provisions") as mock_find:
            with patch.object(service, "_identify_all_references_parallel") as mock_identify:
                with patch.object(service, "_should_inject_context") as mock_should_inject:
                    with patch.object(service, "_inject_act_reference") as mock_inject:
                        mock_find.return_value = context_provisions
                        mock_identify.return_value = all_references
                        mock_should_inject.return_value = True

                        # Mock finding provisions in range
                        mock_dependencies["xml_handler"].find_provisions_in_range_by_number.return_value = [
                            mock_target1,
                            mock_target2,
                        ]

                        # Call the method
                        result = service._inject_document_context(mock_tree, "Test Act", "schedule-123")

        # Should have injected context into 2 provisions
        assert result == 2

        # Verify both injections
        assert mock_inject.call_count == 2

    def test_inject_document_context_multiple_sources(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_document_context when multiple context provisions reference the same target."""
        service = KeelingService(mock_llm_kernel)

        # Create mock elements
        mock_context1 = Mock()
        mock_context2 = Mock()
        mock_target = Mock()
        mock_target.get.return_value = "sec_3"

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Multiple context provisions referencing same target
        context_provisions = [(mock_context1, "context_1"), (mock_context2, "context_2")]
        all_references = {
            (mock_context1, "context_1"): [("section", "3", "3")],
            (mock_context2, "context_2"): [("section", "3", "3")],
        }

        with patch.object(service, "_find_context_provisions") as mock_find:
            with patch.object(service, "_identify_all_references_parallel") as mock_identify:
                with patch.object(service, "_should_inject_context") as mock_should_inject:
                    with patch.object(service, "_inject_act_reference") as mock_inject:
                        mock_find.return_value = context_provisions
                        mock_identify.return_value = all_references
                        mock_should_inject.return_value = True

                        # Mock finding same target for both references
                        mock_dependencies["xml_handler"].find_provision_by_type_and_number.return_value = mock_target

                        # Call the method
                        result = service._inject_document_context(mock_tree, "Test Act", "schedule-123")

        # Should inject only once per target
        assert result == 1

        # Verify combined source reference
        mock_inject.assert_called_once_with(mock_target, "Test Act", "from context_1, context_2")

    def test_identify_all_references_parallel_success(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_all_references_parallel with successful LLM calls."""
        service = KeelingService(mock_llm_kernel)

        # Mock context provisions
        elem1 = Mock()
        elem2 = Mock()
        context_provisions = [(elem1, "eid1"), (elem2, "eid2")]

        # Mock executor
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create futures with results
            future1 = create_completed_future([("section", "3", "3")])
            future2 = create_completed_future([("section", "5", "7")])

            mock_executor.submit.side_effect = [future1, future2]

            # Call method
            result = service._identify_all_references_parallel(context_provisions, "Test Act", "schedule-123")

        # Verify results
        assert len(result) == 2
        assert result[(elem1, "eid1")] == [("section", "3", "3")]
        assert result[(elem2, "eid2")] == [("section", "5", "7")]

    def test_identify_all_references_parallel_with_errors(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_all_references_parallel when some LLM calls fail."""
        service = KeelingService(mock_llm_kernel)

        # Mock context provisions
        elem1 = Mock()
        elem2 = Mock()
        context_provisions = [(elem1, "eid1"), (elem2, "eid2")]

        # Mock executor
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create futures - one succeeds, one fails
            future1 = create_completed_future([("section", "3", "3")])
            future2 = create_failed_future(RuntimeError("LLM error"))

            mock_executor.submit.side_effect = [future1, future2]

            # Call method
            result = service._identify_all_references_parallel(context_provisions, "Test Act", "schedule-123")

        # Should still have result for successful call
        assert len(result) == 2
        assert result[(elem1, "eid1")] == [("section", "3", "3")]
        assert result[(elem2, "eid2")] == []  # Empty list for failed call

    def test_identify_single_context_references_success(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_single_context_references with successful LLM response."""
        service = KeelingService(mock_llm_kernel)

        # Mock context element
        mock_elem = Mock()
        mock_elem.get.return_value = "context_1"

        # Mock XML handler
        mock_dependencies["xml_handler"].element_to_string.return_value = "<provision>context</provision>"

        # Mock LLM response
        csv_response = """provision_type,start_number,end_number
        section,3,3
        section,5,7
        regulation,2,2"""

        mock_llm_kernel.run_inference.return_value = csv_response

        # Call method
        result = service._identify_single_context_references(mock_elem, "Test Act", "schedule-123")

        # Verify results
        assert len(result) == 3
        assert ("section", "3", "3") in result
        assert ("section", "5", "7") in result
        assert ("regulation", "2", "2") in result

        # Verify LLM called correctly
        mock_llm_kernel.run_inference.assert_called_once_with(
            "IdentifyAmendmentReferences",
            "schedule-123",
            None,
            "context_1",
            context_provision="<provision>context</provision>",
            act_name="Test Act",
        )

    def test_identify_single_context_references_empty_response(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_single_context_references with empty LLM response."""
        service = KeelingService(mock_llm_kernel)

        # Mock context element
        mock_elem = Mock()
        mock_elem.get.return_value = "context_1"

        # Mock XML handler
        mock_dependencies["xml_handler"].element_to_string.return_value = "<provision>context</provision>"

        # Mock empty LLM response
        mock_llm_kernel.run_inference.return_value = ""

        # Call method
        result = service._identify_single_context_references(mock_elem, "Test Act", "schedule-123")

        # Should return empty list
        assert result == []

    def test_identify_single_context_references_exception(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_single_context_references when exception occurs."""
        service = KeelingService(mock_llm_kernel)

        # Mock context element
        mock_elem = Mock()
        mock_elem.get.return_value = "context_1"

        # Mock XML handler to raise exception
        mock_dependencies["xml_handler"].element_to_string.side_effect = Exception("XML error")

        # Call method
        result = service._identify_single_context_references(mock_elem, "Test Act", "schedule-123")

        # Should return empty list on error
        assert result == []

    def test_find_context_provisions(self, mock_llm_kernel, mock_dependencies):
        """Test _find_context_provisions method."""
        service = KeelingService(mock_llm_kernel)

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock provisions that contain the act name
        elem1 = Mock()
        elem2 = Mock()
        elem3 = Mock()

        # Mock find_provisions_containing_text
        mock_dependencies["xml_handler"].find_provisions_containing_text.return_value = [
            (elem1, "reg_18"),  # This will be context provision
            (elem2, "reg_19"),  # This will have amendment keywords
            (elem3, "reg_20"),  # This will be context provision
        ]

        # Mock get_text_content for each element
        mock_dependencies["xml_handler"].get_text_content.side_effect = [
            "The Test Act is amended in accordance with regulations 19 to 55",  # Context pattern, no amendment keywords
            "In section 3, insert new paragraph",  # Has amendment keywords
            "The Test Act has effect with the following modifications",  # Context pattern, no amendment keywords
        ]

        # Call method
        result = service._find_context_provisions(mock_tree, "Test Act")

        # Should return only elements with context patterns but no amendment keywords
        assert len(result) == 2
        assert (elem1, "reg_18") in result
        assert (elem3, "reg_20") in result
        assert (elem2, "reg_19") not in result  # Excluded due to amendment keywords

    def test_should_inject_context_already_contains_act(self, mock_llm_kernel, mock_dependencies):
        """Test _should_inject_context when element already contains act name."""
        service = KeelingService(mock_llm_kernel)

        # Mock element
        mock_elem = Mock()
        mock_elem.get.return_value = "sec_3"

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Element already contains act name
        mock_dependencies["xml_handler"].get_text_content.return_value = "This amends the Test Act by inserting..."

        # Call method
        result = service._should_inject_context(mock_elem, "Test Act", mock_tree)

        # Should return False as element already contains act name
        assert result is False

    def test_should_inject_context_ancestor_contains_act(self, mock_llm_kernel, mock_dependencies):
        """Test _should_inject_context when ancestor provision contains act name."""
        service = KeelingService(mock_llm_kernel)

        # Mock element and parent
        mock_elem = Mock()
        mock_elem.get.return_value = "para_a"

        mock_parent = Mock()
        mock_parent.get.side_effect = lambda attr: "prov1" if attr == "class" else None
        mock_parent.getparent.return_value = None

        mock_elem.getparent.return_value = mock_parent

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Element doesn't contain act name, but parent does
        mock_dependencies["xml_handler"].get_text_content.side_effect = [
            "Insert new text",  # Element text
            "The Test Act is amended by...",  # Parent text
        ]

        # Call method
        result = service._should_inject_context(mock_elem, "Test Act", mock_tree)

        # Should return False as ancestor contains act name
        assert result is False

    def test_should_inject_context_no_amendment_keywords(self, mock_llm_kernel, mock_dependencies):
        """Test _should_inject_context when element has no amendment keywords."""
        service = KeelingService(mock_llm_kernel)

        # Mock element with no parent
        mock_elem = Mock()
        mock_elem.get.return_value = "sec_3"
        mock_elem.getparent.return_value = None

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Element doesn't contain act name or amendment keywords
        mock_dependencies["xml_handler"].get_text_content.return_value = "This provision does something else"

        # Call method
        result = service._should_inject_context(mock_elem, "Test Act", mock_tree)

        # Should return False as no amendment keywords
        assert result is False

    def test_should_inject_context_all_conditions_met(self, mock_llm_kernel, mock_dependencies):
        """Test _should_inject_context when all conditions are met for injection."""
        service = KeelingService(mock_llm_kernel)

        # Mock element with no parent
        mock_elem = Mock()
        mock_elem.get.return_value = "sec_3"
        mock_elem.getparent.return_value = None

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Element contains amendment keywords but not act name
        mock_dependencies["xml_handler"].get_text_content.return_value = "Insert new paragraph after subsection (2)"

        # Call method
        result = service._should_inject_context(mock_elem, "Test Act", mock_tree)

        # Should return True as all conditions met
        assert result is True

    def test_inject_act_reference(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_act_reference method."""
        service = KeelingService(mock_llm_kernel)

        # Mock element
        mock_elem = Mock()

        # Call method
        service._inject_act_reference(mock_elem, "Test Act 2020", "from reg_18, reg_25")

        # Verify comment injection
        mock_dependencies["xml_handler"].inject_xml_comment.assert_called_once_with(
            mock_elem, " Amendment context: Test Act 2020 (from reg_18, reg_25) "
        )

        # Verify attribute setting
        assert mock_dependencies["xml_handler"].set_namespaced_attribute.call_count == 2
        mock_dependencies["xml_handler"].set_namespaced_attribute.assert_any_call(
            mock_elem, mock_dependencies["xml_handler"].UKL_URI, "contextAct", "Test Act 2020"
        )
        mock_dependencies["xml_handler"].set_namespaced_attribute.assert_any_call(
            mock_elem, mock_dependencies["xml_handler"].UKL_URI, "contextSource", "from reg_18, reg_25"
        )

    def test_find_keyword_based_candidates_with_ancestor_mention(self, mock_llm_kernel, mock_dependencies):
        """Test _find_keyword_based_candidates when act is mentioned in ancestor."""
        service = KeelingService(mock_llm_kernel)

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock element that contains amendment keywords
        mock_elem = Mock()

        # Mock ancestor provision that contains act name
        mock_ancestor = Mock()
        mock_ancestor.get.side_effect = lambda attr, default=None: "prov1" if attr == "class" else default
        mock_ancestor.getparent.return_value = None

        # Set up parent chain
        mock_elem.getparent.return_value = mock_ancestor

        # Mock finding provisions with keywords
        mock_dependencies["xml_handler"].find_provisions_containing_text.return_value = [(mock_elem, "para_a")]

        # Mock text content - need to return text/comments for both element and ancestor
        get_text_calls = [
            "insert new text",  # Element text (has amendment keyword, no act name)
            "The Test Act is amended as follows",  # Ancestor text (has act name)
        ]

        get_comment_calls = [
            "",  # Element comments
            "",  # Ancestor comments
        ]

        mock_dependencies["xml_handler"].get_text_content.side_effect = get_text_calls
        mock_dependencies["xml_handler"].get_comment_content.side_effect = get_comment_calls

        # Mock element_to_string
        mock_dependencies["xml_handler"].element_to_string.return_value = "<para>content</para>"

        # Call method
        result = service._find_keyword_based_candidates(mock_tree, "Test Act", "schedule-123")

        # Should find the candidate as act is mentioned in ancestor
        assert len(result) == 1
        assert result[0] == ("<para>content</para>", "para_a")

    def test_filter_descendant_candidates(self, mock_llm_kernel, mock_dependencies):
        """Test _filter_descendant_candidates removes descendant provisions."""
        service = KeelingService(mock_llm_kernel)

        # Create candidates where some are descendants of others
        candidates = [
            ("<section>1</section>", "sec_1"),
            ("<para>a</para>", "sec_1__para_a"),  # Descendant of sec_1
            ("<section>2</section>", "sec_2"),
            ("<subsec>1</subsec>", "sec_2__subsec_1"),  # Descendant of sec_2
            ("<para>b</para>", "sec_2__subsec_1__para_b"),  # Descendant of sec_2__subsec_1
        ]

        # Call method
        result = service._filter_descendant_candidates(candidates)

        # Should only keep top-level provisions
        assert len(result) == 2
        assert ("<section>1</section>", "sec_1") in result
        assert ("<section>2</section>", "sec_2") in result

        # Descendants should be filtered out
        assert ("<para>a</para>", "sec_1__para_a") not in result
        assert ("<subsec>1</subsec>", "sec_2__subsec_1") not in result
        assert ("<para>b</para>", "sec_2__subsec_1__para_b") not in result

    def test_inject_crossheading_context_with_ancestors(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_crossheading_context with nested crossheadings."""
        service = KeelingService(mock_llm_kernel)

        # Create mock schedule
        mock_schedule = Mock()
        mock_schedule.get.return_value = "sched_1"

        # Create mock crossheading
        mock_crossheading = Mock()
        mock_crossheading.get.return_value = "xhdg_2"

        # Create mock heading element
        mock_heading_elem = Mock()
        mock_crossheading.find.return_value = mock_heading_elem

        # Create mock child provision
        mock_child = Mock()
        mock_child.get.return_value = "para_5"

        # Setup mock tree
        mock_tree = Mock()
        mock_tree.xpath.return_value = [mock_schedule]

        # Setup mock XML handler methods
        mock_dependencies["xml_handler"].get_schedule_heading_text.return_value = "Schedule 1"
        mock_dependencies["xml_handler"].get_crossheadings_in_schedule.return_value = [mock_crossheading]
        mock_dependencies["xml_handler"].get_text_content.return_value = "Current crossheading"
        mock_dependencies["xml_handler"].get_ancestor_crossheading_contexts.return_value = ["Parent crossheading"]
        mock_dependencies["xml_handler"].get_crossheading_child_provisions.return_value = [mock_child]

        # Call the method
        result = service._inject_crossheading_context(mock_tree)

        # Should have injected context with full hierarchy
        assert result == 1

        # Verify injection with combined context
        mock_dependencies["xml_handler"].inject_xml_comment.assert_called_once_with(
            mock_child, " Crossheading context: Schedule 1 > Parent crossheading > Current crossheading "
        )

    def test_identify_single_context_references_malformed_csv(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_single_context_references with malformed CSV response."""
        service = KeelingService(mock_llm_kernel)

        # Mock context element
        mock_elem = Mock()
        mock_elem.get.return_value = "context_1"

        # Mock XML handler
        mock_dependencies["xml_handler"].element_to_string.return_value = "<provision>context</provision>"

        # Mock malformed LLM response where all rows have issues
        csv_response = """provision_type,start_number,end_number
        section,3
        regulation,5
        ,2,2
        """

        mock_llm_kernel.run_inference.return_value = csv_response

        # Call method
        result = service._identify_single_context_references(mock_elem, "Test Act", "schedule-123")

        # The implementation is lenient and tries to parse what it can,
        # but in this case only the row with missing provision_type would be parsed
        assert len(result) == 1
        assert ("", "2", "2") in result  # The row with missing provision_type

    def test_identify_single_context_references_truly_malformed_csv(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_single_context_references with CSV that has no valid rows."""
        service = KeelingService(mock_llm_kernel)

        # Mock context element
        mock_elem = Mock()
        mock_elem.get.return_value = "context_1"

        # Mock XML handler
        mock_dependencies["xml_handler"].element_to_string.return_value = "<provision>context</provision>"

        # Mock CSV response where no row has enough columns
        csv_response = """provision_type,start_number,end_number
        section
        regulation,5
        incomplete
        """

        mock_llm_kernel.run_inference.return_value = csv_response

        # Call method
        result = service._identify_single_context_references(mock_elem, "Test Act", "schedule-123")

        # Should return empty list as no row has 3 columns
        assert result == []

    def test_inject_document_context_no_target_found(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_document_context when target provisions are not found."""
        service = KeelingService(mock_llm_kernel)

        # Create mock context element
        mock_context_elem = Mock()

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock context provisions
        context_provisions = [(mock_context_elem, "context_1")]

        # Mock references found by LLM
        all_references = {(mock_context_elem, "context_1"): [("section", "999", "999")]}  # Non-existent section

        with patch.object(service, "_find_context_provisions") as mock_find:
            with patch.object(service, "_identify_all_references_parallel") as mock_identify:
                mock_find.return_value = context_provisions
                mock_identify.return_value = all_references

                # Mock XML handler to not find the target
                mock_dependencies["xml_handler"].find_provision_by_type_and_number.return_value = None

                # Call the method
                result = service._inject_document_context(mock_tree, "Test Act", "schedule-123")

        # Should not inject anything as target not found
        assert result == 0

    def test_should_inject_context_non_provision_ancestor(self, mock_llm_kernel, mock_dependencies):
        """Test _should_inject_context when element has non-provision ancestors."""
        service = KeelingService(mock_llm_kernel)

        # Mock element
        mock_elem = Mock()
        mock_elem.get.return_value = "para_a"

        # Create non-provision parent (e.g., a wrapper element)
        mock_wrapper = Mock()
        mock_wrapper.get.side_effect = lambda attr: "wrapper" if attr == "class" else None

        # Create provision grandparent
        mock_grandparent = Mock()
        mock_grandparent.get.side_effect = lambda attr: "section" if attr == "name" else None
        mock_grandparent.getparent.return_value = None

        # Set up parent chain
        mock_wrapper.getparent.return_value = mock_grandparent
        mock_elem.getparent.return_value = mock_wrapper

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Element contains amendment keywords but not act name
        # Grandparent (provision) contains act name
        mock_dependencies["xml_handler"].get_text_content.side_effect = [
            "insert new text",  # Element text
            "The Test Act is amended...",  # Grandparent text (provision)
        ]

        # Call method
        result = service._should_inject_context(mock_elem, "Test Act", mock_tree)

        # Should return False as ancestor provision contains act name
        assert result is False

    def test_identify_single_context_references_header_in_response(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_single_context_references when CSV has header row."""
        service = KeelingService(mock_llm_kernel)

        # Mock context element
        mock_elem = Mock()
        mock_elem.get.return_value = "context_1"

        # Mock XML handler
        mock_dependencies["xml_handler"].element_to_string.return_value = "<provision>context</provision>"

        # Mock LLM response with header row
        csv_response = """provision_type,start_number,end_number
        section,3,3
        section,5,7"""

        mock_llm_kernel.run_inference.return_value = csv_response

        # Call method
        result = service._identify_single_context_references(mock_elem, "Test Act", "schedule-123")

        # Should skip header and return 2 references
        assert len(result) == 2
        assert ("section", "3", "3") in result
        assert ("section", "5", "7") in result

    def test_identify_single_context_references_whitespace_handling(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_single_context_references handles whitespace in CSV."""
        service = KeelingService(mock_llm_kernel)

        # Mock context element
        mock_elem = Mock()
        mock_elem.get.return_value = "context_1"

        # Mock XML handler
        mock_dependencies["xml_handler"].element_to_string.return_value = "<provision>context</provision>"

        # Mock LLM response with extra whitespace
        csv_response = """

        provision_type, start_number , end_number
        section  ,  3  ,  3

        regulation  ,  2  ,  2

        """

        mock_llm_kernel.run_inference.return_value = csv_response

        # Call method
        result = service._identify_single_context_references(mock_elem, "Test Act", "schedule-123")

        # Should handle whitespace correctly
        assert len(result) == 2
        assert ("section", "3", "3") in result
        assert ("regulation", "2", "2") in result

    def test_find_keyword_based_candidates_group_ancestor(self, mock_llm_kernel, mock_dependencies):
        """Test _find_keyword_based_candidates when ancestor is a Group element."""
        service = KeelingService(mock_llm_kernel)

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock element that contains amendment keywords
        mock_elem = Mock()

        # Mock Group ancestor that contains act name
        mock_group = Mock()
        mock_group.get.side_effect = lambda attr, default=None: "schProv1Group" if attr == "class" else default
        mock_group.getparent.return_value = None

        # Set up parent chain
        mock_elem.getparent.return_value = mock_group

        # Mock finding provisions with keywords
        mock_dependencies["xml_handler"].find_provisions_containing_text.return_value = [(mock_elem, "para_1")]

        # Mock text content
        get_text_calls = [
            "insert new paragraph",  # Element text (no act name)
            "Amendments to Test Act",  # Group ancestor text (has act name)
        ]

        get_comment_calls = [
            "",  # Element comments
            "",  # Group ancestor comments
        ]

        mock_dependencies["xml_handler"].get_text_content.side_effect = get_text_calls
        mock_dependencies["xml_handler"].get_comment_content.side_effect = get_comment_calls

        # Mock element_to_string
        mock_dependencies["xml_handler"].element_to_string.return_value = "<para>content</para>"

        # Call method
        result = service._find_keyword_based_candidates(mock_tree, "Test Act", "schedule-123")

        # Should find the candidate as act is mentioned in Group ancestor
        assert len(result) == 1
        assert result[0] == ("<para>content</para>", "para_1")

    def test_find_keyword_based_candidates_schedule_ancestor(self, mock_llm_kernel, mock_dependencies):
        """Test _find_keyword_based_candidates when ancestor is a schedule."""
        service = KeelingService(mock_llm_kernel)

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock element that contains amendment keywords
        mock_elem = Mock()

        # Mock schedule ancestor that contains act name
        mock_schedule = Mock()
        mock_schedule.get.side_effect = lambda attr, default=None: (
            "schedule" if attr == "name" else "sch" if attr == "class" else default
        )
        mock_schedule.getparent.return_value = None

        # Set up parent chain
        mock_elem.getparent.return_value = mock_schedule

        # Mock finding provisions with keywords
        mock_dependencies["xml_handler"].find_provisions_containing_text.return_value = [(mock_elem, "para_1")]

        # Mock text content
        get_text_calls = [
            "omit paragraph (b)",  # Element text (no act name)
            "SCHEDULE 1 - Amendments to Test Act",  # Schedule ancestor text
        ]

        get_comment_calls = [
            "",  # Element comments
            "",  # Schedule ancestor comments
        ]

        mock_dependencies["xml_handler"].get_text_content.side_effect = get_text_calls
        mock_dependencies["xml_handler"].get_comment_content.side_effect = get_comment_calls

        # Mock element_to_string
        mock_dependencies["xml_handler"].element_to_string.return_value = "<para>content</para>"

        # Call method
        result = service._find_keyword_based_candidates(mock_tree, "Test Act", "schedule-123")

        # Should find the candidate as act is mentioned in schedule ancestor
        assert len(result) == 1

    def test_inject_document_context_should_not_inject(self, mock_llm_kernel, mock_dependencies):
        """Test _inject_document_context when _should_inject_context returns False."""
        service = KeelingService(mock_llm_kernel)

        # Create mock elements
        mock_context_elem = Mock()
        mock_target_elem = Mock()
        mock_target_elem.get.return_value = "sec_3"

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock context provisions
        context_provisions = [(mock_context_elem, "context_1")]

        # Mock references found by LLM
        all_references = {(mock_context_elem, "context_1"): [("section", "3", "3")]}

        with patch.object(service, "_find_context_provisions") as mock_find:
            with patch.object(service, "_identify_all_references_parallel") as mock_identify:
                with patch.object(service, "_should_inject_context") as mock_should_inject:
                    mock_find.return_value = context_provisions
                    mock_identify.return_value = all_references
                    mock_should_inject.return_value = False  # Should not inject

                    # Mock XML handler to find the target element
                    mock_dependencies["xml_handler"].find_provision_by_type_and_number.return_value = mock_target_elem

                    # Call the method
                    result = service._inject_document_context(mock_tree, "Test Act", "schedule-123")

        # Should not inject anything
        assert result == 0

    def test_find_keyword_based_candidates_act_in_comments(self, mock_llm_kernel, mock_dependencies):
        """Test _find_keyword_based_candidates when act name is in XML comments."""
        service = KeelingService(mock_llm_kernel)

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock element
        mock_elem = Mock()
        mock_elem.getparent.return_value = None

        # Mock finding provisions with keywords
        mock_dependencies["xml_handler"].find_provisions_containing_text.return_value = [(mock_elem, "sec_1")]

        # Mock text content - no act name in text
        mock_dependencies["xml_handler"].get_text_content.return_value = "insert new subsection"

        # Mock comment content - act name in comments
        mock_dependencies["xml_handler"].get_comment_content.return_value = "Amendment context: Test Act"

        # Mock element_to_string
        mock_dependencies["xml_handler"].element_to_string.return_value = "<section>content</section>"

        # Call method
        result = service._find_keyword_based_candidates(mock_tree, "Test Act", "schedule-123")

        # Should find the candidate as act is mentioned in comments
        assert len(result) == 1
        assert result[0] == ("<section>content</section>", "sec_1")

    def test_find_keyword_based_candidates_ancestor_comment(self, mock_llm_kernel, mock_dependencies):
        """Test _find_keyword_based_candidates when act name is in ancestor's comments."""
        service = KeelingService(mock_llm_kernel)

        # Mock tree
        mock_tree = Mock(spec=etree.ElementTree)

        # Mock element
        mock_elem = Mock()

        # Mock ancestor
        mock_ancestor = Mock()
        mock_ancestor.get.side_effect = lambda attr, default=None: "regulation" if attr == "name" else default
        mock_ancestor.getparent.return_value = None

        mock_elem.getparent.return_value = mock_ancestor

        # Mock finding provisions with keywords
        mock_dependencies["xml_handler"].find_provisions_containing_text.return_value = [(mock_elem, "para_a")]

        # Mock text/comment content
        mock_dependencies["xml_handler"].get_text_content.side_effect = [
            "insert new text",  # Element text
            "Some regulation text",  # Ancestor text
        ]

        mock_dependencies["xml_handler"].get_comment_content.side_effect = [
            "",  # Element comment
            "Context: Test Act amendments",  # Ancestor comment with act name
        ]

        # Mock element_to_string
        mock_dependencies["xml_handler"].element_to_string.return_value = "<para>content</para>"

        # Call method
        result = service._find_keyword_based_candidates(mock_tree, "Test Act", "schedule-123")

        # Should find the candidate
        assert len(result) == 1

    def test_identify_all_references_parallel_empty_references(self, mock_llm_kernel, mock_dependencies):
        """Test _identify_all_references_parallel when LLM returns empty references."""
        service = KeelingService(mock_llm_kernel)

        # Mock context provisions
        elem1 = Mock()
        context_provisions = [(elem1, "eid1")]

        # Mock executor
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create future with empty result
            future = create_completed_future([])  # Empty list of references
            mock_executor.submit.return_value = future

            # Call method
            result = service._identify_all_references_parallel(context_provisions, "Test Act", "schedule-123")

        # Should have entry with empty list
        assert len(result) == 1
        assert result[(elem1, "eid1")] == []

    def test_process_amendment_group_sequential_with_parse_xml_injection(self, mock_llm_kernel, mock_dependencies):
        """Test _process_amendment_group sequential processing with amendment ID injection."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create two amendments
        amendment1 = Mock()
        amendment1.amendment_id = "id1"
        amendment1.affected_provision = "sec_1"

        amendment2 = Mock()
        amendment2.amendment_id = "id2"
        amendment2.affected_provision = "sec_1"

        # Mock successful responses
        response1 = "<section>amended1</section>"
        response2 = "<section>amended2</section>"

        # Mock parsed elements
        parsed_elem1 = Mock()
        parsed_elem2 = Mock()

        # For the second amendment, we need to mock the working tree creation
        mock_target = Mock()
        mock_target.get.return_value = "sec_1"
        mock_parent = Mock()
        mock_target.getparent.return_value = mock_parent

        # Mock find_element_by_eid to return target
        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = [
            mock_target,  # For working tree creation
        ]

        # Mock parse_xml_string to return elements
        mock_dependencies["xml_handler"].parse_xml_string.side_effect = [
            parsed_elem1,  # First response parsing
            parsed_elem1,  # Parse last successful for working tree
            parsed_elem2,  # Second response parsing
        ]

        # Mock inject_amendment_id
        mock_dependencies["xml_handler"].inject_amendment_id.side_effect = [None, None]

        # Mock element_to_string to return responses
        mock_dependencies["xml_handler"].element_to_string.side_effect = [
            response1,
            response2,
        ]

        with patch.object(service, "_fetch_single_amendment_response") as mock_fetch:
            mock_fetch.side_effect = [response1, response2]

            result = service._process_amendment_group([amendment1, amendment2], Mock(), "schedule-123", "sec_1")

        # Should have both responses
        assert len(result["responses"]) == 2
        assert result["responses"]["id1"] == response1
        assert result["responses"]["id2"] == response2

        # Verify inject_amendment_id was called
        assert mock_dependencies["xml_handler"].inject_amendment_id.call_count == 2

    @patch("os.path.getsize")
    def test_apply_amendments_each_place_not_in_validated_patterns(
        self, mock_getsize, mock_llm_kernel, mock_dependencies
    ):
        """Test that EACH_PLACE amendments without validated patterns need LLM."""
        mock_getsize.return_value = 1000
        service = KeelingService(mock_llm_kernel)

        # Create EACH_PLACE amendment
        each_place_amendment = Mock(spec=Amendment)
        each_place_amendment.whole_provision = False
        each_place_amendment.amendment_id = "each1"
        each_place_amendment.location = AmendmentLocation.EACH_PLACE
        each_place_amendment.affected_provision = "sec_1"
        each_place_amendment.amendment_type = Mock(value="SUBSTITUTION")
        each_place_amendment.source_eid = "source_1"
        each_place_amendment.source = "s. 1"

        amendments = [each_place_amendment]

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 0

        # Mock tracker
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": True,
            "stats": {"total": 1, "applied": 1, "with_error_comments": 0, "unresolved": 0},
            "unresolved_amendments": [],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = []

        # Mock pattern extraction/validation to return empty (no validated patterns)
        with patch.object(service, "_extract_and_validate_patterns") as mock_extract_validate:
            mock_extract_validate.return_value = ({}, ["each1"])  # No validated patterns

            # Mock LLM responses
            with patch.object(service, "_fetch_llm_responses_parallel") as mock_fetch:
                mock_fetch.return_value = {"each1": "<amended>content</amended>"}

                # Mock processor
                mock_dependencies["processor"].apply_amendment.return_value = (True, None)

                service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify the amendment was included in LLM fetch
        mock_fetch.assert_called_once()
        fetch_args = mock_fetch.call_args[0][0]
        assert len(fetch_args) == 1
        assert fetch_args[0] == each_place_amendment

    def test_apply_single_amendment_each_place_with_validated_pattern(self, mock_llm_kernel, mock_dependencies):
        """Test applying EACH_PLACE amendment with validated pattern."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create EACH_PLACE amendment
        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.location = AmendmentLocation.EACH_PLACE
        amendment.affected_provision = "sec_1"

        # Mock output act
        output_act = Mock()

        # Validated patterns
        validated_patterns = {"each1": {"find_text": "company", "replace_text": "corporation"}}

        # Mock processor to apply successfully
        mock_dependencies["processor"].apply_each_place_amendment.return_value = (True, None)

        # Mock tracker
        mock_dependencies["tracker"].mark_applying.return_value = True
        mock_dependencies["tracker"].mark_applied.return_value = True

        # Call method
        service._apply_single_amendment(amendment, output_act, "schedule-123", {}, validated_patterns)

        # Verify apply_each_place_amendment was called
        mock_dependencies["processor"].apply_each_place_amendment.assert_called_once_with(
            amendment,
            output_act,
            service._amending_bill,
            {"find_text": "company", "replace_text": "corporation"},
            "schedule-123",
        )

        # Verify amendment marked as applied
        mock_dependencies["tracker"].mark_applied.assert_called_once()

    def test_apply_single_amendment_each_place_pattern_fails(self, mock_llm_kernel, mock_dependencies):
        """Test when validated pattern unexpectedly fails to apply."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create EACH_PLACE amendment
        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.location = AmendmentLocation.EACH_PLACE
        amendment.affected_provision = "sec_1"

        # Mock output act
        output_act = Mock()

        # Validated patterns
        validated_patterns = {"each1": {"find_text": "company", "replace_text": "corporation"}}

        # Mock processor to fail
        error_msg = "Pattern no longer matches"
        mock_dependencies["processor"].apply_each_place_amendment.return_value = (False, error_msg)

        # Mock tracker
        mock_dependencies["tracker"].mark_applying.return_value = True

        # Call method
        with patch("app.services.keeling_service.logger") as mock_logger:
            service._apply_single_amendment(amendment, output_act, "schedule-123", {}, validated_patterns)

        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Validated pattern failed to apply for amendment each1" in error_call
        assert "This should not happen" in error_call
        assert error_msg in error_call

        # Verify amendment marked as failed
        mock_dependencies["tracker"].mark_failed.assert_called_once_with("each1", error_msg, processing_time=ANY)

    def test_extract_and_validate_patterns(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_and_validate_patterns method."""
        service = KeelingService(mock_llm_kernel)

        # Create amendments
        each_place1 = Mock()
        each_place1.amendment_id = "each1"
        each_place1.location = AmendmentLocation.EACH_PLACE

        each_place2 = Mock()
        each_place2.amendment_id = "each2"
        each_place2.location = AmendmentLocation.EACH_PLACE

        regular_amendment = Mock()
        regular_amendment.amendment_id = "reg1"
        regular_amendment.location = AmendmentLocation.AFTER

        amendments = [each_place1, each_place2, regular_amendment]

        # Mock pattern extraction
        with patch.object(service, "_extract_amendment_patterns") as mock_extract:
            mock_extract.return_value = {
                "each1": {"find_text": "company", "replace_text": "corporation"},
                "each2": {"find_text": "section", "replace_text": "clause"},
            }

            # Mock pattern validation
            with patch.object(service, "_validate_pattern_application") as mock_validate:
                # First pattern validates successfully, second fails
                mock_validate.side_effect = [True, False]

                # Mock get_amendment_id
                with patch("app.services.keeling_service.get_amendment_id") as mock_get_id:
                    mock_get_id.side_effect = lambda a: a.amendment_id

                    # Call method
                    validated, failures = service._extract_and_validate_patterns(amendments, "schedule-123")

        # Verify extraction was called with all amendments
        mock_extract.assert_called_once_with(amendments, "schedule-123")

        # Verify validation was called for each pattern
        assert mock_validate.call_count == 2
        mock_validate.assert_any_call(each_place1, {"find_text": "company", "replace_text": "corporation"})
        mock_validate.assert_any_call(each_place2, {"find_text": "section", "replace_text": "clause"})

        # Verify results
        assert len(validated) == 1
        assert "each1" in validated
        assert validated["each1"] == {"find_text": "company", "replace_text": "corporation"}

        assert len(failures) == 1
        assert "each2" in failures

        # Verify failed amendment was marked
        mock_dependencies["tracker"].mark_failed.assert_called_once_with(
            "each2", "Pattern validation failed - will use LLM approach", error_location="pattern_validation"
        )

    def test_extract_and_validate_patterns_amendment_not_found(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_and_validate_patterns when amendment lookup fails."""
        service = KeelingService(mock_llm_kernel)

        amendments = []  # Empty list, but patterns exist

        # Mock pattern extraction returning a pattern
        with patch.object(service, "_extract_amendment_patterns") as mock_extract:
            mock_extract.return_value = {"missing_id": {"find_text": "test", "replace_text": "example"}}

            # Mock get_amendment_id
            with patch("app.services.keeling_service.get_amendment_id") as mock_get_id:
                mock_get_id.return_value = None

                with patch("app.services.keeling_service.logger") as mock_logger:
                    # Call method
                    validated, failures = service._extract_and_validate_patterns(amendments, "schedule-123")

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Could not find amendment for pattern validation: missing_id" in warning_msg

        # Verify results
        assert len(validated) == 0
        assert len(failures) == 1
        assert "missing_id" in failures

    def test_extract_amendment_patterns(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_amendment_patterns method."""
        service = KeelingService(mock_llm_kernel)

        # Create amendments
        each_place1 = Mock()
        each_place1.amendment_id = "each1"
        each_place1.location = AmendmentLocation.EACH_PLACE

        each_place2 = Mock()
        each_place2.amendment_id = "each2"
        each_place2.location = AmendmentLocation.EACH_PLACE

        regular = Mock()
        regular.amendment_id = "reg1"
        regular.location = AmendmentLocation.AFTER

        amendments = [each_place1, each_place2, regular]

        # Mock executor
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create futures
            future1 = create_completed_future({"find_text": "company", "replace_text": "corporation"})
            future2 = create_completed_future(None)  # Extraction fails

            mock_executor.submit.side_effect = [future1, future2]

            # Mock event logging
            with patch("app.services.keeling_service.event") as mock_event:
                # Call method
                result = service._extract_amendment_patterns(amendments, "schedule-123")

        # Verify only EACH_PLACE amendments were processed
        assert mock_executor.submit.call_count == 2

        # Verify results
        assert len(result) == 1
        assert "each1" in result
        assert result["each1"] == {"find_text": "company", "replace_text": "corporation"}

        # Verify events were logged
        success_events = [call for call in mock_event.call_args_list if "PATTERN_EXTRACTION_SUCCESS" in str(call)]
        failed_events = [call for call in mock_event.call_args_list if "PATTERN_EXTRACTION_FAILED" in str(call)]

        assert len(success_events) == 1
        assert len(failed_events) == 1

    def test_extract_amendment_patterns_with_exception(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_amendment_patterns when extraction throws exception."""
        service = KeelingService(mock_llm_kernel)

        each_place = Mock()
        each_place.amendment_id = "each1"
        each_place.location = AmendmentLocation.EACH_PLACE

        amendments = [each_place]

        # Mock executor
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Create future that raises exception
            future = create_failed_future(RuntimeError("Extraction error"))
            mock_executor.submit.return_value = future

            # Mock event logging
            with patch("app.services.keeling_service.event") as mock_event:
                # Call method
                result = service._extract_amendment_patterns(amendments, "schedule-123")

        # Verify empty result
        assert len(result) == 0

        # Verify error event was logged
        error_events = [call for call in mock_event.call_args_list if "PATTERN_EXTRACTION_FAILED" in str(call)]
        assert len(error_events) == 1

    def test_extract_amendment_patterns_no_each_place(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_amendment_patterns with no EACH_PLACE amendments."""
        service = KeelingService(mock_llm_kernel)

        # Only regular amendments
        regular = Mock()
        regular.location = AmendmentLocation.AFTER

        amendments = [regular]

        # Call method
        result = service._extract_amendment_patterns(amendments, "schedule-123")

        # Should return empty dict immediately
        assert result == {}

    def test_validate_pattern_application_success(self, mock_llm_kernel, mock_dependencies):
        """Test _validate_pattern_application successful validation."""
        service = KeelingService(mock_llm_kernel)
        service._target_act = Mock()

        # Create amendment
        amendment = Mock()
        amendment.affected_provision = "sec_1"
        amendment.amendment_type = AmendmentType.SUBSTITUTION

        # Pattern to validate
        pattern = {"find_text": "company", "replace_text": "corporation"}

        # Mock target element
        mock_target = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_target

        # Mock successful validation
        validation_changes = [{"element": mock_target, "occurrences": 3}]

        # Mock the processor method
        mock_dependencies["processor"]._replace_text_occurrences_iteratively.side_effect = (
            lambda elem, find, replace, type, changes: changes.extend(validation_changes)
        )

        with patch("copy.deepcopy") as mock_deepcopy:
            mock_deepcopy.return_value = Mock()  # Return mock validation element

            # Call method
            result = service._validate_pattern_application(amendment, pattern)

        # Should return True
        assert result is True

        # Verify processor was called
        mock_dependencies["processor"]._replace_text_occurrences_iteratively.assert_called_once()

    def test_validate_pattern_application_no_find_text(self, mock_llm_kernel, mock_dependencies):
        """Test _validate_pattern_application with missing find_text."""
        service = KeelingService(mock_llm_kernel)

        amendment = Mock()
        pattern = {"find_text": "", "replace_text": "corporation"}

        with patch("app.services.keeling_service.logger") as mock_logger:
            result = service._validate_pattern_application(amendment, pattern)

        assert result is False
        mock_logger.debug.assert_called_with("Pattern validation failed: no find_text")

    def test_validate_pattern_application_target_not_found(self, mock_llm_kernel, mock_dependencies):
        """Test _validate_pattern_application when target element not found."""
        service = KeelingService(mock_llm_kernel)
        service._target_act = Mock()

        amendment = Mock()
        amendment.affected_provision = "sec_999"

        pattern = {"find_text": "company", "replace_text": "corporation"}

        # Mock target not found
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = None

        with patch("app.services.keeling_service.logger") as mock_logger:
            result = service._validate_pattern_application(amendment, pattern)

        assert result is False
        mock_logger.debug.assert_called_with("Pattern validation failed: target element sec_999 not found")

    def test_validate_pattern_application_no_occurrences(self, mock_llm_kernel, mock_dependencies):
        """Test _validate_pattern_application when pattern finds no occurrences."""
        service = KeelingService(mock_llm_kernel)
        service._target_act = Mock()

        amendment = Mock()
        amendment.affected_provision = "sec_1"
        amendment.amendment_type = AmendmentType.SUBSTITUTION

        pattern = {"find_text": "nonexistent", "replace_text": "replacement"}

        # Mock target element
        mock_target = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_target

        # Mock no occurrences found
        mock_dependencies["processor"]._replace_text_occurrences_iteratively.side_effect = (
            lambda elem, find, replace, type, changes: None
        )  # No changes added

        with patch("copy.deepcopy") as mock_deepcopy:
            mock_deepcopy.return_value = Mock()

            with patch("app.services.keeling_service.logger") as mock_logger:
                result = service._validate_pattern_application(amendment, pattern)

        assert result is False
        mock_logger.debug.assert_called_with("Pattern validation failed: no occurrences of 'nonexistent' found")

    def test_validate_pattern_application_exception(self, mock_llm_kernel, mock_dependencies):
        """Test _validate_pattern_application when exception occurs during validation."""
        service = KeelingService(mock_llm_kernel)
        service._target_act = Mock()

        amendment = Mock()
        amendment.affected_provision = "sec_1"

        pattern = {"find_text": "company", "replace_text": "corporation"}

        # Mock target element
        mock_target = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_target

        # Mock exception during processing
        mock_dependencies["processor"]._replace_text_occurrences_iteratively.side_effect = RuntimeError(
            "Processing error"
        )

        with patch("copy.deepcopy") as mock_deepcopy:
            mock_deepcopy.return_value = Mock()

            with patch("app.services.keeling_service.logger") as mock_logger:
                result = service._validate_pattern_application(amendment, pattern)

        assert result is False
        mock_logger.debug.assert_called_with("Pattern validation failed with exception: Processing error")

    def test_validate_pattern_application_outer_exception(self, mock_llm_kernel, mock_dependencies):
        """Test _validate_pattern_application when outer exception occurs."""
        service = KeelingService(mock_llm_kernel)
        service._target_act = Mock()

        amendment = Mock()
        amendment.affected_provision = "sec_1"
        pattern = {"find_text": "test", "replace_text": "replacement"}

        # Mock find_element_by_eid to raise an exception
        mock_dependencies["xml_handler"].find_element_by_eid.side_effect = Exception("Unexpected error")

        with patch("app.services.keeling_service.logger") as mock_logger:
            result = service._validate_pattern_application(amendment, pattern)

        assert result is False
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        assert "Pattern validation error:" in error_msg

    def test_extract_single_pattern_success(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_single_pattern successful extraction."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        # Create amendment
        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.source_eid = "source_1"

        # Mock source element
        mock_source = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_source
        mock_dependencies["xml_handler"].element_to_string.return_value = "<amendment>xml</amendment>"

        # Mock LLM response
        csv_response = "find_text,replace_text\n" "company,corporation"

        mock_llm_kernel.run_inference.return_value = csv_response

        # Mock event
        with patch("app.services.keeling_service.event") as mock_event:
            # Call method
            result = service._extract_single_pattern(amendment, "schedule-123")

        # Verify result
        assert result == {"find_text": "company", "replace_text": "corporation"}

        # Verify LLM was called
        mock_llm_kernel.run_inference.assert_called_once_with(
            "ExtractEachPlacePattern", "schedule-123", "each1", None, amendment_xml="<amendment>xml</amendment>"
        )

        # Verify event was logged
        start_events = [call for call in mock_event.call_args_list if "PATTERN_EXTRACTION_START" in str(call)]
        assert len(start_events) == 1

    def test_extract_single_pattern_source_not_found(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_single_pattern when source element not found."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.source_eid = "missing_source"

        # Mock source not found
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = None

        with patch("app.services.keeling_service.logger") as mock_logger:
            result = service._extract_single_pattern(amendment, "schedule-123")

        assert result is None
        mock_logger.error.assert_called_with("Source element missing_source not found")

    def test_extract_single_pattern_no_rows(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_single_pattern when LLM returns no data rows."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.source_eid = "source_1"

        # Mock source element
        mock_source = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_source
        mock_dependencies["xml_handler"].element_to_string.return_value = "<amendment>xml</amendment>"

        # Mock LLM response with only headers
        csv_response = """find_text,replace_text"""

        mock_llm_kernel.run_inference.return_value = csv_response

        with patch("app.services.keeling_service.logger") as mock_logger:
            result = service._extract_single_pattern(amendment, "schedule-123")

        assert result is None
        mock_logger.error.assert_called_with("No patterns extracted for amendment each1")

    def test_extract_single_pattern_missing_fields(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_single_pattern when required fields are missing."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.source_eid = "source_1"

        # Mock source element
        mock_source = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_source
        mock_dependencies["xml_handler"].element_to_string.return_value = "<amendment>xml</amendment>"

        # Mock LLM response missing replace_text
        csv_response = """find_text,other_field
    company,something"""

        mock_llm_kernel.run_inference.return_value = csv_response

        with patch("app.services.keeling_service.logger") as mock_logger:
            result = service._extract_single_pattern(amendment, "schedule-123")

        assert result is None
        mock_logger.error.assert_called_with("Missing required fields in pattern for amendment each1")

    def test_extract_single_pattern_empty_string_deletion(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_single_pattern handling empty string for deletions."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.source_eid = "source_1"
        amendment.amendment_type = AmendmentType.DELETION

        # Mock source element
        mock_source = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_source
        mock_dependencies["xml_handler"].element_to_string.return_value = "<amendment>xml</amendment>"

        # Mock LLM response
        csv_response = (
            "find_text,replace_text\n"
            "company,something"  # LLM might incorrectly provide replacement text
        )

        mock_llm_kernel.run_inference.return_value = csv_response

        # Mock logger to verify debug message
        with patch("app.services.keeling_service.logger") as mock_logger:
            # Call method
            result = service._extract_single_pattern(amendment, "schedule-123")

        # Verify empty string is forced for deletions
        assert result == {"find_text": "company", "replace_text": ""}

        # Verify debug log was called
        mock_logger.debug.assert_called_with("Deletion amendment each1: forcing replace_text to empty string")

    def test_extract_single_pattern_exception(self, mock_llm_kernel, mock_dependencies):
        """Test _extract_single_pattern when exception occurs."""
        service = KeelingService(mock_llm_kernel)
        service._amending_bill = Mock()

        amendment = Mock()
        amendment.amendment_id = "each1"
        amendment.source_eid = "source_1"

        # Mock source element
        mock_source = Mock()
        mock_dependencies["xml_handler"].find_element_by_eid.return_value = mock_source
        mock_dependencies["xml_handler"].element_to_string.return_value = "<amendment>xml</amendment>"

        # Mock LLM to throw exception
        mock_llm_kernel.run_inference.side_effect = RuntimeError("LLM error")

        with patch("app.services.keeling_service.logger") as mock_logger:
            result = service._extract_single_pattern(amendment, "schedule-123")

        assert result is None
        mock_logger.error.assert_called_with("Failed to extract pattern for amendment each1: LLM error")

    import os

    @patch("os.path.getsize")
    def test_apply_amendments_each_place_mixed_patterns(mock_getsize, self, mock_llm_kernel, mock_dependencies):
        """Test apply_amendments with mix of validated and non-validated EACH_PLACE amendments."""
        mock_getsize.return_value = 1000
        service = KeelingService(mock_llm_kernel)

        # Create amendments
        each_place_validated = Mock(spec=Amendment)
        each_place_validated.whole_provision = False
        each_place_validated.amendment_id = "each1"
        each_place_validated.location = AmendmentLocation.EACH_PLACE
        each_place_validated.affected_provision = "sec_1"
        each_place_validated.amendment_type = Mock(value="SUBSTITUTION")
        each_place_validated.source_eid = "source_1"
        each_place_validated.source = "s. 1"

        each_place_not_validated = Mock(spec=Amendment)
        each_place_not_validated.whole_provision = False
        each_place_not_validated.amendment_id = "each2"
        each_place_not_validated.location = AmendmentLocation.EACH_PLACE
        each_place_not_validated.affected_provision = "sec_2"
        each_place_not_validated.amendment_type = Mock(value="DELETION")
        each_place_not_validated.source_eid = "source_2"
        each_place_not_validated.source = "s. 2"

        regular_amendment = Mock(spec=Amendment)
        regular_amendment.whole_provision = False
        regular_amendment.amendment_id = "reg1"
        regular_amendment.location = AmendmentLocation.REPLACE
        regular_amendment.affected_provision = "sec_3"
        regular_amendment.amendment_type = Mock(value="INSERTION")
        regular_amendment.source_eid = "source_3"
        regular_amendment.source = "s. 3"

        amendments = [each_place_validated, each_place_not_validated, regular_amendment]

        # Setup mocks
        mock_tree = Mock(spec=etree.ElementTree)
        mock_dependencies["xml_handler"].load_xml.return_value = mock_tree
        mock_dependencies["xml_handler"].find_existing_dnums.return_value = 0

        # Mock tracker
        mock_dependencies["tracker"].ensure_all_amendments_resolved.return_value = {
            "all_resolved": True,
            "stats": {"total": 3, "applied": 3, "with_error_comments": 0, "unresolved": 0},
            "unresolved_amendments": [],
        }
        mock_dependencies["tracker"].get_amendments_by_status.return_value = []

        # Mock pattern extraction/validation
        with patch.object(service, "_extract_and_validate_patterns") as mock_extract_validate:
            # Only each1 is validated
            mock_extract_validate.return_value = (
                {"each1": {"find_text": "company", "replace_text": "corporation"}},
                ["each2"],
            )

            # Mock LLM responses for non-validated amendments
            with patch.object(service, "_fetch_llm_responses_parallel") as mock_fetch:
                mock_fetch.return_value = {
                    "each2": "<deleted>content</deleted>",
                    "reg1": "<inserted>content</inserted>",
                }

                # Mock processor
                mock_dependencies["processor"].apply_amendment.return_value = (True, None)
                mock_dependencies["processor"].apply_each_place_amendment.return_value = (True, None)

                service.apply_amendments("/path/to/act.xml", amendments, "/path/to/output.xml", "schedule-123")

        # Verify fetch was called with correct amendments (non-validated each_place and regular)
        mock_fetch.assert_called_once()
        fetch_args = mock_fetch.call_args[0][0]
        assert len(fetch_args) == 2
        assert each_place_not_validated in fetch_args
        assert regular_amendment in fetch_args
        assert each_place_validated not in fetch_args  # Should not be in LLM fetch
