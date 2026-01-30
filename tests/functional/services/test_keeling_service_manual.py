import json
import uuid
import pytest
from app.services.keeling_service import KeelingService
from app.kernel.llm_kernel import get_kernel


def run_process_amending_bill(keeling_service, act_name):
    """Test that process_amending_bill correctly calls the LLM and parses output."""
    mock_tracking_id = str(uuid.uuid4())
    bill_file = "tests/data/" + act_name + "/bill.xml"
    amendments = keeling_service.process_amending_bill(bill_file, act_name, mock_tracking_id)

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
        for amendment in amendments
    ]
    json_table_of_amendments = json.dumps(table_of_amendments, indent=4)

    with open("tests/data/" + act_name + "/expected_table_of_amendments.json", "r") as f:
        expected_output = json.dumps(json.load(f), indent=4)

    assert json_table_of_amendments == expected_output


@pytest.fixture
def keeling_service():
    """Fixture to provide a KeelingService instance with mocked dependencies."""
    llm_kernel = get_kernel()
    keeling_service = KeelingService(llm_kernel)
    return keeling_service


# Comment the line below to run the tests locally
pytestmark = pytest.mark.skip("These tests should only be run locally because they access external resources.")


def test_process_amending_bill__Bankruptcy(keeling_service):
    act_name = "Bankruptcy (Scotland) Act 2016"
    run_process_amending_bill(keeling_service, act_name)


def test_process_amending_bill__Agriculture(keeling_service):
    act_name = "Agricultural Holdings (Scotland) Act 2003"
    run_process_amending_bill(keeling_service, act_name)


def test_process_amending_bill__Housing(keeling_service):
    act_name = "Housing Act 2004"
    run_process_amending_bill(keeling_service, act_name)


def test_process_amending_bill__Employment(keeling_service):
    act_name = "Employment Rights Act 1996"
    run_process_amending_bill(keeling_service, act_name)
