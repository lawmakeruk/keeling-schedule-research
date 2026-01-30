import pytest
from app.models.amendments import Amendment, AmendmentLocation, AmendmentType


def test_amendment_location_enum():
    """Test that AmendmentLocation enum has expected values."""
    assert AmendmentLocation.BEFORE.value == "Before"
    assert AmendmentLocation.AFTER.value == "After"
    assert AmendmentLocation.REPLACE.value == "Replace"
    assert AmendmentLocation.EACH_PLACE.value == "Each_Place"
    assert len(AmendmentLocation) == 4


def test_amendment_type_enum():
    """Test that AmendmentType enum has expected values."""
    assert AmendmentType.INSERTION.value == "insertion"
    assert AmendmentType.DELETION.value == "deletion"
    assert AmendmentType.SUBSTITUTION.value == "substitution"
    assert len(AmendmentType) == 3


def test_amendment_creation():
    """Test creating an Amendment with valid data."""
    amendment = Amendment(
        source="Test Amendment Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    assert amendment.source == "Test Amendment Source"
    assert amendment.source_eid == "sec_1"
    assert amendment.affected_document == "Test Act"
    assert amendment.affected_provision == "sec_2"
    assert amendment.location == AmendmentLocation.AFTER
    assert amendment.amendment_type == AmendmentType.INSERTION
    assert amendment.whole_provision is True


def test_amendment_from_dict_valid():
    """Test creating an Amendment from a dictionary with valid data."""
    data = {
        "source": "Test Source",
        "source_eid": "sec_1",
        "affected_document": "Test Act",
        "affected_provision": "sec_2",
        "location": "AFTER",
        "type_of_amendment": "INSERTION",
        "whole_provision": True,
    }

    amendment = Amendment.from_dict(data)

    assert amendment.source == "Test Source"
    assert amendment.source_eid == "sec_1"
    assert amendment.affected_document == "Test Act"
    assert amendment.affected_provision == "sec_2"
    assert amendment.location == AmendmentLocation.AFTER
    assert amendment.amendment_type == AmendmentType.INSERTION
    assert amendment.whole_provision is True


def test_amendment_from_dict_case_insensitive():
    """Test that from_dict handles different case formats in the input."""
    data = {
        "source": "Test Source",
        "source_eid": "sec_1",
        "affected_document": "Test Act",
        "affected_provision": "sec_2",
        "location": "after",  # lowercase
        "type_of_amendment": "InSeRtIoN",  # mixed case
        "whole_provision": True,
    }

    amendment = Amendment.from_dict(data)
    assert amendment.location == AmendmentLocation.AFTER
    assert amendment.amendment_type == AmendmentType.INSERTION


@pytest.mark.parametrize(
    "invalid_data,expected_error",
    [
        (
            {
                "source": "Test",
                "source_eid": "sec_1",
                "affected_document": "Test Act",
                "affected_provision": "sec_2",
                "location": "INVALID",  # Invalid location
                "type_of_amendment": "INSERTION",
                "whole_provision": True,
            },
            AttributeError,
        ),
        (
            {
                "source": "Test",
                "source_eid": "sec_1",
                "affected_document": "Test Act",
                "affected_provision": "sec_2",
                "location": "AFTER",
                "type_of_amendment": "INVALID",  # Invalid amendment type
                "whole_provision": True,
            },
            AttributeError,
        ),
        (
            {
                # Missing required field
                "source_eid": "sec_1",
                "affected_document": "Test Act",
                "affected_provision": "sec_2",
                "location": "AFTER",
                "type_of_amendment": "INSERTION",
                "whole_provision": True,
            },
            KeyError,
        ),
    ],
)
def test_amendment_from_dict_invalid(invalid_data, expected_error):
    """Test that from_dict properly handles invalid input data."""
    with pytest.raises(expected_error):
        Amendment.from_dict(invalid_data)


def test_amendment_repr():
    """Test that Amendment has a useful string representation."""
    amendment = Amendment(
        source="Test Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    repr_str = repr(amendment)
    assert "Test Source" in repr_str
    assert "sec_1" in repr_str
    assert "sec_2" in repr_str
    assert "AFTER" in repr_str or "After" in repr_str
    assert "INSERTION" in repr_str or "insertion" in repr_str


def test_amendment_equality():
    """Test that Amendment equality comparison works correctly."""
    amendment1 = Amendment(
        source="Test Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    amendment2 = Amendment(
        source="Test Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    amendment3 = Amendment(
        source="Different Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    assert amendment1 == amendment2
    assert amendment1 != amendment3
    assert amendment1 != "not an amendment"


def test_amendment_to_dict():
    """Test the to_dict method converts Amendment to dictionary correctly."""
    amendment = Amendment(
        source="Test Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.BEFORE,
        amendment_type=AmendmentType.DELETION,
        whole_provision=False,
    )

    result = amendment.to_dict()

    # Check structure and enum conversions
    assert isinstance(result, dict)
    assert result["source"] == "Test Source"
    assert result["source_eid"] == "sec_1"
    assert result["affected_provision"] == "sec_2"
    assert result["affected_document"] == "Test Act"
    assert result["location"] == "Before"  # Enum converted to string value
    assert result["type_of_amendment"] == "deletion"  # Enum converted to string value
    assert result["whole_provision"] is False
    assert "amendment_id" in result  # Key exists even if None


def test_amendment_str():
    """Test the __str__ method for human-readable representation."""
    # Test whole provision amendment
    amendment_whole = Amendment(
        source="Section 10(2)",
        source_eid="sec_10_2",
        affected_document="Test Act",
        affected_provision="sec_5",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    str_whole = str(amendment_whole)
    assert str_whole == "Insertion amendment: whole provision after Test Act sec_5 (from Section 10(2))"

    # Test partial amendment
    amendment_partial = Amendment(
        source="Section 15(3)",
        source_eid="sec_15_3",
        affected_document="Test Act",
        affected_provision="sec_8",
        location=AmendmentLocation.REPLACE,
        amendment_type=AmendmentType.SUBSTITUTION,
        whole_provision=False,
    )

    str_partial = str(amendment_partial)
    assert str_partial == "Substitution amendment: partial replace Test Act sec_8 (from Section 15(3))"


def test_is_insertion():
    """Test amendment type methods with an insertion amendment."""
    amendment = Amendment(
        source="Test Amendment Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    assert amendment.is_insertion() is True
    assert amendment.is_deletion() is False
    assert amendment.is_substitution() is False


def test_is_deletion():
    """Test amendment type methods with a deletion amendment."""
    amendment = Amendment(
        source="Test Amendment Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.REPLACE,
        amendment_type=AmendmentType.DELETION,
        whole_provision=True,
    )

    assert amendment.is_insertion() is False
    assert amendment.is_deletion() is True
    assert amendment.is_substitution() is False


def test_is_substitution():
    """Test amendment type methods with a substitution amendment."""
    amendment = Amendment(
        source="Test Amendment Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.REPLACE,
        amendment_type=AmendmentType.SUBSTITUTION,
        whole_provision=True,
    )

    assert amendment.is_insertion() is False
    assert amendment.is_deletion() is False
    assert amendment.is_substitution() is True
