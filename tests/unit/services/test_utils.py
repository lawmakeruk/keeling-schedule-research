# tests/test_utils.py
"""
Unit tests for the utils module.
"""

import pytest
from unittest.mock import patch
from app.services.utils import (
    csv_to_amendment_dict,
    eid_to_source,
)


class TestCSVToAmendmentDict:
    """Tests for csv_to_amendment_dict function."""

    def test_valid_csv_single_row(self):
        """Test parsing valid CSV with single amendment."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
sec_1__subsec_2,s. 1(2),INSERTION,true,AFTER,sec_3__subsec_1"""

        result = csv_to_amendment_dict(csv_string)

        assert len(result) == 1
        assert result[0]["source_eid"] == "sec_1__subsec_2"
        assert result[0]["source"] == "s. 1(2)"
        assert result[0]["type_of_amendment"] == "INSERTION"
        assert result[0]["whole_provision"] is True
        assert result[0]["location"] == "AFTER"
        assert result[0]["affected_provision"] == "sec_3__subsec_1"

    def test_valid_csv_multiple_rows(self):
        """Test parsing valid CSV with multiple amendments."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
sec_1,s. 1,DELETION,true,REPLACE,sec_5
sec_2__para_a,s. 2(a),SUBSTITUTION,false,REPLACE,sec_6__para_b"""

        result = csv_to_amendment_dict(csv_string)

        assert len(result) == 2
        assert result[0]["type_of_amendment"] == "DELETION"
        assert result[1]["type_of_amendment"] == "SUBSTITUTION"

    def test_normalisation_of_values(self):
        """Test that values are properly normalised."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
SEC_1__SUBSEC_2,s. 1(2),insertion,TRUE,after,SEC_3__SUBSEC_1"""

        result = csv_to_amendment_dict(csv_string)

        assert result[0]["source_eid"] == "sec_1__subsec_2"  # lowercase
        assert result[0]["type_of_amendment"] == "INSERTION"  # uppercase
        assert result[0]["whole_provision"] is True  # boolean
        assert result[0]["location"] == "AFTER"  # uppercase
        assert result[0]["affected_provision"] == "sec_3__subsec_1"  # lowercase

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        csv_string = """source_eid, source , type_of_amendment,whole_provision,location,affected_provision
  sec_1  , s. 1 , DELETION  ,true,  REPLACE  ,  sec_5  """

        result = csv_to_amendment_dict(csv_string)

        assert result[0]["source_eid"] == "sec_1"
        assert result[0]["source"] == "s. 1"
        assert result[0]["location"] == "REPLACE"

    def test_header_variations(self):
        """Test that different header formats are accepted."""
        csv_string = """Source EID,Source,Type of Amendment,Whole Provision,Location,Affected Provision
sec_1,s. 1,DELETION,true,REPLACE,sec_5"""

        result = csv_to_amendment_dict(csv_string)

        assert len(result) == 1
        assert result[0]["source_eid"] == "sec_1"

    def test_skip_preamble_lines(self):
        """Test that preamble lines before header are skipped."""
        csv_string = """Some preamble text
Another line of preamble
source_eid,source,type_of_amendment,whole_provision,location,affected_provision
sec_1,s. 1,DELETION,true,REPLACE,sec_5"""

        result = csv_to_amendment_dict(csv_string)

        assert len(result) == 1
        assert result[0]["source_eid"] == "sec_1"

    def test_invalid_amendment_type(self):
        """Test that invalid amendment types are caught."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
sec_1,s. 1,INVALID_TYPE,true,REPLACE,sec_5"""

        # Invalid rows are skipped, and if no valid rows remain, ValueError is raised
        with pytest.raises(ValueError, match="No valid amendments found in CSV"):
            csv_to_amendment_dict(csv_string)

    def test_invalid_location(self):
        """Test that invalid locations are caught."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
sec_1,s. 1,DELETION,true,INVALID_LOC,sec_5"""

        # Invalid rows are skipped, and if no valid rows remain, ValueError is raised
        with pytest.raises(ValueError, match="No valid amendments found in CSV"):
            csv_to_amendment_dict(csv_string)

    def test_missing_required_column(self):
        """Test that missing required columns raise error."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,affected_provision
sec_1,s. 1,DELETION,true,sec_5"""

        with pytest.raises(ValueError, match="CSV missing required columns: {'location'}"):
            csv_to_amendment_dict(csv_string)

    def test_empty_required_field(self):
        """Test that empty required fields are caught."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
sec_1,,DELETION,true,REPLACE,sec_5"""

        # Empty required fields cause the row to be skipped
        with pytest.raises(ValueError, match="No valid amendments found in CSV"):
            csv_to_amendment_dict(csv_string)

    def test_empty_csv(self):
        """Test that empty CSV raises error."""
        csv_string = ""

        with pytest.raises(ValueError, match="CSV data has no headers"):
            csv_to_amendment_dict(csv_string)

    def test_headers_only(self):
        """Test that CSV with only headers raises error."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision"""

        with pytest.raises(ValueError, match="No valid amendments found in CSV"):
            csv_to_amendment_dict(csv_string)

    def test_whole_provision_variations(self):
        """Test different representations of boolean whole_provision."""
        csv_string = """source_eid,source,type_of_amendment,whole_provision,location,affected_provision
sec_1,s. 1,DELETION,true,REPLACE,sec_5
sec_2,s. 2,DELETION,True,REPLACE,sec_6
sec_3,s. 3,DELETION,TRUE,REPLACE,sec_7
sec_4,s. 4,DELETION,false,REPLACE,sec_8
sec_5,s. 5,DELETION,False,REPLACE,sec_9
sec_6,s. 6,DELETION,FALSE,REPLACE,sec_10"""

        result = csv_to_amendment_dict(csv_string)

        assert len(result) == 6
        assert all(r["whole_provision"] is True for r in result[:3])
        assert all(r["whole_provision"] is False for r in result[3:])


class TestSortAmendmentsByEidStructure:
    """Tests for sort_amendments_by_eid_structure function."""

    def test_sort_sections_numeric_order(self):
        """Test sorting sections in numeric order."""
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        # Create amendments in mixed order
        amendments = [
            Amendment(
                source_eid="sec_25__subsec_2",
                source="s. 25(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_10",
            ),
            Amendment(
                source_eid="sec_15__subsec_1",
                source="s. 15(1)",
                amendment_type=AmendmentType.DELETION,
                whole_provision=True,
                location=AmendmentLocation.REPLACE,
                affected_document="Test Act",
                affected_provision="sec_2",
            ),
            Amendment(
                source_eid="sec_20__subsec_3",
                source="s. 20(3)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_5",
            ),
        ]

        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision(amendments)

        # Check order
        assert sorted_amendments[0].affected_provision == "sec_2"
        assert sorted_amendments[1].affected_provision == "sec_5"
        assert sorted_amendments[2].affected_provision == "sec_10"

    def test_sort_sections_with_letters(self):
        """Test sorting sections with letter suffixes (e.g., 59a, 59b)."""
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        amendments = [
            Amendment(
                source_eid="sec_30__subsec_1",
                source="s. 30(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_59b",
            ),
            Amendment(
                source_eid="sec_30__subsec_2",
                source="s. 30(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_59",
            ),
            Amendment(
                source_eid="sec_30__subsec_3",
                source="s. 30(3)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_59a",
            ),
        ]

        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision(amendments)

        assert sorted_amendments[0].affected_provision == "sec_59"
        assert sorted_amendments[1].affected_provision == "sec_59a"
        assert sorted_amendments[2].affected_provision == "sec_59b"

    def test_sort_nested_provisions(self):
        """Test sorting nested provisions (subsections, paragraphs)."""
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        amendments = [
            Amendment(
                source_eid="sec_40__subsec_1",
                source="s. 40(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_5__para_b",
            ),
            Amendment(
                source_eid="sec_40__subsec_2",
                source="s. 40(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_5__subsec_1",
            ),
            Amendment(
                source_eid="sec_40__subsec_3",
                source="s. 40(3)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_5__para_a",
            ),
            Amendment(
                source_eid="sec_40__subsec_4",
                source="s. 40(4)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_5",
            ),
        ]

        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision(amendments)

        assert sorted_amendments[0].affected_provision == "sec_5"
        assert sorted_amendments[1].affected_provision == "sec_5__para_a"
        assert sorted_amendments[2].affected_provision == "sec_5__para_b"
        assert sorted_amendments[3].affected_provision == "sec_5__subsec_1"

    def test_sort_sections_before_schedules(self):
        """Test that sections always come before schedules."""
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        amendments = [
            Amendment(
                source_eid="sec_50__subsec_1",
                source="s. 50(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sched_1",
            ),
            Amendment(
                source_eid="sec_50__subsec_2",
                source="s. 50(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_100",
            ),
            Amendment(
                source_eid="sec_50__subsec_3",
                source="s. 50(3)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sched_2",
            ),
            Amendment(
                source_eid="sec_50__subsec_4",
                source="s. 50(4)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",
            ),
        ]

        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision(amendments)

        # All sections should come before schedules
        assert sorted_amendments[0].affected_provision == "sec_1"
        assert sorted_amendments[1].affected_provision == "sec_100"
        assert sorted_amendments[2].affected_provision == "sched_1"
        assert sorted_amendments[3].affected_provision == "sched_2"

    def test_sort_complex_hierarchy(self):
        """Test sorting with complex document hierarchy."""
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        amendments = [
            Amendment(
                source_eid="sec_60__subsec_1",
                source="s. 60(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="part_2__chapter_3__sec_10",
            ),
            Amendment(
                source_eid="sec_60__subsec_2",
                source="s. 60(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="part_1__sec_5",
            ),
            Amendment(
                source_eid="sec_60__subsec_3",
                source="s. 60(3)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="part_2__chapter_1__sec_8",
            ),
            Amendment(
                source_eid="sec_60__subsec_4",
                source="s. 60(4)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="preamble__para_1",
            ),
        ]

        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision(amendments)

        # Preamble should come first, then parts in order
        assert sorted_amendments[0].affected_provision == "preamble__para_1"
        assert sorted_amendments[1].affected_provision == "part_1__sec_5"
        assert sorted_amendments[2].affected_provision == "part_2__chapter_1__sec_8"
        assert sorted_amendments[3].affected_provision == "part_2__chapter_3__sec_10"

    def test_sort_with_unknown_types(self):
        """Test sorting handles unknown element types gracefully."""
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        amendments = [
            Amendment(
                source_eid="sec_70__subsec_1",
                source="s. 70(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="regulation_5",  # Not in TYPE_PRIORITY
            ),
            Amendment(
                source_eid="sec_70__subsec_2",
                source="s. 70(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",
            ),
            Amendment(
                source_eid="sec_70__subsec_3",
                source="s. 70(3)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="clause_5",  # Not in TYPE_PRIORITY
            ),
        ]

        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision(amendments)

        # Known types should come first
        assert sorted_amendments[0].affected_provision == "sec_1"
        # Unknown types sorted alphabetically at the end
        assert sorted_amendments[1].affected_provision == "clause_5"
        assert sorted_amendments[2].affected_provision == "regulation_5"

    def test_sort_empty_list(self):
        """Test sorting empty list of amendments."""
        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision([])
        assert sorted_amendments == []

    def test_sort_with_none_affected_provision(self):
        """Test sorting handles None affected_provision gracefully."""
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        amendments = [
            Amendment(
                source_eid="sec_80__subsec_1",
                source="s. 80(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",
            ),
            Amendment(
                source_eid="sec_80__subsec_2",
                source="s. 80(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision=None,  # This should be handled gracefully
            ),
        ]

        from app.services.utils import sort_amendments_by_affected_provision

        sorted_amendments = sort_amendments_by_affected_provision(amendments)

        # sec_1 has priority 3, None (converted to "") has priority 99, so sec_1 comes first
        assert sorted_amendments[0].affected_provision == "sec_1"
        assert sorted_amendments[1].affected_provision is None

    def test_eid_sort_key_edge_cases(self):
        """Test _eid_sort_key with edge cases to cover line 176."""
        from app.services.utils import _eid_sort_key, NON_NUMERIC_SORT_VALUE

        # Test with non-numeric value
        key = _eid_sort_key("article_iv")  # Roman numeral, non-numeric
        assert key == [(3, "article", NON_NUMERIC_SORT_VALUE, "iv")]

        # Test with malformed eid (no underscore)
        key = _eid_sort_key("malformed")
        assert key == [(99, "malformed", NON_NUMERIC_SORT_VALUE, "malformed")]

        # Test empty string - returns list with one tuple containing empty string
        key = _eid_sort_key("")
        assert key == [(99, "", NON_NUMERIC_SORT_VALUE, "")]

        # Test with subsection that has only letters
        key = _eid_sort_key("sec_5__para_aa")  # Only letters, no numbers
        assert len(key) == 2
        assert key[0] == (3, "sec", 5, "")
        assert key[1] == (99, "para", NON_NUMERIC_SORT_VALUE, "aa")

    def test_group_amendments_by_target(self):
        from app.services.utils import group_amendments_by_target
        from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

        amendments = [
            Amendment(
                source_eid="sec_80__subsec_1",
                source="s. 80(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",
            ),
            Amendment(
                source_eid="sec_80__subsec_1",
                source="s. 80(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1__subsec_1",
            ),
            Amendment(
                source_eid="sec_80__subsec_2",
                source="s. 80(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_2",
            ),
        ]
        groups = group_amendments_by_target(amendments)

        assert len(groups) == 3
        assert "sec_1" in groups
        assert "sec_1__subsec_1" in groups
        assert "sec_2" in groups


class TestAmendmentExpansion:
    def test_expand_amendment_if_needed_semicolon_separated(self):
        """Test expansion of semicolon-separated amendments."""
        from app.services.utils import _expand_amendment_if_needed
        from unittest.mock import patch

        amendment = {
            "source_eid": "sec_10__subsec_1",
            "source": "s. 10(1)",
            "type_of_amendment": "INSERTION",
            "whole_provision": False,
            "location": "AFTER",
            "affected_provision": "sec_1__para_a;sec_1__para_b;sec_1__para_c",
        }

        with patch("app.services.utils.logger") as mock_logger:
            result = _expand_amendment_if_needed(amendment, None, None)

            # Should expand to 3 amendments
            assert len(result) == 3
            assert result[0]["affected_provision"] == "sec_1__para_a"
            assert result[1]["affected_provision"] == "sec_1__para_b"
            assert result[2]["affected_provision"] == "sec_1__para_c"

            # Check all fields are copied
            for r in result:
                assert r["source_eid"] == "sec_10__subsec_1"
                assert r["source"] == "s. 10(1)"
                assert r["type_of_amendment"] == "INSERTION"
                assert r["whole_provision"] is False
                assert r["location"] == "AFTER"

            # With recursion: 1 initial + 1 expanded + 3 recursive checks + 3 no expansion
            assert mock_logger.debug.call_count == 8

            # Check the specific calls
            mock_logger.debug.assert_any_call(
                "Checking if expansion needed for: sec_1__para_a;sec_1__para_b;sec_1__para_c"
            )
            mock_logger.debug.assert_any_call(
                "Expanded semicolon-separated amendment into 3 amendments: sec_1__para_a;sec_1__para_b;sec_1__para_c"
            )

    def test_expand_amendment_if_needed_range_success(self):
        """Test successful range expansion."""
        from app.services.utils import _expand_amendment_if_needed
        from unittest.mock import Mock, patch
        from lxml import etree

        mock_xml_handler = Mock()
        mock_xml_handler.find_provisions_in_range = Mock(return_value=["sec_5", "sec_6", "sec_7", "sec_8"])
        mock_target_act = Mock(spec=etree.ElementTree)

        amendment = {
            "source_eid": "sec_25__subsec_2",
            "source": "s. 25(2)",
            "type_of_amendment": "DELETION",
            "whole_provision": True,
            "location": "REPLACE",
            "affected_provision": "sec_5-sec_8",
        }

        with patch("app.services.utils.logger") as mock_logger:
            result = _expand_amendment_if_needed(amendment, mock_target_act, mock_xml_handler)

            # Should return expanded amendments
            assert len(result) == 4
            assert result[0]["affected_provision"] == "sec_5"
            assert result[1]["affected_provision"] == "sec_6"
            assert result[2]["affected_provision"] == "sec_7"
            assert result[3]["affected_provision"] == "sec_8"

            # Check all fields are preserved
            for r in result:
                assert r["source_eid"] == "sec_25__subsec_2"
                assert r["source"] == "s. 25(2)"
                assert r["type_of_amendment"] == "DELETION"
                assert r["whole_provision"] is True
                assert r["location"] == "REPLACE"

            # Verify xml_handler was called correctly
            mock_xml_handler.find_provisions_in_range.assert_called_once_with("sec_5", "sec_8", mock_target_act)

            # Verify debug logging
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("Checking if expansion needed for: sec_5-sec_8" in call for call in debug_calls)
            assert any("Attempting range expansion for: sec_5-sec_8" in call for call in debug_calls)

    def test_expand_amendment_if_needed_recursive_semicolon_and_range(self):
        """Test recursive expansion of combined semicolon and range notation."""
        from app.services.utils import _expand_amendment_if_needed
        from unittest.mock import Mock, patch
        from lxml import etree

        mock_xml_handler = Mock()
        mock_xml_handler.find_provisions_in_range = Mock(return_value=["sec_85", "sec_86", "sec_87"])
        mock_target_act = Mock(spec=etree.ElementTree)

        amendment = {
            "source_eid": "sec_15__para_a",
            "source": "s. 15(a)",
            "type_of_amendment": "DELETION",
            "whole_provision": True,
            "location": "REPLACE",
            "affected_provision": "sec_85-sec_87;pt_II__chp_1__xhdg_85__hdg",
        }

        with patch("app.services.utils.logger"):
            result = _expand_amendment_if_needed(amendment, mock_target_act, mock_xml_handler)

            # Should expand to 4 amendments (3 from range + 1 heading)
            assert len(result) == 4

            # Check the expanded range amendments
            assert result[0]["affected_provision"] == "sec_85"
            assert result[1]["affected_provision"] == "sec_86"
            assert result[2]["affected_provision"] == "sec_87"

            # Check the heading amendment
            assert result[3]["affected_provision"] == "pt_II__chp_1__xhdg_85__hdg"

            # All should have the same source information
            for r in result:
                assert r["source_eid"] == "sec_15__para_a"
                assert r["source"] == "s. 15(a)"
                assert r["type_of_amendment"] == "DELETION"
                assert r["whole_provision"] is True
                assert r["location"] == "REPLACE"

            # Verify the XML handler was called to find provisions in range
            mock_xml_handler.find_provisions_in_range.assert_called_once_with("sec_85", "sec_87", mock_target_act)

    def test_expand_amendment_if_needed_range_exception(self):
        """Test range expansion with exception handling."""
        from app.services.utils import _expand_amendment_if_needed
        from unittest.mock import Mock, patch
        from lxml import etree

        mock_xml_handler = Mock()
        mock_target_act = Mock(spec=etree.ElementTree)

        amendment = {
            "source_eid": "sec_25__subsec_2",
            "source": "s. 25(2)",
            "type_of_amendment": "DELETION",
            "whole_provision": True,
            "location": "REPLACE",
            "affected_provision": "sec_5-sec_8",
        }

        with patch("app.services.utils.logger") as mock_logger:
            with patch("app.services.utils._expand_range_amendment") as mock_expand_range:
                # Mock exception
                mock_expand_range.side_effect = ValueError("Invalid eId format")

                result = _expand_amendment_if_needed(amendment, mock_target_act, mock_xml_handler)

                # Should return original amendment unchanged
                assert len(result) == 1
                assert result[0] == amendment

                # Verify warning was logged
                mock_logger.warning.assert_called_once_with(
                    "Failed to expand range sec_5-sec_8: Invalid eId format. Keeping as-is."
                )

    def test_expand_range_amendment_invalid_format(self):
        """Test _expand_range_amendment with invalid range format."""
        from app.services.utils import _expand_range_amendment
        from unittest.mock import Mock, patch
        from lxml import etree

        amendment = {"source_eid": "sec_1", "type_of_amendment": "DELETION"}
        mock_target_act = Mock(spec=etree.ElementTree)
        mock_xml_handler = Mock()

        # Test with no dash (single provision, not a range)
        with patch("app.services.utils.logger") as mock_logger:
            result = _expand_range_amendment(amendment, "sec_5", mock_target_act, mock_xml_handler)
            assert result == []
            mock_logger.warning.assert_called_with("Invalid range format: sec_5")

        # Test with empty string
        with patch("app.services.utils.logger") as mock_logger:
            result = _expand_range_amendment(amendment, "", mock_target_act, mock_xml_handler)
            assert result == []
            mock_logger.warning.assert_called_with("Invalid range format: ")

        # Test with just a dash - this passes the format check but returns empty
        mock_xml_handler.find_provisions_in_range = Mock(return_value=[])
        with patch("app.services.utils.logger") as mock_logger:
            result = _expand_range_amendment(amendment, "-", mock_target_act, mock_xml_handler)
            assert result == []
            mock_logger.warning.assert_called_with("No provisions found in range -")
            # Verify it was called with two empty strings
            mock_xml_handler.find_provisions_in_range.assert_called_with("", "", mock_target_act)

    def test_expand_range_amendment_no_provisions(self):
        """Test _expand_range_amendment when no provisions found."""
        from app.services.utils import _expand_range_amendment
        from unittest.mock import Mock, patch
        from lxml import etree

        mock_xml_handler = Mock()
        mock_xml_handler.find_provisions_in_range = Mock(return_value=[])
        mock_target_act = Mock(spec=etree.ElementTree)

        amendment = {"source_eid": "sec_25__subsec_2", "source": "s. 25(2)", "type_of_amendment": "DELETION"}

        with patch("app.services.utils.logger") as mock_logger:
            result = _expand_range_amendment(amendment, "sec_99-sec_105", mock_target_act, mock_xml_handler)

            assert result == []
            mock_logger.warning.assert_called_with("No provisions found in range sec_99-sec_105")

            # Verify xml_handler was called correctly
            mock_xml_handler.find_provisions_in_range.assert_called_once_with("sec_99", "sec_105", mock_target_act)

    def test_expand_range_amendment_success(self):
        """Test successful _expand_range_amendment."""
        from app.services.utils import _expand_range_amendment
        from unittest.mock import Mock, patch
        from lxml import etree

        mock_xml_handler = Mock()
        mock_xml_handler.find_provisions_in_range = Mock(return_value=["sec_5", "sec_6", "sec_7", "sec_8"])
        mock_target_act = Mock(spec=etree.ElementTree)

        amendment = {
            "source_eid": "sec_25__subsec_2",
            "source": "s. 25(2)",
            "type_of_amendment": "DELETION",
            "whole_provision": True,
            "location": "REPLACE",
            "extra_field": "test_value",
        }

        with patch("app.services.utils.logger") as mock_logger:
            result = _expand_range_amendment(amendment, "sec_5-sec_8", mock_target_act, mock_xml_handler)

            # Should create 4 amendments
            assert len(result) == 4

            # Check each expanded amendment
            for i, provision_eid in enumerate(["sec_5", "sec_6", "sec_7", "sec_8"]):
                assert result[i]["affected_provision"] == provision_eid
                assert result[i]["source_eid"] == "sec_25__subsec_2"
                assert result[i]["source"] == "s. 25(2)"
                assert result[i]["type_of_amendment"] == "DELETION"
                assert result[i]["whole_provision"] is True
                assert result[i]["location"] == "REPLACE"
                assert result[i]["extra_field"] == "test_value"

            # Verify info logging
            mock_logger.info.assert_called_once_with(
                "Expanded range sec_5-sec_8 into 4 provisions: ['sec_5', 'sec_6', 'sec_7', 'sec_8']"
            )

    def test_expand_range_amendment_whitespace_handling(self):
        """Test _expand_range_amendment strips whitespace from eIds."""
        from app.services.utils import _expand_range_amendment
        from unittest.mock import Mock, patch
        from lxml import etree

        mock_xml_handler = Mock()
        mock_xml_handler.find_provisions_in_range = Mock(return_value=["sec_5", "sec_6"])
        mock_target_act = Mock(spec=etree.ElementTree)

        amendment = {"source_eid": "sec_1", "type_of_amendment": "INSERTION"}

        with patch("app.services.utils.logger"):
            result = _expand_range_amendment(amendment, " sec_5 - sec_6 ", mock_target_act, mock_xml_handler)

            assert len(result) == 2

            # Verify handler was called with stripped values
            mock_xml_handler.find_provisions_in_range.assert_called_once_with("sec_5", "sec_6", mock_target_act)

    def test_expand_amendment_if_needed_no_expansion(self):
        """Test when no expansion is needed (single provision)."""
        from app.services.utils import _expand_amendment_if_needed

        amendment = {
            "source_eid": "sec_10__subsec_1",
            "source": "s. 10(1)",
            "type_of_amendment": "INSERTION",
            "whole_provision": True,
            "location": "BEFORE",
            "affected_provision": "sec_5__subsec_2",
        }

        # No semicolon, no dash
        result = _expand_amendment_if_needed(amendment, None, None)

        assert len(result) == 1
        assert result[0] is amendment  # Should return exact same object

    def test_expand_amendment_if_needed_dash_no_handler(self):
        """Test amendment with dash but no handler/act provided."""
        from app.services.utils import _expand_amendment_if_needed

        amendment = {
            "source_eid": "sec_10__subsec_1",
            "source": "s. 10(1)",
            "type_of_amendment": "DELETION",
            "whole_provision": True,
            "location": "REPLACE",
            "affected_provision": "sec_5-sec_8",
        }

        # Has dash but no handler/act
        result = _expand_amendment_if_needed(amendment, None, None)

        assert len(result) == 1
        assert result[0] is amendment  # Should return unchanged

    def test_expand_amendment_if_needed_empty_range_return(self):
        """Test when _expand_range_amendment returns empty list."""
        from app.services.utils import _expand_amendment_if_needed
        from unittest.mock import Mock, patch
        from lxml import etree

        mock_xml_handler = Mock()
        mock_target_act = Mock(spec=etree.ElementTree)

        amendment = {
            "source_eid": "sec_25__subsec_2",
            "source": "s. 25(2)",
            "type_of_amendment": "DELETION",
            "whole_provision": True,
            "location": "REPLACE",
            "affected_provision": "sec_5-sec_8",
        }

        with patch("app.services.utils._expand_range_amendment") as mock_expand_range:
            # Mock empty return (no provisions found)
            mock_expand_range.return_value = []

            result = _expand_amendment_if_needed(amendment, mock_target_act, mock_xml_handler)

            # Should fall through and return original
            assert len(result) == 1
            assert result[0] is amendment


class TestEidToSource:
    """Tests for eid_to_source function and related formatting functions."""

    def test_eid_to_source_empty_input(self):
        """Test handling of empty or None input."""
        assert eid_to_source("") == ""
        assert eid_to_source(None) is None

    def test_eid_to_source_simple_section(self):
        """Test formatting simple section references."""
        assert eid_to_source("sec_40") == "s.40"
        assert eid_to_source("sec_1") == "s.1"
        assert eid_to_source("sec_59A") == "s.59A"

    def test_eid_to_source_section_with_subsection(self):
        """Test formatting sections with subsections."""
        assert eid_to_source("sec_40__subsec_2") == "s.40(2)"
        assert eid_to_source("sec_10__subsec_1A") == "s.10(1A)"

    def test_eid_to_source_complex_section(self):
        """Test formatting complex section hierarchies."""
        assert eid_to_source("sec_40__subsec_2__para_b") == "s.40(2)(b)"
        assert eid_to_source("sec_23__subsec_3__para_a__subpara_i") == "s.23(3)(a)(i)"
        assert eid_to_source("sec_23__subsec_3__para_a__subpara_i__subsubpara_A") == "s.23(3)(a)(i)(A)"

    def test_eid_to_source_regulation(self):
        """Test formatting regulation references."""
        assert eid_to_source("reg_5") == "reg.5"
        assert eid_to_source("reg_1__para_2") == "reg.1(2)"
        assert eid_to_source("reg_10__subsec_3__para_a") == "reg.10(3)(a)"

    def test_eid_to_source_article(self):
        """Test formatting article references."""
        assert eid_to_source("art_12") == "art.12"
        assert eid_to_source("art_1__para_2") == "art.1(2)"

    def test_eid_to_source_rule(self):
        """Test formatting rule references."""
        assert eid_to_source("rule_3") == "rule 3"
        assert eid_to_source("rule_1__para_2") == "rule 1(2)"

    def test_eid_to_source_schedule_simple(self):
        """Test formatting simple schedule references."""
        assert eid_to_source("sched_1") == "Sch.1"
        assert eid_to_source("sched") == "Sch."

    def test_eid_to_source_schedule_with_paragraph(self):
        """Test formatting schedule with paragraph references."""
        assert eid_to_source("sched_3__para_5") == "Sch.3 para.5"
        assert eid_to_source("sched_1__para_2__subpara_3") == "Sch.1 para.2(3)"
        assert eid_to_source("sched_2__para_1__subpara_2__para_a") == "Sch.2 para.1(2)(a)"
        assert eid_to_source("sched_2__para_61__para_a") == "Sch.2 para.61(a)"

    def test_eid_to_source_schedule_with_part(self):
        """Test formatting schedule with part references."""
        assert eid_to_source("sched_1__pt_2__para_5") == "Sch.1 pt.2 para.5"

    def test_eid_to_source_definition(self):
        """Test formatting definition references."""
        assert eid_to_source("sec_93__def_tenant") == 'definition of "tenant" in s.93'
        assert (
            eid_to_source("sec_40__subsec_2__def_agricultural_land") == 'definition of "agricultural land" in s.40(2)'
        )
        assert (
            eid_to_source("sched_1__para_5__def_relevant_person") == 'definition of "relevant person" in Sch.1 para.5'
        )

    def test_eid_to_source_heading(self):
        """Test formatting heading references."""
        assert eid_to_source("sec_4__hdg") == "heading of s.4"
        assert eid_to_source("pt_2__hdg") == "heading of pt.2"
        assert eid_to_source("sched_1__hdg") == "heading of Sch.1"

    def test_eid_to_source_crossheading(self):
        """Test formatting crossheading references."""
        assert eid_to_source("pt_1__chp_2__xhdg_28__hdg") == "pt.1 ch.2 crossheading 28"
        assert eid_to_source("xhdg_15__hdg") == "crossheading 15"

    def test_eid_to_source_part_chapter(self):
        """Test formatting part and chapter references."""
        assert eid_to_source("pt_1") == "pt.1"
        assert eid_to_source("pt_2__chp_3") == "pt.2 ch.3"
        assert eid_to_source("pt_2__chp_3__sec_5") == "pt.2 ch.3 s.5"
        assert eid_to_source("chp_5") == "ch.5"

    def test_eid_to_source_standalone_paragraph(self):
        """Test formatting standalone paragraph references."""
        assert eid_to_source("para_5") == "para.5"
        assert eid_to_source("para_5__subpara_a") == "para.5(a)"

    def test_eid_to_source_generic_level(self):
        """Test formatting generic level elements."""
        assert eid_to_source("sec_40__level_1") == "s.40(1)"

    def test_eid_to_source_unknown_type(self):
        """Test handling of unknown provision types."""
        assert eid_to_source("unknown_type_5") == "unknown_type_5"
        assert eid_to_source("custom__subsec_2") == "custom__subsec_2"

    def test_eid_to_source_malformed_input(self):
        """Test handling of malformed input."""
        # No double underscore
        assert eid_to_source("sec40subsec2") == "sec40subsec2"
        # Just underscores
        assert eid_to_source("__") == "__"
        # Empty parts
        assert eid_to_source("sec_40__") == "sec_40__"

    @patch("app.services.utils.logger")
    def test_eid_to_source_exception_handling(self, mock_logger):
        """Test that exceptions are logged and original eId returned."""
        # Create a malformed parts list that will cause an exception
        with patch("app.services.utils._format_provision_parts", side_effect=Exception("Test error")):
            result = eid_to_source("sec_40__subsec_2")

            assert result == "sec_40__subsec_2"
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0]
            assert "Failed to format eId 'sec_40__subsec_2'" in args[0]
            assert "Test error" in args[0]

    def test_format_provision_parts_empty(self):
        """Test _format_provision_parts with empty input."""
        from app.services.utils import _format_provision_parts

        assert _format_provision_parts([]) == ""

    def test_format_provision_parts_no_match(self):
        """Test _format_provision_parts when regex doesn't match."""
        from app.services.utils import _format_provision_parts

        # Input that won't match the regex pattern
        assert _format_provision_parts(["123"]) == "123"

    def test_format_section_style_no_number(self):
        """Test _format_section_style with no main number."""
        from app.services.utils import _format_section_style

        assert _format_section_style(["sec"], "s.", "") == "s"
        assert _format_section_style(["reg"], "reg.", "") == "reg"

    def test_format_schedule_style_no_paragraph(self):
        """Test _format_schedule_style with no paragraph information."""
        from app.services.utils import _format_schedule_style

        assert _format_schedule_style(["sched_1"], "1") == "Sch.1"
        assert _format_schedule_style(["sched"], "") == "Sch."

    def test_format_paragraph_standalone_no_match(self):
        """Test _format_paragraph_standalone when regex doesn't match."""
        from app.services.utils import _format_paragraph_standalone

        # Input that won't match the para_ pattern
        assert _format_paragraph_standalone(["invalid"]) == "invalid"

    def test_eid_to_source_complex_examples(self):
        """Test complex real-world examples."""
        # From the human reviewed CSV examples
        assert eid_to_source("sec_28__subsec_2__para_a") == "s.28(2)(a)"
        assert eid_to_source("sec_28__subsec_7__para_b__subpara_ii") == "s.28(7)(b)(ii)"
        assert eid_to_source("sched_2__para_61__para_a") == "Sch.2 para.61(a)"
        assert eid_to_source("sched_4__para_3__subpara_5__para_b") == "Sch.4 para.3(5)(b)"

    def test_eid_to_source_part_without_number(self):
        """Test formatting part without number."""
        assert eid_to_source("pt") == "pt."

    def test_eid_to_source_chapter_without_number(self):
        """Test formatting chapter without number."""
        assert eid_to_source("chp") == "ch."


class TestFixProvisionNestingInEid:
    """Tests for fix_provision_nesting_in_eid function."""

    def test_fix_schedule_nesting_subpara_to_para(self):
        """Test fixing schedule nesting where subpara should be para for letters."""
        from app.services.utils import _fix_provision_nesting_in_eid

        # Basic schedule with letter that needs fixing
        assert _fix_provision_nesting_in_eid("sched_1__para_1__subpara_a") == "sched_1__para_1__para_a"
        assert _fix_provision_nesting_in_eid("sched_2__para_15__subpara_b") == "sched_2__para_15__para_b"
        assert _fix_provision_nesting_in_eid("sched_10__para_1A__subpara_c") == "sched_10__para_1A__para_c"

        # Schedule without number
        assert _fix_provision_nesting_in_eid("sched__para_5__subpara_d") == "sched__para_5__para_d"

    def test_fix_provision_nesting_para_to_subpara(self):
        """Test fixing provision nesting where para should be subpara for letters."""
        from app.services.utils import _fix_provision_nesting_in_eid

        # Sections
        assert _fix_provision_nesting_in_eid("sec_2__para_1__para_a") == "sec_2__para_1__subpara_a"
        assert _fix_provision_nesting_in_eid("sec_15A__para_3__para_b") == "sec_15A__para_3__subpara_b"

        # Regulations
        assert _fix_provision_nesting_in_eid("reg_5__para_2__para_c") == "reg_5__para_2__subpara_c"
        assert _fix_provision_nesting_in_eid("reg_100__para_1B__para_d") == "reg_100__para_1B__subpara_d"

        # Articles
        assert _fix_provision_nesting_in_eid("art_3__para_4__para_e") == "art_3__para_4__subpara_e"

        # Rules
        assert _fix_provision_nesting_in_eid("rule_7__para_1__para_f") == "rule_7__para_1__subpara_f"

    def test_no_changes_needed(self):
        """Test cases where no changes should be made."""
        from app.services.utils import _fix_provision_nesting_in_eid

        # Already correct schedule format
        assert _fix_provision_nesting_in_eid("sched_1__para_1__para_a") == "sched_1__para_1__para_a"

        # Already correct provision format
        assert _fix_provision_nesting_in_eid("sec_2__para_1__subpara_a") == "sec_2__para_1__subpara_a"

        # No letter parts (numbers instead)
        assert _fix_provision_nesting_in_eid("sched_1__para_1__subpara_1") == "sched_1__para_1__subpara_1"
        assert _fix_provision_nesting_in_eid("sec_2__para_1__para_2") == "sec_2__para_1__para_2"

        # Different structure entirely
        assert _fix_provision_nesting_in_eid("sec_5__subsec_2") == "sec_5__subsec_2"
        assert _fix_provision_nesting_in_eid("sched_1__pt_2__para_5") == "sched_1__pt_2__para_5"

        # Unknown provision types
        assert _fix_provision_nesting_in_eid("clause_1__para_2__para_a") == "clause_1__para_2__para_a"
        assert _fix_provision_nesting_in_eid("definition_1__para_2__subpara_a") == "definition_1__para_2__subpara_a"

    def test_complex_nested_structures(self):
        """Test fixing complex nested structures with multiple levels."""
        from app.services.utils import _fix_provision_nesting_in_eid

        # Schedule with multiple issues
        assert (
            _fix_provision_nesting_in_eid("sched_1__para_1__subpara_a__subsubpara_i")
            == "sched_1__para_1__para_a__subsubpara_i"
        )

        # Provision with deeper nesting after the fix point
        assert (
            _fix_provision_nesting_in_eid("sec_2__para_1__para_a__subsubpara_ii")
            == "sec_2__para_1__subpara_a__subsubpara_ii"
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        from app.services.utils import _fix_provision_nesting_in_eid

        # Empty string
        assert _fix_provision_nesting_in_eid("") == ""

        # Just the prefix
        assert _fix_provision_nesting_in_eid("sched") == "sched"
        assert _fix_provision_nesting_in_eid("sec_") == "sec_"

        # Capital letters (should not be fixed - only lowercase)
        assert _fix_provision_nesting_in_eid("sched_1__para_1__subpara_A") == "sched_1__para_1__subpara_A"
        assert _fix_provision_nesting_in_eid("sec_2__para_1__para_B") == "sec_2__para_1__para_B"

        # Multiple occurrences in one eid (only first should be fixed)
        assert (
            _fix_provision_nesting_in_eid("sched_1__para_1__subpara_a__para_2__subpara_b")
            == "sched_1__para_1__para_a__para_2__subpara_b"
        )

    @patch("app.services.utils.logger")
    def test_logging(self, mock_logger):
        """Test that fixes are logged at debug level."""
        from app.services.utils import _fix_provision_nesting_in_eid

        # Fix that should log
        result = _fix_provision_nesting_in_eid("sched_1__para_1__subpara_a")
        assert result == "sched_1__para_1__para_a"
        mock_logger.debug.assert_called_once_with(
            "Fixed schedule nesting: sched_1__para_1__subpara_a -> sched_1__para_1__para_a"
        )

        # Reset mock
        mock_logger.reset_mock()

        # Fix for provision
        result = _fix_provision_nesting_in_eid("sec_2__para_1__para_a")
        assert result == "sec_2__para_1__subpara_a"
        mock_logger.debug.assert_called_once_with(
            "Fixed provision nesting: sec_2__para_1__para_a -> sec_2__para_1__subpara_a"
        )

        # Reset mock
        mock_logger.reset_mock()

        # No fix needed - no logging
        result = _fix_provision_nesting_in_eid("sec_5__subsec_2")
        assert result == "sec_5__subsec_2"
        mock_logger.debug.assert_not_called()

    def test_pattern_boundary_matching(self):
        """Test that patterns correctly match at word boundaries."""
        from app.services.utils import _fix_provision_nesting_in_eid

        # Should fix - ends with letter
        assert _fix_provision_nesting_in_eid("sched_1__para_1__subpara_a") == "sched_1__para_1__para_a"

        # Should fix - has more parts after
        assert _fix_provision_nesting_in_eid("sched_1__para_1__subpara_a__item_1") == "sched_1__para_1__para_a__item_1"

        # Should not fix - not at boundary (part of larger string)
        assert _fix_provision_nesting_in_eid("sched_1__para_1__subpara_alpha") == "sched_1__para_1__subpara_alpha"
