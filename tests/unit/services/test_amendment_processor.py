# tests/unit/services/test_amendment_processor.py
"""
Unit tests for AmendmentProcessor class that handles applying amendments to XML documents.
"""

import unittest
from unittest.mock import Mock, patch
from lxml import etree

from app.services.amendment_processor import AmendmentProcessor
from app.services.xml_handler import XMLHandler
from app.models.amendments import Amendment, AmendmentType, AmendmentLocation


class TestAmendmentProcessor(unittest.TestCase):
    """Test cases for AmendmentProcessor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.xml_handler = XMLHandler()
        self.llm_kernel = Mock()
        self.processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)

        # Create sample XML trees
        self.create_sample_trees()

        # Create sample amendment
        self.amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,  # Use enum
            whole_provision=False,
            location=AmendmentLocation.AFTER,  # Use enum
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

    def create_sample_trees(self):
        """Create sample XML trees for testing."""
        # Working tree (target act)
        self.working_xml = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section class="prov1" eId="sec_1">
                    <heading>Test Section</heading>
                    <subsection class="prov2" eId="sec_1__subsec_1">
                        <intro>
                            <p>Original text.</p>
                        </intro>
                        <level class="para1" eId="sec_1__subsec_1__para_a">
                            <num>(a)</num>
                            <content>
                                <p>More text.</p>
                            </content>
                        </level>
                    </subsection>
                </section>
            </body>
        </act>"""
        self.working_tree = etree.fromstring(self.working_xml.encode()).getroottree()

        # Amending bill tree
        self.amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>In section 1(1), after "text" insert ", including new text".</p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        self.amending_tree = etree.fromstring(self.amending_xml.encode()).getroottree()

    def test_init(self):
        """Test processor initialisation."""
        processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)

        self.assertEqual(processor.xml_handler, self.xml_handler)
        self.assertEqual(processor.llm_kernel, self.llm_kernel)
        self.assertEqual(processor.namespaces, self.xml_handler.namespaces)

    def test_apply_amendment_target_not_found(self):
        """Test applying amendment when target element not found."""
        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=False,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_99__subsec_1",  # Non-existent
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertFalse(success)
        self.assertIn("Target element sec_99__subsec_1 not found", error)

    def test_apply_amendment_source_not_found(self):
        """Test applying amendment when source element not found."""
        amendment = Amendment(
            source_eid="sec_99__subsec_2",  # Non-existent
            source="s. 99(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=False,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertFalse(success)
        self.assertIn("Source element sec_99__subsec_2 not found", error)

    def test_apply_amendment_invalid_type(self):
        """Test applying amendment with invalid type."""
        amendment = Mock()
        amendment.affected_provision = "sec_1__subsec_1"
        amendment.source_eid = "sec_25__subsec_2"
        amendment.amendment_type = Mock()
        amendment.amendment_type.value = "INVALID_TYPE"
        amendment.whole_provision = True

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertFalse(success)
        self.assertIn("Unknown amendment type", error)

    def test_apply_insertion_whole_provision(self):
        """Test applying whole provision insertion."""
        # Create amendment with quoted structure
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>After subsection (1) insert—
                                <quotedStructure>
                                    <subsection eId="sec_1__subsec_1A">
                                        <num>(1A)</num>
                                        <content><p>New subsection text.</p></content>
                                    </subsection>
                                </quotedStructure>
                            </p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

        self.assertTrue(success)
        self.assertIsNone(error)

        # Check that new subsection was inserted
        new_subsection = self.working_tree.find(
            ".//akn:subsection[@eId='sec_1__subsec_1A']", self.xml_handler.namespaces
        )
        self.assertIsNotNone(new_subsection)

        # Check change markup
        self.assertEqual(new_subsection.get(f"{{{XMLHandler.UKL_URI}}}change"), "ins")
        self.assertEqual(new_subsection.get(f"{{{XMLHandler.UKL_URI}}}changeStart"), "true")
        self.assertEqual(new_subsection.get(f"{{{XMLHandler.UKL_URI}}}changeEnd"), "true")
        self.assertIsNotNone(new_subsection.get(f"{{{XMLHandler.UKL_URI}}}changeDnum"))

    def test_apply_insertion_partial_with_llm_response(self):
        """Test applying partial insertion using pre-fetched LLM response."""
        llm_response = f"""<subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Original text<akn:ins ukl:changeStart="true" ukl:changeEnd="true"
                   ukl:changeGenerated="true">, including new text</akn:ins>.</p>
            </content>
        </subsection>"""

        success, error = self.processor.apply_amendment(
            self.amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
        )

        self.assertTrue(success)
        self.assertIsNone(error)

    def test_apply_insertion_partial_no_llm_response(self):
        """Test applying partial insertion without LLM response fails."""
        success, error = self.processor.apply_amendment(
            self.amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertFalse(success)
        self.assertIn("Pre-fetched LLM response required", error)

    def test_apply_deletion_whole_provision(self):
        """Test applying whole provision deletion."""
        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.DELETION,
            whole_provision=True,
            location=AmendmentLocation.REPLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertTrue(success)
        self.assertIsNone(error)

        # Check deletion markup
        target = self.working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)
        self.assertEqual(target.get(f"{{{XMLHandler.UKL_URI}}}change"), "del")
        self.assertEqual(target.get(f"{{{XMLHandler.UKL_URI}}}changeStart"), "true")
        self.assertEqual(target.get(f"{{{XMLHandler.UKL_URI}}}changeEnd"), "true")
        self.assertIsNotNone(target.get(f"{{{XMLHandler.UKL_URI}}}changeDnum"))

    def test_apply_substitution_whole_provision(self):
        """Test applying whole provision substitution."""
        # Create amendment with quoted structure for substitution
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>For subsection (1) substitute—
                                <quotedStructure>
                                    <subsection eId="sec_1__subsec_1">
                                        <num>(1)</num>
                                        <content><p>Replacement text.</p></content>
                                    </subsection>
                                </quotedStructure>
                            </p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=True,
            location=AmendmentLocation.REPLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

        self.assertTrue(success)
        self.assertIsNone(error)

        # Check original marked for deletion
        subsections = self.working_tree.findall(
            ".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces
        )

        # Should have two: deleted original and inserted replacement
        self.assertEqual(len(subsections), 2)

        # First should be marked for deletion
        self.assertEqual(subsections[0].get(f"{{{XMLHandler.UKL_URI}}}change"), "delReplace")

        # Second should be marked for insertion
        self.assertEqual(subsections[1].get(f"{{{XMLHandler.UKL_URI}}}change"), "insReplace")

    def test_llm_response_invalid_xml(self):
        """Test handling invalid XML from LLM."""
        llm_response = "<invalid>unclosed tag"

        success, error = self.processor.apply_amendment(
            self.amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
        )

        self.assertFalse(success)
        self.assertIn("Failed to apply partial amendment", error)

    def test_llm_response_changed_eid(self):
        """Test handling LLM response that changes eId."""
        llm_response = f"""<subsection eId="sec_1__subsec_2"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content><p>Text</p></content>
        </subsection>"""

        success, error = self.processor.apply_amendment(
            self.amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
        )

        self.assertFalse(success)
        self.assertIn("Failed to apply partial amendment", error)

    def test_llm_response_missing_markup(self):
        """Test handling LLM response without required markup."""
        llm_response = f"""<subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}">
            <content><p>Text without any change markup</p></content>
        </subsection>"""

        success, error = self.processor.apply_amendment(
            self.amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
        )

        self.assertFalse(success)
        self.assertIn("Failed to apply partial amendment", error)

    def test_insert_error_comment(self):
        """Test inserting error comment."""
        element = self.working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)

        self.processor.insert_error_comment(element, self.amendment, "Test error message")

        # Check processing instructions were added
        pis = [child for child in element if isinstance(child, etree._ProcessingInstruction)]
        self.assertEqual(len(pis), 2)

        # Check start PI
        self.assertEqual(pis[0].target, "oxy_comment_start")
        # Check that both the standard message and error are included
        self.assertIn("Unable to apply the", pis[0].text)
        self.assertIn("Error: Test error message", pis[0].text)

        # Check end PI
        self.assertEqual(pis[1].target, "oxy_comment_end")

    def test_insert_before_location(self):
        """Test insertion with Before location."""
        # Create amendment with quoted structure
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>Before subsection (1) insert—
                                <quotedStructure>
                                    <subsection eId="sec_1__subsec_0">
                                        <num>(A1)</num>
                                        <content><p>New subsection before.</p></content>
                                    </subsection>
                                </quotedStructure>
                            </p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            location=AmendmentLocation.BEFORE,
            affected_provision="sec_1__subsec_1",
            affected_document="Test Act",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

        self.assertTrue(success)

        # Check order - new subsection should be before original
        section = self.working_tree.find(".//akn:section[@eId='sec_1']", self.xml_handler.namespaces)
        subsections = section.findall(".//akn:subsection", self.xml_handler.namespaces)

        self.assertEqual(subsections[0].get("eId"), "sec_1__subsec_0")
        self.assertEqual(subsections[1].get("eId"), "sec_1__subsec_1")

    def test_apply_deletion_partial_with_llm_response(self):
        """Test applying partial deletion using pre-fetched LLM response."""
        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.DELETION,
            whole_provision=False,
            location=AmendmentLocation.REPLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        llm_response = f"""<subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Original <akn:del ukl:changeStart="true" ukl:changeEnd="true"
                ukl:changeGenerated="true">deleted text</akn:del>.</p>
            </content>
        </subsection>"""

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
        )

        self.assertTrue(success)
        self.assertIsNone(error)

    def test_apply_deletion_partial_no_llm_response(self):
        """Test applying partial deletion without LLM response fails."""
        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.DELETION,
            whole_provision=False,
            location=AmendmentLocation.REPLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertFalse(success)
        self.assertIn("Pre-fetched LLM response required", error)

    def test_apply_substitution_partial_with_llm_response(self):
        """Test applying partial substitution using pre-fetched LLM response."""
        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=False,
            location=AmendmentLocation.REPLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        llm_response = f"""<subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p><akn:del ukl:changeStart="true">Original</akn:del><akn:ins ukl:changeEnd="true"
                ukl:changeGenerated="true">Replacement</akn:ins> text.</p>
            </content>
        </subsection>"""

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
        )

        self.assertTrue(success)
        self.assertIsNone(error)

    def test_apply_substitution_partial_no_llm_response(self):
        """Test applying partial substitution without LLM response fails."""
        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=False,
            location=AmendmentLocation.REPLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertFalse(success)
        self.assertIn("Pre-fetched LLM response required", error)

    def test_no_quoted_structure(self):
        """Test handling missing quoted structure for whole provision."""
        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(
            amendment, self.working_tree, self.amending_tree, "schedule-001"
        )

        self.assertFalse(success)
        self.assertEqual(error, "Whole provision amendment application failed")

    def test_quoted_structure_empty(self):
        """Test handling empty quoted structure."""
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>After subsection (1) insert—
                                <quotedStructure>
                                </quotedStructure>
                            </p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

        self.assertFalse(success)
        self.assertEqual(error, "Whole provision amendment application failed")

    def test_insertion_target_has_no_parent(self):
        """Test handling when target has no parent for insertion."""
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>After subsection (1) insert—
                                <quotedStructure>
                                    <subsection eId="sec_1__subsec_1A">
                                        <num>(1A)</num>
                                        <content><p>New subsection text.</p></content>
                                    </subsection>
                                </quotedStructure>
                            </p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        # Create a detached element with no parent
        detached_target = etree.Element(f"{{{self.xml_handler.namespaces['akn']}}}subsection")
        detached_target.set("eId", "sec_1__subsec_1")

        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        with patch.object(self.xml_handler, "find_element_by_eid") as mock_find:
            mock_find.side_effect = [
                # detached_target,  # Target with no parent
                detached_target,  # Target with no parent
                amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces),
            ]

            success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

            self.assertFalse(success)
            self.assertEqual(error, "Whole provision amendment application failed")

    def test_invalid_location_for_insertion(self):
        """Test handling invalid location value."""
        # Create amendment with quoted structure but invalid location
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>After subsection (1) insert—
                                <quotedStructure>
                                    <subsection eId="sec_1__subsec_1A">
                                        <num>(1A)</num>
                                        <content><p>New subsection text.</p></content>
                                    </subsection>
                                </quotedStructure>
                            </p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        # Create a mock amendment with a non-standard location
        amendment = Mock()
        amendment.source_eid = "sec_25__subsec_2"
        amendment.affected_provision = "sec_1__subsec_1"
        amendment.amendment_type = AmendmentType.INSERTION
        amendment.whole_provision = True
        amendment.location = "INVALID_LOCATION"  # Not BEFORE, AFTER, or REPLACE

        success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

        self.assertFalse(success)
        self.assertEqual(error, "Whole provision amendment application failed")

    def test_target_not_in_parent_children(self):
        """Test handling when target element not found in parent's children."""
        # Create an amending bill with a provision to insert
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <subsection eId="sec_25__subsec_2">
                        <content>
                            <p>After subsection (1) insert—
                                <quotedStructure>
                                    <subsection eId="sec_1__subsec_1A">
                                        <num>(1A)</num>
                                        <content><p>New subsection text.</p></content>
                                    </subsection>
                                </quotedStructure>
                            </p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        # Mock the list() function to simulate a parent that doesn't contain the target element
        calls_count = 0
        original_list = list

        def selective_mock_list(iterable):
            nonlocal calls_count
            calls_count += 1
            result = original_list(iterable)

            # On the second call to list() (which converts parent to a list),
            # return an empty list to trigger ValueError when index() is called
            if calls_count == 2:
                return []
            return result

        # Apply the mock and test that the error is handled correctly
        with patch("app.services.amendment_processor.list", selective_mock_list):
            success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

            self.assertFalse(success)
            self.assertEqual(error, "Whole provision amendment application failed")

    def test_llm_response_no_parent_for_replacement(self):
        """Test handling LLM response when target has no parent."""
        # Create a detached element
        detached = etree.Element(f"{{{XMLHandler.AKN_URI}}}subsection", eId="sec_1__subsec_1")

        llm_response = f"""<subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Text with <akn:ins ukl:changeStart="true" ukl:changeEnd="true"
                ukl:changeGenerated="true">changes</akn:ins>.</p>
            </content>
        </subsection>"""

        with patch.object(self.xml_handler, "find_element_by_eid", return_value=detached):
            success, error = self.processor.apply_amendment(
                self.amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
            )

            self.assertFalse(success)
            self.assertIn("Failed to apply partial amendment", error)

    def test_correct_amendments(self):
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_80" class="prov1">
                    <subsection eId="sec_80__subsec_1" class="prov2">
                        <content>
                            <p>In section 1(1), after "text" insert ", including new text".</p>
                        </content>
                    </subsection>
                    <subsection eId="sec_80__subsec_2" class="prov2">
                        <content>
                            <p>In the heading of section 1, after "word", insert "word".</p>
                        </content>
                    </subsection>
                    <subsection eId="sec_80__subsec_3" class="prov2">
                        <content>
                            <p>In section 1(1)(a), omit "text".</p>
                        </content>
                    </subsection>
                    <subsection eId="sec_80__subsec_4" class="prov2">
                        <content>
                            <p><mod>After section 1, insert-<quotedStructure eId="sec_80__subsec_4__qstr">
                                   <section eId="sec_80__subsec_4__qstr__sec_2" class="prov1">
                                        <num ukl:autonumber="no">2</num>
                                        <content>
                                            <p>text</p>
                                        </content>
                                   </section>
                               </quotedStructure><inline name="AppendText"/></mod></p>
                        </content>
                    </subsection>
                    <subsection eId="sec_80__subsec_5" class="prov2">
                        <content>
                            <p><mod>After paragraph (a), insert-<quotedStructure eId="sec_80__subsec_5__qstr">
                                   <hcontainer class="definition"/>
                               </quotedStructure><inline name="AppendText"/></mod></p>
                        </content>
                    </subsection>
                    <subsection eId="sec_80__subsec_6" class="prov2">
                        <content>
                            <p><mod>After paragraph (a), insert-<quotedStructure eId="sec_80__subsec_6__qstr"/>
                            </mod></p>
                        </content>
                    </subsection>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        amendments = [
            Amendment(
                source_eid="sec_80__subsec_1",
                source="s. 80(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=False,  # Will remain unchanged
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1__subsec_1",  # Will remain unchanged
            ),
            Amendment(
                source_eid="sec_80__subsec_2",
                source="s. 80(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,  # Will be changed to False, because there is no quoted structure
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",  # Will remain unchanged
            ),
            Amendment(
                source_eid="sec_80__subsec_3",
                source="s. 80(3)",
                amendment_type=AmendmentType.DELETION,
                whole_provision=True,  # Will remain unchanged
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1__subsec_1__para_a",  # Will remain unchanged
            ),
            Amendment(
                source_eid="sec_80__subsec_4",
                source="s. 80(4)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,  # Will remain unchanged
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",  # Will remain unchanged
            ),
            Amendment(
                source_eid="sec_80__subsec_5",
                source="s. 80(5)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,  # Will be changed to False, because it targets a low-level provision in QS
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",  # Will remain unchanged
            ),
            Amendment(
                source_eid="sec_80__subsec_6",
                source="s. 80(6)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=True,  # Will remain unchanged, because the QS is malformed
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1",  # Will remain unchanged
            ),
        ]
        self.processor.correct_amendments(amendments, self.working_tree, amending_tree)
        assert amendments[0].whole_provision is False
        assert amendments[0].affected_provision == "sec_1__subsec_1"

        assert amendments[1].whole_provision is False
        assert amendments[1].affected_provision == "sec_1"

        assert amendments[2].whole_provision is True
        assert amendments[2].affected_provision == "sec_1__subsec_1__para_a"

        assert amendments[3].whole_provision is True
        assert amendments[3].affected_provision == "sec_1"

        assert amendments[4].whole_provision is False
        assert amendments[4].affected_provision == "sec_1"

        assert amendments[5].whole_provision is True
        assert amendments[5].affected_provision == "sec_1"

    def test_correct_affected_provision_field_with_no_change(self):
        amendment = Amendment(
            source_eid="sec_80__subsec_1",
            source="s. 80(1)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=False,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_2",  # Will remain unchanged because the affected provision could not be found
        )

        self.processor.correct_affected_provision_field(amendment, self.working_tree)

        assert amendment.whole_provision is False
        assert amendment.affected_provision == "sec_2"

    def test_correct_affected_provision_field_with_change(self):
        amendment = Amendment(
            source_eid="sec_80__subsec_1",
            source="s. 80(1)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=False,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1__def_test",  # Will be changed to it's parent because of whole_provision
        )

        self.processor.correct_affected_provision_field(amendment, self.working_tree)

        assert amendment.whole_provision is False
        assert amendment.affected_provision == "sec_1__subsec_1"

    def test_llm_response_general_exception(self):
        """Test handling general exception in LLM response processing."""
        llm_response = f"""<subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Valid XML with <akn:ins ukl:changeStart="true" ukl:changeEnd="true"
                ukl:changeGenerated="true">changes</akn:ins>.</p>
            </content>
        </subsection>"""

        # Force an exception during tree operations
        with patch.object(self.xml_handler, "parse_xml_string", side_effect=Exception("Unexpected error")):
            success, error = self.processor.apply_amendment(
                self.amendment, self.working_tree, self.amending_tree, "schedule-001", llm_response
            )

            self.assertFalse(success)
            self.assertIn("Failed to apply partial amendment", error)

    def test_prepare_llm_amendment(self):
        """Test prepare_llm_amendment method."""
        target = self.working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)
        source = self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces)

        result = self.processor.prepare_llm_amendment(self.amendment, target, source, "schedule-001")

        self.assertEqual(result["prompt_name"], "ApplyInsertionAmendment")
        self.assertEqual(result["schedule_id"], "schedule-001")
        self.assertEqual(result["amendment_id"], "test-001")
        self.assertIn("source", result["kwargs"])
        self.assertIn("amendment_xml", result["kwargs"])

    def test_validate_llm_response_all_attributes_present(self):
        """Test validation passes when all required attributes are present."""
        # Create valid XML with all attributes
        xml_string = f"""<wrapper><subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Text with <akn:ins ukl:changeStart="true"
                ukl:changeEnd="true" ukl:changeGenerated="true">changes</akn:ins>.</p>
            </content>
        </subsection></wrapper>"""

        element = etree.fromstring(xml_string)

        is_valid, error = self.processor._validate_llm_amendment_response(element, "sec_1__subsec_1", "insertion")

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_element_has_eid_raises_error(self):
        """Test when validate_element_has_eid raises ValueError."""
        # Create an element without eId
        target_no_eid = etree.Element(f"{{{XMLHandler.AKN_URI}}}subsection")
        # Don't set eId attribute

        # Mock find_element_by_eid to return our element without eId
        with patch.object(self.xml_handler, "find_element_by_eid") as mock_find:
            mock_find.side_effect = [
                target_no_eid,  # Target without eId
                target_no_eid,  # Target without eId
                self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces),
            ]

            success, error = self.processor.apply_amendment(
                self.amendment, self.working_tree, self.amending_tree, "schedule-001"
            )

            self.assertFalse(success)
            self.assertIn("missing eId attribute", error)

    def test_apply_amendment_general_exception(self):
        """Test general exception handling in apply_amendment."""
        # Mock find_element_by_eid_components to raise an unexpected exception
        with patch.object(
            self.xml_handler, "find_element_by_eid_components", side_effect=RuntimeError("Unexpected error")
        ):
            success, error = self.processor.apply_amendment(
                self.amendment, self.working_tree, self.amending_tree, "schedule-001"
            )

            self.assertFalse(success)
            self.assertIn("Unexpected error", error)

    def test_prepare_llm_amendment_unknown_type(self):
        """Test prepare_llm_amendment with unknown amendment type."""
        target = self.working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)
        source = self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces)

        # Create amendment with invalid type
        invalid_amendment = Mock()
        invalid_amendment.amendment_type = "INVALID_TYPE"
        invalid_amendment.source = "test source"

        with self.assertRaises(ValueError) as context:
            self.processor.prepare_llm_amendment(invalid_amendment, target, source, "schedule-001")

        self.assertIn("Unknown amendment type", str(context.exception))

    def test_quoted_structure_as_parent(self):
        """Test when source element's parent is quotedStructure."""
        # Create amending XML where quotedStructure is the parent
        amending_xml = f"""<?xml version="1.0"?>
        <bill xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                <section eId="sec_25">
                    <quotedStructure>
                        <subsection eId="sec_25__subsec_2">
                            <num>(2)</num>
                            <content><p>New subsection text.</p></content>
                        </subsection>
                    </quotedStructure>
                </section>
            </body>
        </bill>"""
        amending_tree = etree.fromstring(amending_xml.encode()).getroottree()

        amendment = Amendment(
            source_eid="sec_25__subsec_2",
            source="s. 25(2)",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            location=AmendmentLocation.AFTER,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        success, error = self.processor.apply_amendment(amendment, self.working_tree, amending_tree, "schedule-001")

        self.assertTrue(success)

    def test_prepare_quoted_elements_with_qstr_split(self):
        """Test _prepare_quoted_elements_for_insertion with __qstr__ in eId."""
        # Create element with __qstr__ in eId that splits into exactly 2 parts
        elem = etree.Element(f"{{{XMLHandler.AKN_URI}}}subsection")
        elem.set("eId", "sec_21__subsec_2__qstr__sec_59b")

        # Add a child element with similar eId pattern
        child = etree.SubElement(elem, f"{{{XMLHandler.AKN_URI}}}num")
        child.set("eId", "sec_21__subsec_2__qstr__sec_59b__num")

        elements = [elem]

        # Mock the logger to verify debug message
        with patch("app.services.amendment_processor.logger") as mock_logger:
            self.processor._prepare_quoted_elements_for_insertion(elements)

            # Verify debug log was called
            mock_logger.debug.assert_called_once()
            self.assertIn("Transforming eIds", mock_logger.debug.call_args[0][0])

        # Verify eIds were transformed
        self.assertEqual(elem.get("eId"), "sec_59b")
        self.assertEqual(child.get("eId"), "sec_59b__num")

    def test_validate_llm_response_missing_all_attributes(self):
        """Test validation when all tracking attributes are missing."""
        # Create XML without any change tracking attributes
        xml_string = f"""<wrapper><subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Text with <akn:ins>changes</akn:ins>.</p>
            </content>
        </subsection></wrapper>"""

        element = etree.fromstring(xml_string)

        # Mock the handler methods to return False for all attributes
        with patch.object(self.xml_handler, "element_has_change_markup", return_value=True):
            with patch.object(
                self.xml_handler,
                "get_change_tracking_attributes",
                return_value={"changeStart": False, "changeEnd": False, "changeGenerated": False},
            ):
                is_valid, error = self.processor._validate_llm_amendment_response(
                    element, "sec_1__subsec_1", "insertion"
                )

                self.assertFalse(is_valid)
                self.assertIn("Missing required attributes", error)
                self.assertIn("ukl:changeStart", error)
                self.assertIn("ukl:changeEnd", error)
                self.assertIn("ukl:changeGenerated", error)

    def test_validate_llm_response_missing_some_attributes(self):
        """Test validation when some tracking attributes are missing."""
        xml_string = f"""<wrapper><subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Text with <akn:ins ukl:changeStart="true">changes</akn:ins>.</p>
            </content>
        </subsection></wrapper>"""

        element = etree.fromstring(xml_string)

        # Mock to return only changeStart as true
        with patch.object(self.xml_handler, "element_has_change_markup", return_value=True):
            with patch.object(
                self.xml_handler,
                "get_change_tracking_attributes",
                return_value={"changeStart": True, "changeEnd": False, "changeGenerated": False},
            ):
                is_valid, error = self.processor._validate_llm_amendment_response(
                    element, "sec_1__subsec_1", "insertion"
                )

                self.assertFalse(is_valid)
                self.assertIn("ukl:changeEnd", error)
                self.assertIn("ukl:changeGenerated", error)
                self.assertNotIn("ukl:changeStart", error)

    def test_validate_llm_response_missing_some_attributes_multiple_elements(self):
        """Test validation when some tracking attributes are missing."""
        xml_string = f"""<wrapper><subsection eId="sec_1__subsec_1"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Text with <akn:ins ukl:changeStart="true">changes</akn:ins>.</p>
            </content>
        </subsection>
        <subsection eId="sec_1__subsec_2"
            xmlns:akn="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <content>
                <p>Text with <akn:ins ukl:changeStart="true">changes</akn:ins>.</p>
            </content>
        </subsection></wrapper>"""

        element = etree.fromstring(xml_string)

        # Mock to return only changeStart as true
        with patch.object(self.xml_handler, "element_has_change_markup", return_value=True):
            with patch.object(
                self.xml_handler,
                "get_change_tracking_attributes",
                return_value={"changeStart": True, "changeEnd": False, "changeGenerated": False},
            ):
                is_valid, error = self.processor._validate_llm_amendment_response(
                    element, "sec_1__subsec_1", "insertion"
                )

                self.assertFalse(is_valid)
                self.assertIn("ukl:changeEnd", error)
                self.assertIn("ukl:changeGenerated", error)
                self.assertNotIn("ukl:changeStart", error)

    def test_insert_oxy_comment_with_all_pi_children(self):
        """Test _insert_oxy_comment when element has only ProcessingInstruction children."""
        element = etree.Element(f"{{{XMLHandler.AKN_URI}}}section")

        # Add only PI children
        pi1 = etree.ProcessingInstruction("existing_pi", "data1")
        pi2 = etree.ProcessingInstruction("another_pi", "data2")
        element.append(pi1)
        element.append(pi2)

        self.processor._insert_oxy_comment(element, "Test Author", "Test comment")

        # Check that new PIs were added at the end
        pis = [child for child in element if isinstance(child, etree._ProcessingInstruction)]
        self.assertEqual(len(pis), 4)  # 2 existing + 2 new

        # Check the new PIs are at the end
        self.assertEqual(pis[2].target, "oxy_comment_start")
        self.assertEqual(pis[3].target, "oxy_comment_end")

    def test_insert_oxy_comment_with_no_children(self):
        """Test _insert_oxy_comment when element has no children."""
        element = etree.Element(f"{{{XMLHandler.AKN_URI}}}section")

        self.processor._insert_oxy_comment(element, "Test Author", "Test comment")

        # Check that PIs were added
        pis = [child for child in element if isinstance(child, etree._ProcessingInstruction)]
        self.assertEqual(len(pis), 2)
        self.assertEqual(pis[0].target, "oxy_comment_start")
        self.assertEqual(pis[1].target, "oxy_comment_end")

    def test_insert_all_error_comments_with_multiple_records(self):
        """Test inserting error comments for multiple failed amendments."""
        # Create mock tracker
        mock_tracker = Mock()

        # Create mock records
        record1 = Mock()
        record1.source_eid = "sec_1"
        record1.source = "s. 1"
        record1.amendment_type = "INSERTION"
        record1.whole_provision = True
        record1.location = "AFTER"
        record1.affected_provision = "sec_2"
        record1.amendment_id = "id1"
        record1.error_message = "Custom error message"

        record2 = Mock()
        record2.source_eid = "sec_3"
        record2.source = "s. 3"
        record2.amendment_type = "DELETION"
        record2.whole_provision = False
        record2.location = "REPLACE"
        record2.affected_provision = "sec_4"
        record2.amendment_id = "id2"
        record2.error_message = None  # Test default message

        mock_tracker.get_all_requiring_comments.return_value = [record1, record2]

        # Create mock XML tree with target elements
        output_xml = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_2">
                    <heading>Section 2</heading>
                </section>
                <section eId="sec_4">
                    <heading>Section 4</heading>
                </section>
            </body>
        </act>"""
        output_act = etree.fromstring(output_xml.encode()).getroottree()

        # Call the method
        self.processor.insert_all_error_comments(output_act, mock_tracker)

        # Verify error comments were added
        sec2 = output_act.find(".//akn:section[@eId='sec_2']", self.xml_handler.namespaces)
        sec4 = output_act.find(".//akn:section[@eId='sec_4']", self.xml_handler.namespaces)

        # Check PIs were added to both sections
        sec2_pis = [child for child in sec2 if isinstance(child, etree._ProcessingInstruction)]
        sec4_pis = [child for child in sec4 if isinstance(child, etree._ProcessingInstruction)]

        self.assertEqual(len(sec2_pis), 2)
        self.assertEqual(len(sec4_pis), 2)

        # Verify custom error message was used for record1
        self.assertIn("Custom error message", sec2_pis[0].text)

        # Verify default message was used for record2
        self.assertIn("Amendment could not be applied", sec4_pis[0].text)

        # Verify tracker was updated
        mock_tracker.mark_error_commented.assert_any_call("id1")
        mock_tracker.mark_error_commented.assert_any_call("id2")
        self.assertEqual(mock_tracker.mark_error_commented.call_count, 2)

    def test_insert_all_error_comments_target_not_found(self):
        """Test handling when target element cannot be found."""
        # Create mock tracker
        mock_tracker = Mock()

        # Create mock record with non-existent target
        record = Mock()
        record.source_eid = "sec_1"
        record.source = "s. 1"
        record.amendment_type = "INSERTION"
        record.whole_provision = True
        record.location = "AFTER"
        record.affected_provision = "non_existent_section"
        record.amendment_id = "id1"
        record.error_message = "Error message"

        mock_tracker.get_all_requiring_comments.return_value = [record]

        # Create empty XML tree
        output_xml = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
            </body>
        </act>"""
        output_act = etree.fromstring(output_xml.encode()).getroottree()

        # Mock find_element_by_eid_with_fallback to return None
        with patch.object(self.xml_handler, "find_element_by_eid_with_fallback", return_value=None):
            # Mock logger to verify error was logged
            with patch("app.services.amendment_processor.logger") as mock_logger:
                self.processor.insert_all_error_comments(output_act, mock_tracker)

                # Verify error was logged
                mock_logger.error.assert_called_once_with(
                    "Could not find target for error comment: non_existent_section"
                )

        # Verify tracker was NOT updated for this record
        mock_tracker.mark_error_commented.assert_not_called()

    def test_insert_all_error_comments_empty_list(self):
        """Test handling when no amendments require comments."""
        # Create mock tracker that returns empty list
        mock_tracker = Mock()
        mock_tracker.get_all_requiring_comments.return_value = []

        # Create XML tree
        output_xml = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <heading>Section 1</heading>
                </section>
            </body>
        </act>"""
        output_act = etree.fromstring(output_xml.encode()).getroottree()

        # Call the method
        self.processor.insert_all_error_comments(output_act, mock_tracker)

        # Verify no comments were added
        section = output_act.find(".//akn:section[@eId='sec_1']", self.xml_handler.namespaces)
        pis = [child for child in section if isinstance(child, etree._ProcessingInstruction)]
        self.assertEqual(len(pis), 0)

        # Verify tracker was not called
        mock_tracker.mark_error_commented.assert_not_called()

    def test_insert_all_error_comments_mixed_results(self):
        """Test with mix of successful and failed comment insertions."""
        # Create mock tracker
        mock_tracker = Mock()

        # Create records
        record1 = Mock()
        record1.source_eid = "sec_1"
        record1.source = "s. 1"
        record1.amendment_type = "SUBSTITUTION"
        record1.whole_provision = True
        record1.location = "REPLACE"
        record1.affected_provision = "sec_exists"
        record1.amendment_id = "id1"
        record1.error_message = "Failed to substitute"

        record2 = Mock()
        record2.source_eid = "sec_2"
        record2.source = "s. 2"
        record2.amendment_type = "INSERTION"
        record2.whole_provision = False
        record2.location = "BEFORE"
        record2.affected_provision = "sec_missing"
        record2.amendment_id = "id2"
        record2.error_message = "Failed to insert"

        mock_tracker.get_all_requiring_comments.return_value = [record1, record2]

        # Create XML tree with only one section
        output_xml = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_exists">
                    <heading>Existing Section</heading>
                </section>
            </body>
        </act>"""
        output_act = etree.fromstring(output_xml.encode()).getroottree()

        # Mock find_element_by_eid_with_fallback to return element for first, None for second
        with patch.object(self.xml_handler, "find_element_by_eid_with_fallback") as mock_find:
            existing_section = output_act.find(".//akn:section[@eId='sec_exists']", self.xml_handler.namespaces)
            mock_find.side_effect = [existing_section, None]

            with patch("app.services.amendment_processor.logger") as mock_logger:
                self.processor.insert_all_error_comments(output_act, mock_tracker)

                # Verify error logged for missing target
                mock_logger.error.assert_called_once_with("Could not find target for error comment: sec_missing")

        # Verify comment added to existing section
        section = output_act.find(".//akn:section[@eId='sec_exists']", self.xml_handler.namespaces)
        pis = [child for child in section if isinstance(child, etree._ProcessingInstruction)]
        self.assertEqual(len(pis), 2)
        self.assertIn("Failed to substitute", pis[0].text)

        # Verify only successful one was marked in tracker
        mock_tracker.mark_error_commented.assert_called_once_with("id1")

    def test_tokenizer_initialisation_openai_success(self):
        """Test successful OpenAI tokenizer initialisation."""
        # Mock LLM config for OpenAI
        mock_llm_config = Mock()
        mock_llm_config.enable_azure_openai = True
        mock_llm_config.enable_aws_bedrock = False

        mock_llm_kernel = Mock()
        mock_llm_kernel.llm_config = mock_llm_config

        # Create processor - should initialise tokenizer successfully
        processor = AmendmentProcessor(self.xml_handler, mock_llm_kernel)

        # Verify tokenizer was initialised (tiktoken's cl100k_base encoding)
        self.assertIsNotNone(processor.tokenizer)

    def test_tokenizer_initialisation_openai_failure(self):
        """Test OpenAI tokenizer initialisation failure with fallback."""
        # Mock LLM config for OpenAI
        mock_llm_config = Mock()
        mock_llm_config.enable_azure_openai = True
        mock_llm_config.enable_aws_bedrock = False

        mock_llm_kernel = Mock()
        mock_llm_kernel.llm_config = mock_llm_config

        # Mock tiktoken to raise an exception
        with patch("app.services.amendment_processor.tiktoken.get_encoding", side_effect=Exception("Encoding error")):
            with patch("app.services.amendment_processor.logger") as mock_logger:
                processor = AmendmentProcessor(self.xml_handler, mock_llm_kernel)

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                self.assertIn("Failed to initialise OpenAI tokenizer", mock_logger.warning.call_args[0][0])

                # Verify tokenizer is None (will use character-based estimation)
                self.assertIsNone(processor.tokenizer)

    def test_tokenizer_initialisation_bedrock(self):
        """Test tokenizer initialisation for AWS Bedrock (Claude)."""
        # Mock LLM config for Bedrock
        mock_llm_config = Mock()
        mock_llm_config.enable_azure_openai = False
        mock_llm_config.enable_aws_bedrock = True

        mock_llm_kernel = Mock()
        mock_llm_kernel.llm_config = mock_llm_config

        with patch("app.services.amendment_processor.logger") as mock_logger:
            processor = AmendmentProcessor(self.xml_handler, mock_llm_kernel)

            # Verify debug message was logged
            mock_logger.debug.assert_called_with("Using character-based token estimation for Claude models")

            # Verify tokenizer is None (character-based estimation)
            self.assertIsNone(processor.tokenizer)

    def test_tokenizer_initialisation_no_llm_kernel(self):
        """Test tokenizer initialisation when no LLM kernel provided."""
        processor = AmendmentProcessor(self.xml_handler, None)

        # Verify tokenizer is None when no LLM kernel
        self.assertIsNone(processor.tokenizer)

    def test_check_token_limits_within_limits(self):
        """Test check_token_limits when content is within limits."""
        # Set up processor with mocked LLM config
        mock_llm_config = Mock()
        mock_llm_config.get_max_completion_tokens.return_value = 16384  # GPT-4o limit

        self.llm_kernel.llm_config = mock_llm_config
        self.processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)

        # Create a small target element
        target_xml = f"""<subsection eId="sec_1__subsec_1" xmlns="{XMLHandler.AKN_URI}">
            <content><p>Short text that won't exceed limits.</p></content>
        </subsection>"""
        target = etree.fromstring(target_xml)

        source = self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces)

        # Test the method
        is_within_limits, estimated_tokens = self.processor.check_token_limits(self.amendment, target, source)

        self.assertTrue(is_within_limits)
        self.assertGreater(estimated_tokens, 0)
        self.assertLess(estimated_tokens, 16384)

    def test_check_token_limits_exceeds_limits(self):
        """Test check_token_limits when content exceeds limits."""
        # Set up processor with mocked LLM config
        mock_llm_config = Mock()
        mock_llm_config.get_max_completion_tokens.return_value = 100  # Artificially low limit

        self.llm_kernel.llm_config = mock_llm_config
        self.processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)

        # Create a large target element
        large_content = "This is a very long text. " * 100
        target_xml = f"""<subsection eId="sec_1__subsec_1" xmlns="{XMLHandler.AKN_URI}">
            <content><p>{large_content}</p></content>
        </subsection>"""
        target = etree.fromstring(target_xml)

        source = self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces)

        # Test the method
        is_within_limits, estimated_tokens = self.processor.check_token_limits(self.amendment, target, source)

        self.assertFalse(is_within_limits)
        self.assertGreater(estimated_tokens, 100)

    def test_check_token_limits_with_tiktoken(self):
        """Test check_token_limits using tiktoken for accurate counting."""
        # Set up processor with tiktoken
        mock_llm_config = Mock()
        mock_llm_config.get_max_completion_tokens.return_value = 16384
        mock_llm_config.enable_azure_openai = True

        mock_llm_kernel = Mock()
        mock_llm_kernel.llm_config = mock_llm_config

        # Create processor with real tiktoken
        processor = AmendmentProcessor(self.xml_handler, mock_llm_kernel)

        # Create target element
        target_xml = f"""<subsection eId="sec_1__subsec_1" xmlns="{XMLHandler.AKN_URI}">
            <content><p>This is some test content for token counting.</p></content>
        </subsection>"""
        target = etree.fromstring(target_xml)

        source = self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces)

        with patch("app.services.amendment_processor.logger") as mock_logger:
            is_within_limits, estimated_tokens = processor.check_token_limits(self.amendment, target, source)

            # Verify debug logging
            mock_logger.debug.assert_called_once()
            self.assertIn("Token estimation for amendment", mock_logger.debug.call_args[0][0])

            self.assertTrue(is_within_limits)
            self.assertGreater(estimated_tokens, 0)

    def test_check_token_limits_exception_handling(self):
        """Test check_token_limits handles exceptions gracefully."""
        # Set up processor
        mock_llm_config = Mock()
        mock_llm_config.get_max_completion_tokens.side_effect = Exception("Config error")

        self.llm_kernel.llm_config = mock_llm_config
        self.processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)

        target = self.working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)
        source = self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces)

        with patch("app.services.amendment_processor.logger") as mock_logger:
            # Should fail open (return True, 0)
            is_within_limits, estimated_tokens = self.processor.check_token_limits(self.amendment, target, source)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            self.assertIn("Error estimating tokens", mock_logger.warning.call_args[0][0])

            # Verify fail-open behaviour
            self.assertTrue(is_within_limits)
            self.assertEqual(estimated_tokens, 0)

    def test_estimate_tokens_with_tiktoken(self):
        """Test _estimate_tokens using tiktoken."""
        # Set up processor with tiktoken
        mock_llm_config = Mock()
        mock_llm_config.enable_azure_openai = True

        mock_llm_kernel = Mock()
        mock_llm_kernel.llm_config = mock_llm_config

        processor = AmendmentProcessor(self.xml_handler, mock_llm_kernel)

        test_text = "This is a test string for token counting."

        # Call the method
        token_count = processor._estimate_tokens(test_text)

        # Should return actual token count (not character-based)
        self.assertGreater(token_count, 0)
        # Tiktoken should give us approximately 9-10 tokens for this text
        self.assertLess(token_count, 15)

    def test_estimate_tokens_tiktoken_error(self):
        """Test _estimate_tokens when tiktoken encode fails."""
        # Create a mock tokenizer that raises an exception
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Encoding failed")

        processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)
        processor.tokenizer = mock_tokenizer

        test_text = "This is a test string."

        with patch("app.services.amendment_processor.logger") as mock_logger:
            token_count = processor._estimate_tokens(test_text)

            # Should fall back to character-based estimation
            mock_logger.debug.assert_called_once()
            self.assertIn("Tokenizer error", mock_logger.debug.call_args[0][0])

            # Character-based: len(text) / 2.92
            expected_tokens = int(len(test_text) / 2.92)
            self.assertEqual(token_count, expected_tokens)

    def test_estimate_tokens_character_based(self):
        """Test _estimate_tokens using character-based estimation."""
        # Processor without tokenizer (e.g., Bedrock/Claude)
        processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)
        processor.tokenizer = None

        # Test with known text length
        test_text = "a" * 292  # 292 characters

        token_count = processor._estimate_tokens(test_text)

        # Should be 292 / 2.92 = 100
        self.assertEqual(token_count, 100)

    def test_estimate_tokens_empty_string(self):
        """Test _estimate_tokens with empty string."""
        processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)
        processor.tokenizer = None

        token_count = processor._estimate_tokens("")

        self.assertEqual(token_count, 0)

    def test_estimate_tokens_unicode_content(self):
        """Test _estimate_tokens with Unicode content."""
        processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)
        processor.tokenizer = None

        # Unicode text with special characters
        test_text = "Testing with Unicode: £€¥ and émojis"

        token_count = processor._estimate_tokens(test_text)

        # Should handle Unicode correctly
        expected_tokens = int(len(test_text) / 2.92)
        self.assertEqual(token_count, expected_tokens)

    def test_check_token_limits_debug_logging(self):
        """Test debug logging in check_token_limits."""
        # Set up processor
        mock_llm_config = Mock()
        mock_llm_config.get_max_completion_tokens.return_value = 16384

        self.llm_kernel.llm_config = mock_llm_config
        self.processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)

        # Set amendment_id for logging
        self.amendment.amendment_id = "test-123"

        target = self.working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)
        source = self.amending_tree.find(".//akn:subsection[@eId='sec_25__subsec_2']", self.xml_handler.namespaces)

        with patch("app.services.amendment_processor.logger") as mock_logger:
            is_within_limits, estimated_tokens = self.processor.check_token_limits(self.amendment, target, source)

            # Check debug message format
            debug_call = mock_logger.debug.call_args[0][0]
            self.assertIn("Token estimation for amendment test-123", debug_call)
            self.assertIn("target provision tokens=", debug_call)
            self.assertIn("max_completion=16384", debug_call)


class TestEachPlaceAmendments(unittest.TestCase):
    """Test cases for 'each place' amendment functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.xml_handler = XMLHandler()
        self.llm_kernel = Mock()
        self.processor = AmendmentProcessor(self.xml_handler, self.llm_kernel)

    def create_test_xml(self, content):
        """Helper to create test XML."""
        xml = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <body>
                {content}
            </body>
        </act>"""
        return etree.fromstring(xml.encode()).getroottree()

    # ==================== Tests for apply_each_place_amendment ====================

    def test_apply_each_place_amendment_success(self):
        """Test successful application of each place amendment."""
        # Create XML with multiple occurrences
        working_tree = self.create_test_xml("""
            <section eId="sec_1" class="prov1">
                <heading>Test Section</heading>
                <subsection eId="sec_1__subsec_1" class="prov2">
                    <content>
                        <p>The company shall notify the company when the company is ready.</p>
                    </content>
                </subsection>
            </section>
        """)

        amending_tree = self.create_test_xml("""
            <section eId="sec_25">
                <content><p>Amendment text</p></content>
            </section>
        """)

        amendment = Amendment(
            source_eid="sec_25",
            source="s. 25",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=False,
            location=AmendmentLocation.EACH_PLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-001",
        )

        pattern_data = {"find_text": "company", "replace_text": "corporation"}

        success, error = self.processor.apply_each_place_amendment(
            amendment, working_tree, amending_tree, pattern_data, "schedule-001"
        )

        self.assertTrue(success)
        self.assertIsNone(error)

        # Verify changes were made
        subsection = working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)

        # Check that del and ins elements were created for each occurrence
        del_elements = subsection.findall(".//akn:del", self.xml_handler.namespaces)
        ins_elements = subsection.findall(".//akn:ins", self.xml_handler.namespaces)

        self.assertEqual(len(del_elements), 3)  # 3 occurrences of "company"
        self.assertEqual(len(ins_elements), 3)  # 3 replacements with "corporation"

    def test_apply_each_place_amendment_no_find_text(self):
        """Test when pattern_data has no find_text."""
        working_tree = self.create_test_xml("<section eId='sec_1'><p>Text</p></section>")
        amending_tree = self.create_test_xml("<section eId='sec_25'><p>Amendment</p></section>")

        amendment = Amendment(
            source_eid="sec_25",
            source="s. 25",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=False,
            location=AmendmentLocation.EACH_PLACE,
            affected_document="Test Act",
            affected_provision="sec_1",
            amendment_id="test-001",
        )

        pattern_data = {"find_text": "", "replace_text": "corporation"}  # Empty find_text

        success, error = self.processor.apply_each_place_amendment(
            amendment, working_tree, amending_tree, pattern_data, "schedule-001"
        )

        self.assertFalse(success)
        self.assertEqual(error, "No find_text in pattern data")

    def test_apply_each_place_amendment_affected_provision_not_found(self):
        """Test when affected provision cannot be found."""
        working_tree = self.create_test_xml("<section eId='sec_1'><p>Text</p></section>")
        amending_tree = self.create_test_xml("<section eId='sec_25'><p>Amendment</p></section>")

        amendment = Amendment(
            source_eid="sec_25",
            source="s. 25",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=False,
            location=AmendmentLocation.EACH_PLACE,
            affected_document="Test Act",
            affected_provision="sec_99",  # Non-existent
            amendment_id="test-001",
        )

        pattern_data = {"find_text": "Text", "replace_text": "Content"}

        success, error = self.processor.apply_each_place_amendment(
            amendment, working_tree, amending_tree, pattern_data, "schedule-001"
        )

        self.assertFalse(success)
        self.assertEqual(error, "Affected provision sec_99 not found")

    def test_apply_each_place_amendment_pattern_not_found(self):
        """Test when the pattern is not found in the text."""
        working_tree = self.create_test_xml("""
            <section eId="sec_1">
                <content><p>This is some text without the pattern.</p></content>
            </section>
        """)
        amending_tree = self.create_test_xml("<section eId='sec_25'><p>Amendment</p></section>")

        amendment = Amendment(
            source_eid="sec_25",
            source="s. 25",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=False,
            location=AmendmentLocation.EACH_PLACE,
            affected_document="Test Act",
            affected_provision="sec_1",
            amendment_id="test-001",
        )

        pattern_data = {"find_text": "company", "replace_text": "corporation"}  # Not in the text

        success, error = self.processor.apply_each_place_amendment(
            amendment, working_tree, amending_tree, pattern_data, "schedule-001"
        )

        self.assertFalse(success)
        self.assertEqual(error, "Pattern 'company' not found in sec_1")

    def test_apply_each_place_amendment_exception_handling(self):
        """Test exception handling in apply_each_place_amendment."""
        working_tree = self.create_test_xml("<section eId='sec_1'><p>Text</p></section>")
        amending_tree = self.create_test_xml("<section eId='sec_25'><p>Amendment</p></section>")

        amendment = Amendment(
            source_eid="sec_25",
            source="s. 25",
            amendment_type=AmendmentType.SUBSTITUTION,
            whole_provision=False,
            location=AmendmentLocation.EACH_PLACE,
            affected_document="Test Act",
            affected_provision="sec_1",
            amendment_id="test-001",
        )

        pattern_data = {"find_text": "Text", "replace_text": "Content"}

        # Mock to raise an exception
        with patch.object(self.xml_handler, "find_element_by_eid", side_effect=Exception("Test error")):
            success, error = self.processor.apply_each_place_amendment(
                amendment, working_tree, amending_tree, pattern_data, "schedule-001"
            )

            self.assertFalse(success)
            self.assertEqual(error, "Unexpected error: Test error")

    # ==================== Tests for _replace_text_occurrences_iteratively ====================

    def test_replace_text_occurrences_iteratively_simple(self):
        """Test iterative replacement with simple text."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}">The company notifies the company.</p>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        self.assertEqual(len(changes_made), 1)  # One text node processed
        self.assertEqual(changes_made[0]["occurrences"], 2)  # Two occurrences

    def test_replace_text_occurrences_iteratively_with_tail(self):
        """Test iterative replacement with tail text."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}">Start <span>middle</span> company end company.</p>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        self.assertEqual(len(changes_made), 1)  # One tail text processed
        self.assertEqual(changes_made[0]["attr"], "tail")

    def test_replace_text_occurrences_iteratively_skip_quoted_structure(self):
        """Test that quoted structures are skipped."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}">
            The company is here.
            <quotedStructure>
                <p>The company should be skipped.</p>
            </quotedStructure>
            Another company here.
        </p>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should only process text outside quotedStructure
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 2)  # Not 3

    def test_replace_text_occurrences_iteratively_no_matches(self):
        """Test when no text matches are found."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}">No matching text here.</p>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertFalse(result)
        self.assertEqual(len(changes_made), 0)

    def test_replace_text_occurrences_iteratively_complex_nesting(self):
        """Test with deeply nested elements."""
        xml = f"""<section xmlns="{XMLHandler.AKN_URI}">
            <p>Level 1 company
                <span>Level 2 company
                    <b>Level 3 company</b>
                </span>
            </p>
        </section>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 3)

    def test_replace_text_occurrences_skip_ins_elements(self):
        """Test that text within <ins> elements is skipped during 'each place' amendments."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            The hazard exists on the premises.
            The hazard<ins ukl:changeStart="true" ukl:changeEnd="true" ukl:changeGenerated="true"> or failure</ins>
            must be addressed.
        </p>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "or failure", "or failures", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertFalse(result)
        # Should find 0 occurrences - the "or failure" is inside <ins>
        self.assertEqual(len(changes_made), 0)

        # Verify the <ins> element wasn't modified
        ins_elements = element.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(ins_elements), 1)
        self.assertEqual(ins_elements[0].text, " or failure")  # Original text unchanged

    def test_replace_text_occurrences_skip_del_elements(self):
        """Test that text within <del> elements is skipped."""
        xml = f"""<intro xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <p>The notice must specify, in relation to the hazard
            <del ukl:changeStart="true">(or each of the hazards)</del>
            <ins ukl:changeEnd="true" ukl:changeGenerated="true">or each of the hazards</ins>
            to which it relates—</p>
        </intro>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "or", "and", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertFalse(result)  # No changes made
        self.assertEqual(len(changes_made), 0)  # No occurrences found

        # Verify the <del> and <ins> elements weren't modified
        del_elements = element.findall(".//akn:del", self.xml_handler.namespaces)
        ins_elements = element.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(del_elements), 1)
        self.assertEqual(len(ins_elements), 1)
        self.assertIn("or each of the hazards", del_elements[0].text)
        self.assertIn("or each of the hazards", ins_elements[0].text)

    def test_replace_text_occurrences_complex_real_scenario(self):
        """Test observed bug scenario with multiple 'or failure' insertions."""
        xml = f"""<subsection xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}" eId="sec_28__subsec_4">
            <num>(4)</num>
            <intro>
                <p>The notice may not be served unless the authority are satisfied—</p>
            </intro>
            <level class="para1" eId="sec_28__subsec_4__para_a">
                <num>(a)</num>
                <content>
                    <p>that the deficiency from which the hazard<ins ukl:changeDnum="KS85" ukl:changeStart="true"
                    ukl:changeEnd="true" ukl:changeGenerated="true"> or failure</ins> arises is situated there, and</p>
                </content>
            </level>
        </subsection>"""
        element = etree.fromstring(xml)
        changes_made = []

        # Apply "each place" amendment to replace "or failure" with "or failures"
        result = self.processor._replace_text_occurrences_iteratively(
            element, "or failure", "or failures", AmendmentType.SUBSTITUTION, changes_made
        )

        # Should find 0 occurrences because "or failure" only exists inside <ins>
        self.assertFalse(result)  # No changes made
        self.assertEqual(len(changes_made), 0)

        # Verify the inserted text remains unchanged
        ins_elements = element.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(ins_elements), 1)
        self.assertEqual(ins_elements[0].text, " or failure")  # Not "or failures"

    def test_replace_text_occurrences_mixed_content(self):
        """Test with both original text and change tracking elements."""
        xml = f"""<content xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <p>if the residential premises on which the hazard exists<ins ukl:changeDnum="KS83"
            ukl:changeStart="true" ukl:changeEnd="true" ukl:changeGenerated="true">, or which fail to meet the
            requirement,</ins> are a dwelling or HMO</p>
            <p>the hazard exists on other premises</p>
        </content>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "hazard exists", "hazard is present", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should find 2 occurrences (both in main text, not in <ins>)
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 2)

        # Verify total <ins> elements after processing
        ins_elements = element.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(ins_elements), 3)  # 1 original + 2 new from substitutions

    def test_replace_text_occurrences_adjacent_del_ins_elements(self):
        """Test with adjacent <del> and <ins> elements (typical substitution pattern)."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            A notice under this section must specify, in relation to the
            <del ukl:changeStart="true">hazard</del><ins ukl:changeEnd="true"
            ukl:changeGenerated="true">hazard or failure</ins>
            (or each of the hazards<ins ukl:changeStart="true" ukl:changeEnd="true"
            ukl:changeGenerated="true"> or failures</ins>)
            to which it relates. The hazard must be addressed promptly.
        </p>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "hazard", "issue", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should find 1 occurrence - the standalone "hazard" at the end
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 1)

        # Get original elements before the new ones are created
        all_ins_before = element.findall(".//akn:ins", self.xml_handler.namespaces)

        # Verify original change elements weren't modified
        del_elements = element.findall(".//akn:del", self.xml_handler.namespaces)
        self.assertEqual(del_elements[0].text, "hazard")  # Original del unchanged

        # Check the original ins elements by their text content
        self.assertEqual(all_ins_before[0].text, "hazard or failure")  # First ins
        self.assertEqual(all_ins_before[1].text, " or failures")  # Second ins

        # Verify the new substitution was created
        all_del_after = element.findall(".//akn:del", self.xml_handler.namespaces)
        all_ins_after = element.findall(".//akn:ins", self.xml_handler.namespaces)

        # Should have 2 del elements (1 original + 1 new) and 3 ins elements (2 original + 1 new)
        self.assertEqual(len(all_del_after), 2)
        self.assertEqual(len(all_ins_after), 3)

    def test_replace_text_occurrences_tail_text_not_skipped(self):
        """Test that tail text of <ins> and <del> elements is NOT skipped."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            The <ins ukl:changeStart="true" ukl:changeEnd="true">inserted</ins> company provides services.
            The <del ukl:changeStart="true" ukl:changeEnd="true">deleted</del> company manages assets.
            The company operates globally.
        </p>"""
        element = etree.fromstring(xml)
        changes_made = []

        # Get references to original elements before processing
        original_ins = element.find(".//akn:ins", self.xml_handler.namespaces)
        original_del = element.find(".//akn:del", self.xml_handler.namespaces)

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should find all 3 occurrences of "company" in tail text
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 3)

        # Verify text inside original ins/del was skipped
        self.assertEqual(original_ins.text, "inserted")  # Not changed
        self.assertEqual(original_del.text, "deleted")  # Not changed

    def test_replace_text_occurrences_nested_change_elements(self):
        """Test that nested elements within <ins>/<del> are also skipped."""
        xml = f"""<p xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            Regular text with hazard here.
            <ins ukl:changeStart="true" ukl:changeEnd="true">
                Added text with hazard and
                <span>nested hazard in span</span>
                more hazard text.
            </ins>
            Final hazard reference.
        </p>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "hazard", "risk", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should find 2 occurrences (first and last, not the 3 inside <ins>)
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 2)

    def test_replace_text_occurrences_skip_deleted_sections(self):
        """Test that elements marked with ukl:change='del' are skipped entirely."""
        xml = f"""<body xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <section eId="sec_4">
                <num>4</num>
                <heading>Active Section</heading>
                <p>This company should be found and replaced.</p>
            </section>
            <section eId="sec_5" ukl:change="del" ukl:changeStart="true" ukl:changeEnd="true">
                <num>5</num>
                <heading>Deleted Section</heading>
                <subsection>
                    <p>This company should be skipped entirely.</p>
                    <p>Another company reference that should also be skipped.</p>
                </subsection>
            </section>
            <section eId="sec_6">
                <num>6</num>
                <heading>Another Active Section</heading>
                <p>This company should also be found.</p>
            </section>
        </body>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should find 2 occurrences (in sections 4 and 6, not in deleted section 5)
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 2)

        # Verify the deleted section wasn't modified
        deleted_section = element.find(".//akn:section[@eId='sec_5']", self.xml_handler.namespaces)
        deleted_paragraphs = deleted_section.findall(".//akn:p", self.xml_handler.namespaces)
        self.assertIn("This company should be skipped", deleted_paragraphs[0].text)
        self.assertIn("Another company reference", deleted_paragraphs[1].text)

        # Verify no ins/del elements were created in the deleted section
        deleted_section_changes = deleted_section.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(deleted_section_changes), 0)

    def test_replace_text_occurrences_skip_inserted_sections(self):
        """Test that elements marked with ukl:change='ins' are skipped entirely."""
        xml = f"""<body xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <section eId="sec_4">
                <num>4</num>
                <heading>Active Section</heading>
                <p>This company should be found and replaced.</p>
            </section>
            <section eId="sec_4A" ukl:change="ins" ukl:changeStart="true" ukl:changeEnd="true">
                <num>4A</num>
                <heading>Newly Inserted Section</heading>
                <subsection>
                    <p>This company is in a newly inserted section.</p>
                    <p>Another company reference in the new section.</p>
                </subsection>
            </section>
            <section eId="sec_5">
                <num>5</num>
                <heading>Another Active Section</heading>
                <p>This company should also be found.</p>
            </section>
        </body>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should find 2 occurrences (in sections 4 and 5, not in inserted section 4A)
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 2)

        # Verify the inserted section wasn't modified
        inserted_section = element.find(".//akn:section[@eId='sec_4A']", self.xml_handler.namespaces)
        inserted_paragraphs = inserted_section.findall(".//akn:p", self.xml_handler.namespaces)
        self.assertIn("This company is in a newly inserted section", inserted_paragraphs[0].text)
        self.assertIn("Another company reference", inserted_paragraphs[1].text)

        # Verify no additional ins/del elements were created in the inserted section
        inserted_section_changes = inserted_section.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(inserted_section_changes), 0)

    def test_replace_text_occurrences_skip_substitution_pairs(self):
        """Test that substitution pairs (delReplace/insReplace) are skipped entirely."""
        xml = f"""<body xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <section eId="sec_4">
                <num>4</num>
                <heading>Active Section</heading>
                <p>This company should be found and replaced.</p>
            </section>
            <subsection eId="sec_5__subsec_2_old" ukl:change="delReplace" ukl:changeStart="true">
                <num>(2)</num>
                <content>
                    <p>This company in old text should be skipped entirely.</p>
                </content>
            </subsection>
            <subsection eId="sec_5__subsec_2_new" ukl:change="insReplace" ukl:changeEnd="true">
                <num>(2)</num>
                <intro>
                    <p>This company in new replacement text should also be skipped.</p>
                </intro>
                <level class="para1">
                    <num>(a)</num>
                    <content>
                        <p>Another company reference in the new text.</p>
                    </content>
                </level>
            </subsection>
            <section eId="sec_6">
                <num>6</num>
                <heading>Another Active Section</heading>
                <p>This company should also be found.</p>
            </section>
        </body>"""
        element = etree.fromstring(xml)
        changes_made = []

        result = self.processor._replace_text_occurrences_iteratively(
            element, "company", "corporation", AmendmentType.SUBSTITUTION, changes_made
        )

        self.assertTrue(result)
        # Should find 2 occurrences (in sections 4 and 6, not in substitution pair)
        total_occurrences = sum(change["occurrences"] for change in changes_made)
        self.assertEqual(total_occurrences, 2)

        # Verify the old (delReplace) subsection wasn't modified
        old_subsection = element.find(".//akn:subsection[@eId='sec_5__subsec_2_old']", self.xml_handler.namespaces)
        old_para = old_subsection.find(".//akn:p", self.xml_handler.namespaces)
        self.assertIn("This company in old text", old_para.text)

        # Verify the new (insReplace) subsection wasn't modified
        new_subsection = element.find(".//akn:subsection[@eId='sec_5__subsec_2_new']", self.xml_handler.namespaces)
        new_paras = new_subsection.findall(".//akn:p", self.xml_handler.namespaces)
        self.assertIn("This company in new replacement", new_paras[0].text)
        self.assertIn("Another company reference", new_paras[1].text)

        # Verify no ins/del elements were created in either substitution part
        old_changes = old_subsection.findall(".//akn:ins", self.xml_handler.namespaces)
        new_changes = new_subsection.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(old_changes), 0)
        self.assertEqual(len(new_changes), 0)

    def test_replace_text_occurrences_iteratively_after_previous_insertion(self):
        """Test applying 'each place' amendment after previous insertions."""
        # Create section 13 subsection 2 intro from bug scenario
        intro = etree.Element("intro")
        intro.set("eId", "sec_13__subsec_2__intro")
        p = etree.SubElement(intro, "p")
        p.text = "The notice must specify, in relation to the hazard (or each of the hazards) to which it relates—"

        # Apply first amendment: after "hazards" insert "or failures"
        changes_made = []
        self.processor._replace_text_occurrences_iteratively(
            p, "hazards", "hazards or failures", AmendmentType.INSERTION, changes_made
        )

        self.assertEqual(len(changes_made), 1)

        # Apply second amendment: after "hazard" insert "or failure"
        changes_made_each_place = []
        self.processor._replace_text_occurrences_iteratively(
            intro, "hazard", "hazard or failure", AmendmentType.INSERTION, changes_made_each_place
        )

        # Should find only standalone "hazard", not within "hazards"
        self.assertEqual(len(changes_made_each_place), 1)

        # Verify final structure
        xml_str = etree.tostring(intro, encoding="unicode")
        self.assertIn("<ns0:span", xml_str)
        self.assertIn("hazard</ns0:span>", xml_str)
        self.assertIn("<ns0:ins", xml_str)
        self.assertIn(" or failure</ns0:ins>", xml_str)

        # Verify text content
        text_content = "".join(p.itertext())
        expected = (
            "The notice must specify, in relation to the hazard or failure "
            "(or each of the hazards or failures) to which it relates—"
        )
        self.assertEqual(text_content, expected)

    def test_replace_text_occurrences_iteratively_with_existing_ins(self):
        """Test replacement when element has existing <ins> element."""
        # Create paragraph with existing amendment structure
        p = etree.Element("p")
        p.text = "The notice must specify, in relation to the hazard (or each of the "

        # Add existing elements
        span = etree.SubElement(p, "span")
        span.text = "hazards"

        ins = etree.SubElement(p, "ins")
        ins.text = " or failures"
        ins.tail = ") to which it relates—"

        # Apply amendment to replace "hazard" with "hazard or failure"
        changes_made = []
        count = self.processor._replace_text_occurrences_iteratively(
            p, "hazard", "hazard or failure", AmendmentType.INSERTION, changes_made
        )

        # Should find 1 occurrence in p.text
        self.assertTrue(count)
        self.assertEqual(len(changes_made), 1)

        # Verify text content
        text_content = "".join(p.itertext())
        expected = (
            "The notice must specify, in relation to the hazard or failure "
            "(or each of the hazards or failures) to which it relates—"
        )
        self.assertEqual(text_content, expected)

        # Verify no text mangling
        self.assertNotIn("relates—hazard", text_content)
        self.assertNotIn(") to which it relates— (or each", text_content)

    def test_replace_text_occurrences_iteratively_word_boundary_with_children(self):
        """Test word boundary matching when element has existing children."""
        # Create paragraph simulating state after first amendment
        p = etree.Element("p")
        p.text = "The notice must specify, in relation to the hazard (or each of the hazards"

        # Add existing <ins> element
        ins = etree.SubElement(p, "ins")
        ins.text = " or failures"
        ins.tail = ") to which it relates—"

        # Apply "hazard -> hazard or failure" amendment
        changes_made = []
        self.processor._replace_text_occurrences_iteratively(
            p, "hazard", "hazard or failure", AmendmentType.INSERTION, changes_made
        )

        # Check final XML structure
        xml_str = etree.tostring(p, encoding="unicode")

        # Should not have loose 's' character
        self.assertNotIn(">s<", xml_str)

        # Should find only standalone "hazard"
        self.assertEqual(len(changes_made), 1)

    # ==================== Tests for _replace_in_element_with_children ====================

    def test_replace_in_element_with_children_no_text(self):
        """Test when element with children has no text."""
        element = etree.Element("p")
        child = etree.SubElement(element, "span")
        child.text = "child text"

        count = self.processor._replace_in_element_with_children(
            element, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 0)

    def test_replace_in_element_with_children_text_not_containing_find(self):
        """Test when element text doesn't contain find_text."""
        element = etree.Element("p")
        element.text = "This text does not contain the search term"
        child = etree.SubElement(element, "span")
        child.text = "child"

        count = self.processor._replace_in_element_with_children(
            element, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 0)

    def test_replace_in_element_with_children_no_word_boundary_match(self):
        """Test when pattern doesn't match with word boundaries."""
        element = etree.Element("p")
        element.text = "The personal information"
        child = etree.SubElement(element, "span")
        child.text = "child"

        count = self.processor._replace_in_element_with_children(
            element, "person", "individual", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 0)
        self.assertEqual(element.text, "The personal information")

    # ==================== Tests for _replace_occurrences_in_text_node ====================

    def test_replace_occurrences_in_text_node_single(self):
        """Test replacing single occurrence in text node."""
        element = etree.Element("p")
        element.text = "The company is here."

        count = self.processor._replace_occurrences_in_text_node(
            element, "text", "The company is here.", "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 1)
        # After processing, element.text should have the text before first occurrence
        self.assertEqual(element.text, "The ")  # Text before "company"
        # Should have del and ins elements as children
        self.assertEqual(len(element), 2)  # del and ins elements

    def test_replace_occurrences_in_text_node_multiple(self):
        """Test replacing multiple occurrences in text node."""
        element = etree.Element("p")
        element.text = "company and company again"

        count = self.processor._replace_occurrences_in_text_node(
            element, "text", "company and company again", "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 2)
        self.assertEqual(len(element), 4)  # 2 pairs of del/ins

    def test_replace_occurrences_in_text_node_tail(self):
        """Test replacing in tail text via _replace_occurrences_in_text_node."""
        parent = etree.Element("div")
        child = etree.SubElement(parent, "span")
        child.tail = "The company is here."

        original_len = len(parent)

        # Process tail text
        count = self.processor._replace_occurrences_in_text_node(
            child, "tail", "The company is here.", "company", "corporation", AmendmentType.SUBSTITUTION
        )

        # Verify results
        self.assertEqual(count, 1)
        self.assertIsNone(child.tail)
        self.assertGreater(len(parent), original_len)

    def test_replace_occurrences_in_text_node_word_boundaries(self):
        """Test that word boundaries prevent substring matching."""
        element = etree.Element("p")

        # Test case from observed issue - "on" should not match in "person" or "control"
        text = "(in the case of a dwelling) on the person having control"

        count = self.processor._replace_occurrences_in_text_node(
            element, "text", text, "on", "", AmendmentType.DELETION
        )

        # Should find 1 occurrence of standalone "on", not 2 (not matching in "person" or "control")
        self.assertEqual(count, 1)

        # Verify the correct "on" was replaced
        # The deletion element should contain just "on"
        del_elements = [child for child in element if child.tag.endswith("del")]
        self.assertEqual(len(del_elements), 1)
        self.assertEqual(del_elements[0].text, "on")

    def test_replace_occurrences_in_text_node_pattern_not_found(self):
        """Test when pattern is not found in the text (covers line 939)."""
        element = etree.Element("p")

        # Text that doesn't contain the pattern
        text = "This text does not contain the search term"

        count = self.processor._replace_occurrences_in_text_node(
            element, "text", text, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        # Should return 0 when pattern not found
        self.assertEqual(count, 0)
        # Element should remain unchanged
        self.assertIsNone(element.text)
        self.assertEqual(len(element), 0)

    def test_replace_occurrences_in_text_node_complex_word_boundaries(self):
        """Test various word boundary scenarios."""
        test_cases = [
            # (text, find, expected_count, description)
            ("for failures or failure", "or failure", 1, "should not match in 'for failures'"),
            ("control of the HMO", "on", 0, "should not match in 'control'"),
            ("person on the list", "on", 1, "should match standalone 'on'"),
            ("on the person", "on", 1, "should match at start"),
            ("the person on", "on", 1, "should match at end"),
            ("Don't match on", "on", 1, "should match after apostrophe"),
            ("(on)", "on", 1, "should match in parentheses"),
            ("on,on.on;on", "on", 4, "should match with punctuation"),
        ]

        for text, find_text, expected, description in test_cases:
            element = etree.Element("p")

            count = self.processor._replace_occurrences_in_text_node(
                element, "text", text, find_text, "REPLACED", AmendmentType.SUBSTITUTION
            )

            self.assertEqual(count, expected, f"Failed: {description}")

            # If matches found, verify they're in the right places
            if expected > 0:
                # Count del and ins elements (2 per substitution)
                change_elements = [child for child in element if child.tag.endswith(("del", "ins"))]
                self.assertEqual(len(change_elements), expected * 2)

    # ==================== Tests for _replace_occurrences_in_tail ====================

    def test_replace_occurrences_in_tail_simple(self):
        """Test simple tail replacement."""
        parent = etree.Element("div")
        child = etree.SubElement(parent, "span")
        parts = ["The ", " is here."]

        count = self.processor._replace_occurrences_in_tail(
            child, parts, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        # Verify tail cleared and elements added
        self.assertEqual(count, 1)
        self.assertIsNone(child.tail)
        self.assertGreater(len(parent), 1)

        # Verify del/ins elements created
        self.assertTrue(any(elem.tag.endswith("del") for elem in parent))
        self.assertTrue(any(elem.tag.endswith("ins") for elem in parent))

    def test_replace_occurrences_in_tail_no_parent(self):
        """Test tail replacement when element has no parent."""
        element = etree.Element("p")
        parts = ["The ", " is here."]

        with patch("app.services.amendment_processor.logger") as mock_logger:
            count = self.processor._replace_occurrences_in_tail(
                element, parts, "company", "corporation", AmendmentType.SUBSTITUTION
            )

            self.assertEqual(count, 0)
            mock_logger.warning.assert_called_once_with("Cannot process tail text without parent")

    def test_replace_occurrences_in_tail_multiple(self):
        """Test multiple occurrences in tail."""
        parent = etree.Element("div")
        child = etree.SubElement(parent, "span")
        parts = ["Start ", " middle ", " end"]  # Two occurrences

        count = self.processor._replace_occurrences_in_tail(
            child, parts, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 2)

    def test_replace_occurrences_in_tail_with_empty_parts(self):
        """Test tail replacement with empty parts."""
        parent = etree.Element("div")
        child = etree.SubElement(parent, "span")
        parts = ["", " after"]  # Empty first part

        count = self.processor._replace_occurrences_in_tail(
            child, parts, "company", "corporation", AmendmentType.DELETION
        )

        self.assertEqual(count, 1)

    # ==================== Tests for _replace_occurrences_in_element_text ====================

    def test_replace_occurrences_in_element_text_simple(self):
        """Test simple element text replacement."""
        element = etree.Element("p")
        parts = ["The ", " is here."]

        count = self.processor._replace_occurrences_in_element_text(
            element, parts, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 1)
        self.assertEqual(element.text, "The ")  # First part becomes text
        self.assertEqual(len(element), 2)  # del and ins elements

    def test_replace_occurrences_in_element_text_multiple(self):
        """Test multiple occurrences in element text."""
        element = etree.Element("p")
        parts = ["Start ", " middle ", " end"]

        count = self.processor._replace_occurrences_in_element_text(
            element, parts, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 2)
        self.assertEqual(element.text, "Start ")
        self.assertEqual(len(element), 4)  # 2 pairs of del/ins

    def test_replace_occurrences_in_element_text_empty_first_part(self):
        """Test when first part is empty."""
        element = etree.Element("p")
        parts = ["", " after"]

        count = self.processor._replace_occurrences_in_element_text(
            element, parts, "company", "corporation", AmendmentType.INSERTION
        )

        self.assertEqual(count, 1)
        self.assertIsNone(element.text)  # No text when first part is empty

    def test_replace_occurrences_in_element_text_with_children(self):
        """Test replacement when element has existing children."""
        element = etree.Element("p")
        element.text = "Before company after"

        existing_child = etree.SubElement(element, "span")
        existing_child.text = "existing"
        existing_child.tail = " more text"
        parts = element.text.split("company")

        count = self.processor._replace_occurrences_in_element_text(
            element, parts, "company", "corporation", AmendmentType.DELETION
        )

        self.assertEqual(count, 1)

        # Verify the structure is correct
        self.assertEqual(element.text, "Before ")

        # Check there are at least 2 children (del element + original span)
        self.assertGreaterEqual(len(element), 2)

        # The del element should be first
        del_elem = element[0]
        self.assertTrue(del_elem.tag.endswith("del"))
        self.assertEqual(del_elem.text, "company")

        # Find the original span element
        span_found = False
        for child in element:
            if child.tag.endswith("span") and child.text == "existing":
                span_found = True
                self.assertEqual(child.text, "existing")
                self.assertEqual(child.tail, " more text")
                break

        self.assertTrue(span_found, "Original span element not found")

    def test_replace_occurrences_in_element_text_word_boundaries_with_children(self):
        """Test word boundary matching when element has existing children."""
        element = etree.Element("p")
        element.text = "The hazard (or each of the hazards"

        # Add existing child from previous amendment
        ins = etree.SubElement(element, "ins")
        ins.text = " or failures"
        ins.tail = ") exists"

        # Match hazard parttern
        import re

        escaped_find = re.escape("hazard")
        pattern = rf"(?<!\w){escaped_find}(?!\w)"
        parts = re.split(pattern, element.text)

        count = self.processor._replace_occurrences_in_element_text(
            element, parts, "hazard", "hazard or failure", AmendmentType.INSERTION
        )

        # Should find only 1 occurrence (not in "hazards")
        self.assertEqual(count, 1)

        # Verify "hazards" wasn't split
        full_text = "".join(element.itertext())
        self.assertIn("hazards or failures", full_text)
        self.assertNotIn(">s<", etree.tostring(element, encoding="unicode"))

    def test_replace_occurrences_in_element_text_no_occurrences(self):
        """Test when parts has only one element (no occurrences found)."""
        element = etree.Element("p")
        parts = ["No occurrences here"]

        count = self.processor._replace_occurrences_in_element_text(
            element, parts, "company", "corporation", AmendmentType.SUBSTITUTION
        )

        self.assertEqual(count, 0)
        self.assertEqual(len(element), 0)

    def test_replace_occurrences_in_element_text_single_occurrence_with_final_text(self):
        """Test when single occurrence leaves final text with no children."""
        element = etree.Element("p")
        parts = ["", "final text"]

        # Mock to create no elements
        with patch.object(self.processor, "_create_tracked_change_elements", return_value=0):
            self.processor._replace_occurrences_in_element_text(
                element, parts, "company", "corporation", AmendmentType.SUBSTITUTION
            )

            self.assertEqual(element.text, "final text")
            self.assertEqual(len(element), 0)

    # ==================== Tests for _insert_text_content ====================

    def test_insert_text_content_add_to_tail(self):
        """Test inserting text as tail of previous element."""
        parent = etree.Element("div")
        child1 = etree.SubElement(parent, "span")
        child1.text = "child1"
        child2 = etree.SubElement(parent, "span")
        child2.text = "child2"

        new_pos = self.processor._insert_text_content(parent, 1, "inserted text")

        self.assertEqual(new_pos, 1)  # Position unchanged
        self.assertEqual(child1.tail, "inserted text")

    def test_insert_text_content_create_span(self):
        """Test creating span when position is at start."""
        parent = etree.Element("div")

        # Insert at beginning of empty parent
        new_pos = self.processor._insert_text_content(parent, 0, "inserted text")

        # Verify span creation
        self.assertEqual(new_pos, 1)
        self.assertEqual(len(parent), 1)
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}span")
        self.assertEqual(parent[0].text, "inserted text")

    def test_insert_text_content_existing_tail(self):
        """Test when previous element already has tail."""
        parent = etree.Element("div")
        child = etree.SubElement(parent, "span")
        child.tail = "existing tail"

        new_pos = self.processor._insert_text_content(parent, 1, "new text")

        self.assertEqual(new_pos, 2)
        self.assertEqual(len(parent), 2)  # New span created

    # ==================== Tests for _add_tail_text ====================

    def test_add_tail_text_valid_index(self):
        """Test adding tail text at valid index."""
        parent = etree.Element("div")
        child = etree.SubElement(parent, "span")

        self.processor._add_tail_text(parent, 0, "tail text")

        self.assertEqual(child.tail, "tail text")

    def test_add_tail_text_invalid_index(self):
        """Test adding tail text at invalid index."""
        parent = etree.Element("div")

        # Should not raise exception
        self.processor._add_tail_text(parent, 5, "tail text")

        # No elements to add tail to
        self.assertEqual(len(parent), 0)

    # ==================== Tests for _append_text_to_last_child ====================

    def test_append_text_to_last_child_with_children(self):
        """Test appending text when element has children."""
        element = etree.Element("p")
        child = etree.SubElement(element, "span")
        child.tail = "existing "

        self.processor._append_text_to_last_child(element, "appended")

        self.assertEqual(child.tail, "existing appended")

    def test_append_text_to_last_child_no_children(self):
        """Test appending text when element has no children."""
        element = etree.Element("p")
        element.text = "existing "

        self.processor._append_text_to_last_child(element, "appended")

        self.assertEqual(element.text, "existing appended")

    def test_append_text_to_last_child_null_tail(self):
        """Test appending when last child has no tail."""
        element = etree.Element("p")
        child = etree.SubElement(element, "span")

        self.processor._append_text_to_last_child(element, "new text")

        self.assertEqual(child.tail, "new text")

    # ==================== Tests for _create_tracked_change_elements ====================

    def test_create_tracked_change_elements_deletion(self):
        """Test creating deletion change elements."""
        parent = etree.Element("div")

        count = self.processor._create_tracked_change_elements(parent, 0, "old text", "", AmendmentType.DELETION)

        self.assertEqual(count, 1)
        self.assertEqual(len(parent), 1)
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}del")
        self.assertEqual(parent[0].text, "old text")

    def test_create_tracked_change_elements_substitution(self):
        """Test creating substitution change elements."""
        parent = etree.Element("div")

        count = self.processor._create_tracked_change_elements(parent, 0, "old", "new", AmendmentType.SUBSTITUTION)

        self.assertEqual(count, 2)
        self.assertEqual(len(parent), 2)
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}del")
        self.assertEqual(parent[0].text, "old")
        self.assertEqual(parent[1].tag, f"{{{XMLHandler.AKN_URI}}}ins")
        self.assertEqual(parent[1].text, "new")

    def test_create_tracked_change_elements_insertion(self):
        """Test creating insertion change elements."""
        parent = etree.Element("div")

        count = self.processor._create_tracked_change_elements(parent, 0, "text", "text added", AmendmentType.INSERTION)

        self.assertEqual(count, 2)  # span + ins

    def test_create_tracked_change_elements_unexpected_type(self):
        """Test handling unexpected amendment type."""
        parent = etree.Element("div")

        # Create a mock amendment type that's not in the enum
        mock_type = Mock()
        mock_type.name = "UNEXPECTED"

        with patch("app.services.amendment_processor.logger") as mock_logger:
            count = self.processor._create_tracked_change_elements(parent, 0, "text", "new", mock_type)

            self.assertEqual(count, 0)
            mock_logger.error.assert_called_once()
            self.assertIn("Unexpected amendment type", mock_logger.error.call_args[0][0])

    # ==================== Tests for _create_deletion_change ====================

    def test_create_deletion_change(self):
        """Test creating a deletion element."""
        parent = etree.Element("div")

        count = self.processor._create_deletion_change(parent, 0, "deleted text")

        self.assertEqual(count, 1)
        self.assertEqual(len(parent), 1)
        del_elem = parent[0]
        self.assertEqual(del_elem.tag, f"{{{XMLHandler.AKN_URI}}}del")
        self.assertEqual(del_elem.text, "deleted text")
        self.assertEqual(del_elem.get(f"{{{XMLHandler.UKL_URI}}}changeStart"), "true")
        self.assertEqual(del_elem.get(f"{{{XMLHandler.UKL_URI}}}changeEnd"), "true")

    # ==================== Tests for _create_substitution_change ====================

    def test_create_substitution_change(self):
        """Test creating substitution elements."""
        parent = etree.Element("div")

        count = self.processor._create_substitution_change(parent, 0, "old text", "new text")

        self.assertEqual(count, 2)
        self.assertEqual(len(parent), 2)

        # Check del element
        del_elem = parent[0]
        self.assertEqual(del_elem.tag, f"{{{XMLHandler.AKN_URI}}}del")
        self.assertEqual(del_elem.text, "old text")
        self.assertEqual(del_elem.get(f"{{{XMLHandler.UKL_URI}}}changeStart"), "true")
        self.assertIsNone(del_elem.get(f"{{{XMLHandler.UKL_URI}}}changeEnd"))

        # Check ins element
        ins_elem = parent[1]
        self.assertEqual(ins_elem.tag, f"{{{XMLHandler.AKN_URI}}}ins")
        self.assertEqual(ins_elem.text, "new text")
        self.assertIsNone(ins_elem.get(f"{{{XMLHandler.UKL_URI}}}changeStart"))
        self.assertEqual(ins_elem.get(f"{{{XMLHandler.UKL_URI}}}changeEnd"), "true")

    # ==================== Tests for _create_insertion_change ====================

    def test_create_insertion_change_after(self):
        """Test creating insertion after reference text."""
        parent = etree.Element("div")

        count = self.processor._create_insertion_change(parent, 0, "reference", "reference and more")

        self.assertEqual(count, 2)
        # First element should be span with reference text
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}span")
        self.assertEqual(parent[0].text, "reference")
        # Second should be ins with added text
        self.assertEqual(parent[1].tag, f"{{{XMLHandler.AKN_URI}}}ins")
        self.assertEqual(parent[1].text, " and more")

    def test_create_insertion_change_before(self):
        """Test creating insertion before reference text."""
        parent = etree.Element("div")

        count = self.processor._create_insertion_change(parent, 0, "reference", "prefix reference")

        self.assertEqual(count, 2)
        # First element should be ins with prefix
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}ins")
        self.assertEqual(parent[0].text, "prefix ")
        # Second should be span with reference
        self.assertEqual(parent[1].tag, f"{{{XMLHandler.AKN_URI}}}span")
        self.assertEqual(parent[1].text, "reference")

    def test_create_insertion_change_complex(self):
        """Test complex insertion (neither before nor after)."""
        parent = etree.Element("div")

        count = self.processor._create_insertion_change(parent, 0, "find", "completely different")

        self.assertEqual(count, 1)
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}ins")
        self.assertEqual(parent[0].text, "completely different")

    # ==================== Tests for _create_insert_after_elements ====================

    def test_create_insert_after_elements(self):
        """Test creating elements for insertion after reference."""
        parent = etree.Element("div")

        count = self.processor._create_insert_after_elements(parent, 0, "text", "text with addition")

        self.assertEqual(count, 2)
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}span")
        self.assertEqual(parent[0].text, "text")
        self.assertEqual(parent[1].tag, f"{{{XMLHandler.AKN_URI}}}ins")
        self.assertEqual(parent[1].text, " with addition")

    # ==================== Tests for _create_insert_before_elements ====================

    def test_create_insert_before_elements(self):
        """Test creating elements for insertion before reference."""
        parent = etree.Element("div")

        count = self.processor._create_insert_before_elements(parent, 0, "text", "prefix before text")

        self.assertEqual(count, 2)
        self.assertEqual(parent[0].tag, f"{{{XMLHandler.AKN_URI}}}ins")
        self.assertEqual(parent[0].text, "prefix before ")
        self.assertEqual(parent[1].tag, f"{{{XMLHandler.AKN_URI}}}span")
        self.assertEqual(parent[1].text, "text")

    # ==================== Integration Tests ====================

    def test_each_place_deletion_integration(self):
        """Integration test for deletion across multiple elements."""
        working_tree = self.create_test_xml("""
            <section eId="sec_1">
                <subsection eId="sec_1__subsec_1">
                    <content>
                        <p>Remove this <b>word</b> and this word too.</p>
                    </content>
                </subsection>
            </section>
        """)

        amending_tree = self.create_test_xml("<section eId='sec_25'><p>Amendment</p></section>")

        amendment = Amendment(
            source_eid="sec_25",
            source="s. 25",
            amendment_type=AmendmentType.DELETION,
            whole_provision=False,
            location=AmendmentLocation.EACH_PLACE,
            affected_document="Test Act",
            affected_provision="sec_1__subsec_1",
            amendment_id="test-del-001",
        )

        pattern_data = {"find_text": "word", "replace_text": ""}

        success, error = self.processor.apply_each_place_amendment(
            amendment, working_tree, amending_tree, pattern_data, "schedule-001"
        )

        self.assertTrue(success)

        # Verify both occurrences were deleted
        subsection = working_tree.find(".//akn:subsection[@eId='sec_1__subsec_1']", self.xml_handler.namespaces)
        del_elements = subsection.findall(".//akn:del", self.xml_handler.namespaces)
        self.assertEqual(len(del_elements), 2)
        for del_elem in del_elements:
            self.assertEqual(del_elem.text, "word")

    def test_each_place_insertion_integration(self):
        """Integration test for insertion pattern."""
        working_tree = self.create_test_xml("""
            <section eId="sec_1">
                <content>
                    <p>Insert after this text and after this text too.</p>
                </content>
            </section>
        """)

        amending_tree = self.create_test_xml("<section eId='sec_25'><p>Amendment</p></section>")

        amendment = Amendment(
            source_eid="sec_25",
            source="s. 25",
            amendment_type=AmendmentType.INSERTION,
            whole_provision=False,
            location=AmendmentLocation.EACH_PLACE,
            affected_document="Test Act",
            affected_provision="sec_1",
            amendment_id="test-ins-001",
        )

        pattern_data = {"find_text": "text", "replace_text": "text [new]"}

        success, error = self.processor.apply_each_place_amendment(
            amendment, working_tree, amending_tree, pattern_data, "schedule-001"
        )

        self.assertTrue(success)

        # Verify insertions were made
        section = working_tree.find(".//akn:section[@eId='sec_1']", self.xml_handler.namespaces)
        ins_elements = section.findall(".//akn:ins", self.xml_handler.namespaces)
        self.assertEqual(len(ins_elements), 2)
        for ins_elem in ins_elements:
            self.assertEqual(ins_elem.text, " [new]")
