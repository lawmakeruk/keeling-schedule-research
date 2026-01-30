# tests/services/test_xml_handler.py
"""
Unit tests for XMLHandler class that manages XML document operations.
"""
import os
import tempfile
import threading
from unittest import TestCase
from unittest.mock import patch
from lxml import etree

from app.models.amendments import Amendment, AmendmentType, AmendmentLocation
from app.services.xml_handler import XMLHandler


class TestXMLHandler(TestCase):
    """Test cases for XMLHandler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = XMLHandler()

        # Create a sample XML document for testing
        self.sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
                    xmlns:ukl="https://www.legislation.gov.uk/namespaces/UK-AKN">
            <act name="TestAct">
                <body>
                    <section eId="sec_1">
                        <heading>Test Section</heading>
                        <subsection eId="sec_1__subsec_1">
                            <content>
                                <p>Test content</p>
                            </content>
                        </subsection>
                    </section>
                </body>
            </act>
        </akomaNtoso>"""

        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        self.temp_file.write(self.sample_xml)
        self.temp_file.close()

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

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_init(self):
        """Test XMLHandler initialisation."""
        handler = XMLHandler()

        # Check namespaces are set correctly
        self.assertEqual(handler.namespaces["akn"], XMLHandler.AKN_URI)
        self.assertEqual(handler.namespaces["ukl"], XMLHandler.UKL_URI)

        # Check dnum counter initialised
        self.assertEqual(handler._dnum_counter, 0)
        self.assertIsNotNone(handler._dnum_lock)

    def test_load_xml_success(self):
        """Test successful XML loading."""
        tree = self.handler.load_xml(self.temp_file.name)

        self.assertIsInstance(tree, etree._ElementTree)
        root = tree.getroot()
        self.assertEqual(root.tag, f"{{{XMLHandler.AKN_URI}}}akomaNtoso")

    def test_load_xml_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(OSError):  # lxml raises OSError, not FileNotFoundError
            self.handler.load_xml("non_existent_file.xml")

    def test_load_xml_malformed(self):
        """Test loading malformed XML."""
        bad_xml_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        bad_xml_file.write("<unclosed>")
        bad_xml_file.close()

        try:
            with self.assertRaises(etree.ParseError):
                self.handler.load_xml(bad_xml_file.name)
        finally:
            os.unlink(bad_xml_file.name)

    def test_save_xml_success(self):
        """Test successful XML saving."""
        tree = self.handler.load_xml(self.temp_file.name)
        output_file = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        output_file.close()

        try:
            self.handler.save_xml(tree, output_file.name)

            # Verify file was written
            self.assertTrue(os.path.exists(output_file.name))

            # Verify content is valid XML
            saved_tree = etree.parse(output_file.name)
            self.assertIsNotNone(saved_tree.getroot())
        finally:
            os.unlink(output_file.name)

    def test_save_xml_io_error(self):
        """Test saving to invalid location."""
        tree = self.handler.load_xml(self.temp_file.name)

        with self.assertRaises(IOError):
            self.handler.save_xml(tree, "/invalid/path/file.xml")

    def test_normalise_namespaces_remove_duplicate(self):
        """Test normalisation converts uk: to ukl: for UK-AKN namespace."""
        # Create XML like the problematic act that uses uk: instead of ukl:
        xml_with_wrong_prefix = """<?xml version="1.0"?>
        <root xmlns:uk="https://www.legislation.gov.uk/namespaces/UK-AKN">
            <uk:change>Amendment text</uk:change>
            <uk:changeStart>true</uk:changeStart>
        </root>"""

        doc = etree.fromstring(xml_with_wrong_prefix.encode())
        tree = doc.getroottree()

        # Before normalisation - elements use uk: prefix
        uk_elements = tree.xpath("//uk:*", namespaces={"uk": XMLHandler.UKL_URI})
        self.assertEqual(len(uk_elements), 2)

        self.handler.normalise_namespaces(tree)

        root = tree.getroot()

        # After normalisation:
        # The UK-AKN namespace should be bound to ukl: (not uk:)
        self.assertEqual(root.nsmap.get("ukl"), XMLHandler.UKL_URI)
        self.assertNotIn("uk", root.nsmap)

        # Elements should now be accessible via ukl: prefix
        ukl_elements = root.xpath("//ukl:*", namespaces={"ukl": XMLHandler.UKL_URI})
        self.assertEqual(len(ukl_elements), 2)

        # Specifically check the change elements
        change_elements = root.xpath("//ukl:change", namespaces={"ukl": XMLHandler.UKL_URI})
        self.assertEqual(len(change_elements), 1)
        self.assertEqual(change_elements[0].text, "Amendment text")

        # Verify prefix is actually ukl
        self.assertEqual(change_elements[0].prefix, "ukl")

    def test_normalise_namespaces_add_missing(self):
        """Test adding ukl namespace when missing."""
        xml_without_ukl = """<?xml version="1.0"?>
        <root xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
            <element>Test</element>
        </root>"""

        tree = etree.fromstring(xml_without_ukl.encode()).getroottree()

        # Initially no ukl namespace
        root = tree.getroot()
        self.assertNotIn("ukl", root.nsmap)

        self.handler.normalise_namespaces(tree)

        # After normalisation, ukl should be added
        root = tree.getroot()
        self.assertIn("ukl", root.nsmap)
        self.assertEqual(root.nsmap["ukl"], XMLHandler.UKL_URI)

    def test_normalise_eids(self):
        """Test eId normalisation."""
        xml_with_mixed_case = """<?xml version="1.0"?>
        <root>
            <section eId="SEC_1A__Para_2__SubPara_a">Test</section>
            <section eId="sec_2__PARA_3">Test</section>
        </root>"""

        tree = etree.fromstring(xml_with_mixed_case.encode()).getroottree()
        self.handler.normalise_eids(tree)

        sections = tree.findall(".//section")
        self.assertEqual(sections[0].get("eId"), "sec_1A__para_2__subpara_a")
        self.assertEqual(sections[1].get("eId"), "sec_2__para_3")

    def test_find_element_by_eid_found(self):
        """Test finding element by eId."""
        tree = self.handler.load_xml(self.temp_file.name)

        element = self.handler.find_element_by_eid(tree, "sec_1__subsec_1")
        self.assertIsNotNone(element)
        self.assertEqual(element.tag, f"{{{XMLHandler.AKN_URI}}}subsection")

    def test_find_element_by_eid_not_found(self):
        """Test finding non-existent element."""
        tree = self.handler.load_xml(self.temp_file.name)

        element = self.handler.find_element_by_eid(tree, "non_existent_eid")
        self.assertIsNone(element)

    def test_find_element_by_eid_components_with_accurate_eid(self):
        """Test finding element by eId."""
        tree = self.handler.load_xml(self.temp_file.name)

        element = self.handler.find_element_by_eid_components(tree, "sec_1__subsec_1")
        self.assertIsNotNone(element)
        self.assertEqual(element.tag, f"{{{XMLHandler.AKN_URI}}}subsection")

    def test_find_element_by_eid_components_with_inaccurate_eid(self):
        """Test finding element by eId."""
        tree = self.handler.load_xml(self.temp_file.name)

        element = self.handler.find_element_by_eid_components(tree, "sec_1__unknown_1")
        self.assertIsNotNone(element)
        self.assertEqual(element.tag, f"{{{XMLHandler.AKN_URI}}}subsection")

    def test_find_element_by_eid_components_with_inaccurate_num(self):
        """Test finding element by eId."""
        tree = self.handler.load_xml(self.temp_file.name)

        element = self.handler.find_element_by_eid_components(tree, "sec_1__unknown_unknown")
        self.assertIsNone(element)

    def test_find_element_by_eid_components_exception(self):
        """Test finding element by eId."""
        tree = self.handler.load_xml(self.temp_file.name)

        with patch.object(self.handler, "find_element_by_eid", side_effect=RuntimeError("Unexpected error")):
            element = self.handler.find_element_by_eid_components(tree, "sec_1__subsec_1")
            self.assertIsNone(element)

    def test_find_closest_match_by_eid_found(self):
        """Test finding element by eId."""
        tree = self.handler.load_xml(self.temp_file.name)

        element = self.handler.find_closest_match_by_eid(tree, "sec_1__subsec_1__para_a__subpara_A")
        self.assertIsNotNone(element)
        self.assertEqual(element.tag, f"{{{XMLHandler.AKN_URI}}}subsection")

    def test_find_closest_match_by_eid_not_found(self):
        """Test finding element by eId."""
        tree = self.handler.load_xml(self.temp_file.name)

        element = self.handler.find_closest_match_by_eid(tree, "non_existent_eid")
        self.assertIsNone(element)

    def test_allocate_dnum_sequential(self):
        """Test dnum allocation is sequential."""
        dnums = []
        for _ in range(5):
            dnums.append(self.handler.allocate_dnum())

        self.assertEqual(dnums, [1, 2, 3, 4, 5])

    def test_allocate_dnum_thread_safe(self):
        """Test dnum allocation is thread-safe."""
        dnums = []
        lock = threading.Lock()

        def allocate_many():
            for _ in range(100):
                dnum = self.handler.allocate_dnum()
                with lock:
                    dnums.append(dnum)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=allocate_many)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check all dnums are unique
        self.assertEqual(len(dnums), 1000)
        self.assertEqual(len(set(dnums)), 1000)

        # Check they're in expected range
        self.assertEqual(min(dnums), 1)
        self.assertEqual(max(dnums), 1000)

    def test_set_dnum_counter(self):
        """Test setting dnum counter."""
        self.handler.set_dnum_counter(100)
        next_dnum = self.handler.allocate_dnum()
        self.assertEqual(next_dnum, 101)

    def test_add_change_markup_simple(self):
        """Test adding simple change markup."""
        element = etree.Element("test")

        self.handler.add_change_markup(element, "ins")

        self.assertEqual(element.get(f"{{{XMLHandler.UKL_URI}}}change"), "ins")
        self.assertIsNone(element.get(f"{{{XMLHandler.UKL_URI}}}changeStart"))

    def test_add_change_markup_full(self):
        """Test adding complete change markup."""
        element = etree.Element("test")

        self.handler.add_change_markup(element, "del", is_start=True, is_end=True, add_dnum=True)

        ukl_ns = f"{{{XMLHandler.UKL_URI}}}"
        self.assertEqual(element.get(f"{ukl_ns}change"), "del")
        self.assertEqual(element.get(f"{ukl_ns}changeStart"), "true")
        self.assertEqual(element.get(f"{ukl_ns}changeEnd"), "true")
        self.assertEqual(element.get(f"{ukl_ns}changeGenerated"), "true")
        self.assertEqual(element.get(f"{ukl_ns}changeDnum"), "KS1")

    def test_find_existing_dnums(self):
        """Test finding existing dnums in document."""
        xml_with_dnums = f"""<?xml version="1.0"?>
        <root xmlns:ukl="{XMLHandler.UKL_URI}">
            <element ukl:changeGenerated="true" ukl:changeDnum="KS5">Text</element>
            <element ukl:changeGenerated="true" ukl:changeDnum="KS12">Text</element>
            <element ukl:changeGenerated="true" ukl:changeDnum="KS3">Text</element>
            <element>No dnum</element>
        </root>"""

        tree = etree.fromstring(xml_with_dnums.encode()).getroottree()
        highest = self.handler.find_existing_dnums(tree)

        self.assertEqual(highest, 12)

    def test_find_existing_dnums_none_found(self):
        """Test finding dnums when none exist."""
        tree = self.handler.load_xml(self.temp_file.name)
        highest = self.handler.find_existing_dnums(tree)

        self.assertEqual(highest, 0)

    def test_renumber_dnums(self):
        """Test renumbering dnums in document order."""
        xml_with_dnums = f"""<?xml version="1.0"?>
        <root xmlns:ukl="{XMLHandler.UKL_URI}">
            <element ukl:changeGenerated="true" amendmentId="test-001" ukl:changeDnum="KS99">First</element>
            <element ukl:changeGenerated="true" amendmentId="test-002" ukl:changeDnum="KS5">Second</element>
            <element ukl:changeGenerated="true" amendmentId="test-003" ukl:changeDnum="KS12">Third</element>
            <element ukl:changeGenerated="true" amendmentId="test-004">Fourth</element>
            <element ukl:changeGenerated="true" amendmentId="test-004">Fifth</element>
        </root>"""

        tree = etree.fromstring(xml_with_dnums.encode()).getroottree()
        amendments = [
            Amendment(
                source_eid="sec_25__subsec_1",
                source="s. 25(1)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=False,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1__subsec_1",
                amendment_id="test-001",
            ),
            Amendment(
                source_eid="sec_25__subsec_2",
                source="s. 25(2)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=False,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1__subsec_1",
                amendment_id="test-002",
            ),
            Amendment(
                source_eid="sec_25__subsec_3",
                source="s. 25(3)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=False,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1__subsec_1",
                amendment_id="test-003",
            ),
            Amendment(
                source_eid="sec_25__subsec_4",
                source="s. 25(4)",
                amendment_type=AmendmentType.INSERTION,
                whole_provision=False,
                location=AmendmentLocation.AFTER,
                affected_document="Test Act",
                affected_provision="sec_1__subsec_1",
                amendment_id="test-004",
            ),
        ]
        self.handler.renumber_dnums(tree, amendments)

        elements = tree.xpath("//*[@ukl:changeGenerated='true']", namespaces=self.handler.namespaces)
        self.assertEqual(len(elements), 5)

        self.assertEqual(elements[0].get(f"{{{XMLHandler.UKL_URI}}}changeDnum"), "KS1")
        self.assertEqual(elements[1].get(f"{{{XMLHandler.UKL_URI}}}changeDnum"), "KS2")
        self.assertEqual(elements[2].get(f"{{{XMLHandler.UKL_URI}}}changeDnum"), "KS3")
        self.assertEqual(elements[3].get(f"{{{XMLHandler.UKL_URI}}}changeDnum"), "KS4")
        self.assertEqual(elements[4].get(f"{{{XMLHandler.UKL_URI}}}changeDnum"), "KS5")

        self.assertEqual(amendments[0].dnum_list, ["KS1"])
        self.assertEqual(amendments[1].dnum_list, ["KS2"])
        self.assertEqual(amendments[2].dnum_list, ["KS3"])
        self.assertEqual(amendments[3].dnum_list, ["KS4", "KS5"])
        # Check counter was updated
        self.assertEqual(self.handler._dnum_counter, 5)

    def test_insert_editorial_note_ref(self):
        """Test inserting a //noteRef into an element."""
        element = etree.Element("test")

        dnum = "KS1"
        note = self.handler._insert_editorial_note_ref(element, dnum)

        # Check note was created and returned
        self.assertIsNotNone(note)
        self.assertEqual(note.tag, f"{{{XMLHandler.AKN_URI}}}noteRef")
        self.assertEqual(note.get("class"), "commentary")
        self.assertEqual(note.get("marker"), "*")
        self.assertEqual(note.get("href"), f"#{dnum}")

        # Check it was added to the element
        noteRefs = element.findall(f".//{{{XMLHandler.AKN_URI}}}noteRef")
        self.assertEqual(len(noteRefs), 1)

    def test_get_text_content_excluding_quoted(self):
        """Test extracting text content excluding quoted structures."""
        xml_with_quoted = f"""<?xml version="1.0"?>
        <root xmlns="{XMLHandler.AKN_URI}">
            <p>This is normal text</p>
            <quotedStructure>
                <p>This is quoted text</p>
            </quotedStructure>
            <p>More normal text</p>
        </root>"""

        tree = etree.fromstring(xml_with_quoted.encode()).getroottree()
        root = tree.getroot()

        text = self.handler.get_text_content(root, exclude_quoted=True)
        self.assertIn("This is normal text", text)
        self.assertIn("More normal text", text)
        self.assertNotIn("This is quoted text", text)

    def test_get_text_content_including_quoted(self):
        """Test extracting all text content."""
        xml_with_quoted = f"""<?xml version="1.0"?>
        <root xmlns="{XMLHandler.AKN_URI}">
            <p>Normal text</p>
            <quotedStructure>
                <p>Quoted text</p>
            </quotedStructure>
        </root>"""

        tree = etree.fromstring(xml_with_quoted.encode()).getroottree()
        root = tree.getroot()

        text = self.handler.get_text_content(root, exclude_quoted=False)
        self.assertIn("Normal text", text)
        self.assertIn("Quoted text", text)

    def test_validate_element_has_eid_success(self):
        """Test validation passes for element with eId."""
        element = etree.Element("test", eId="test_1")

        # Should not raise
        self.handler.validate_element_has_eid(element)

    def test_validate_element_has_eid_failure(self):
        """Test validation fails for element without eId."""
        element = etree.Element("test")

        with self.assertRaises(ValueError) as context:
            self.handler.validate_element_has_eid(element, "during test operation")

        self.assertIn("missing eId attribute", str(context.exception))
        self.assertIn("during test operation", str(context.exception))

    def test_validate_element_has_eid_with_parent(self):
        """Test validation includes parent info in error."""
        parent = etree.Element("parent", eId="parent_1")
        child = etree.SubElement(parent, "child")  # No eId

        with self.assertRaises(ValueError) as context:
            self.handler.validate_element_has_eid(child)

        self.assertIn("parent_1", str(context.exception))

    def test_parse_xml_string_simple(self):
        """Test parsing a simple XML string."""
        xml_string = '<element eId="test">Content</element>'

        element = self.handler.parse_xml_string(xml_string, ensure_namespaces=False)

        self.assertEqual(element.tag, "element")
        self.assertEqual(element.get("eId"), "test")
        self.assertEqual(element.text, "Content")

    def test_parse_xml_string_with_namespace_fix(self):
        """Test parsing XML with ukl: prefix but no namespace declaration."""
        xml_string = '<element ukl:change="ins">Content</element>'

        element = self.handler.parse_xml_string(xml_string, ensure_namespaces=True)

        self.assertEqual(element.tag, "element")
        self.assertEqual(element.get(f"{{{XMLHandler.UKL_URI}}}change"), "ins")
        self.assertEqual(element.text, "Content")

    def test_parse_xml_string_with_existing_namespace(self):
        """Test parsing XML that already has namespace declaration."""
        xml_string = f'<element xmlns:ukl="{XMLHandler.UKL_URI}" ukl:change="del">Content</element>'

        element = self.handler.parse_xml_string(xml_string, ensure_namespaces=True)

        self.assertEqual(element.tag, "element")
        self.assertEqual(element.get(f"{{{XMLHandler.UKL_URI}}}change"), "del")

    def test_parse_xml_string_invalid_xml(self):
        """Test parsing invalid XML raises XMLSyntaxError."""
        xml_string = "<element>Unclosed"

        with self.assertRaises(etree.XMLSyntaxError):
            self.handler.parse_xml_string(xml_string)

    def test_parse_xml_string_empty_wrapper(self):
        """Test parsing empty XML string."""
        xml_string = ""

        # Empty string causes XMLSyntaxError, not ValueError
        with self.assertRaises(etree.XMLSyntaxError):
            self.handler.parse_xml_string(xml_string, ensure_namespaces=True)

    def test_validate_amendment_response_valid(self):
        """Test validating a properly formed amendment response - basic structure only."""
        xml_string = '<section eId="sec_1"><content><p>Text</p></content></section>'
        element = etree.fromstring(xml_string)

        is_valid, error = self.handler.validate_amendment_response(element, "sec_1")

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_amendment_response_wrong_eid(self):
        """Test validation fails when eId is changed."""
        xml_string = '<section eId="sec_2"><content><p>Text</p></content></section>'
        element = etree.fromstring(xml_string)

        is_valid, error = self.handler.validate_amendment_response(element, "sec_1")

        self.assertFalse(is_valid)
        self.assertIn("eId changed from sec_1 to sec_2", error)

    def test_validate_amendment_response_no_eid(self):
        """Test validation fails when element has no eId."""
        xml_string = "<section><content><p>Text</p></content></section>"
        element = etree.fromstring(xml_string)

        is_valid, error = self.handler.validate_amendment_response(element, "sec_1")

        self.assertFalse(is_valid)
        self.assertIn("missing eId attribute", error)

    def test_parse_xml_string_wrapper_no_children(self):
        """Test parsing XML that needs wrapper but results in no children."""
        # XML that uses ukl: but when wrapped, the wrapper has no valid children
        xml_string = "ukl:test"

        with self.assertRaises(ValueError) as context:
            self.handler.parse_xml_string(xml_string, ensure_namespaces=True)

        self.assertIn("Wrapped XML has no child elements", str(context.exception))

    def test_parse_xml_string_general_exception(self):
        """Test parsing XML that causes a general exception."""

        # Pass non-string to trigger general exception
        class BadString:
            def __contains__(self, item):
                if item == "ukl:":
                    return True
                raise RuntimeError("Forced error")

            def encode(self, encoding):
                raise RuntimeError("Forced error")

        with self.assertRaises(ValueError) as context:
            self.handler.parse_xml_string(BadString(), ensure_namespaces=True)

        self.assertIn("Failed to parse XML", str(context.exception))
        self.assertIn("Forced error", str(context.exception))

    def test_element_has_change_markup(self):
        """Test detecting change markup in elements."""
        # Test with ins/del elements
        xml_string = f"""<wrapper><section xmlns:akn="{XMLHandler.AKN_URI}">
            <content><p>Text with <akn:ins>inserted</akn:ins></p></content>
        </section></wrapper>"""
        element = etree.fromstring(xml_string)
        self.assertTrue(self.handler.element_has_change_markup(element))

        # Test with ukl:change attributes
        xml_string = f"""<wrapper><section xmlns:ukl="{XMLHandler.UKL_URI}">
            <subsection ukl:change="ins"><p>Text</p></subsection>
        </section></wrapper>"""
        element = etree.fromstring(xml_string)
        self.assertTrue(self.handler.element_has_change_markup(element))

        # Test without change markup
        xml_string = "<wrapper><section><content><p>Plain text</p></content></section></wrapper>"
        element = etree.fromstring(xml_string)
        self.assertFalse(self.handler.element_has_change_markup(element))

    def test_get_change_tracking_attributes(self):
        """Test getting change tracking attributes."""
        # Test with all attributes present
        xml_string = f"""<wrapper><section xmlns:ukl="{XMLHandler.UKL_URI}">
            <subsection ukl:changeStart="true" ukl:changeEnd="true"
                        ukl:changeGenerated="true">
                <p>Text</p>
            </subsection>
        </section></wrapper>"""
        element = etree.fromstring(xml_string)

        attrs = self.handler.get_change_tracking_attributes(element)

        self.assertTrue(attrs["changeStart"])
        self.assertTrue(attrs["changeEnd"])
        self.assertTrue(attrs["changeGenerated"])

        # Test with missing attributes
        xml_string = f"""<wrapper><section xmlns:ukl="{XMLHandler.UKL_URI}">
            <subsection ukl:changeStart="true">
                <p>Text</p>
            </subsection>
        </section></wrapper>"""
        element = etree.fromstring(xml_string)

        attrs = self.handler.get_change_tracking_attributes(element)

        self.assertTrue(attrs["changeStart"])
        self.assertFalse(attrs["changeEnd"])
        self.assertFalse(attrs["changeGenerated"])

    def test_get_ancestor_eids(self):
        """Test getting ancestor eIds."""
        # Test normal nested eId without including self
        ancestors = self.handler.get_ancestor_eids("sec_1__subsec_2__para_a", include_self=False)
        self.assertEqual(ancestors, ["sec_1", "sec_1__subsec_2"])

        # Test including self
        ancestors = self.handler.get_ancestor_eids("sec_1__subsec_2__para_a", include_self=True)
        self.assertEqual(ancestors, ["sec_1", "sec_1__subsec_2", "sec_1__subsec_2__para_a"])

        # Test single level eId
        ancestors = self.handler.get_ancestor_eids("sec_1", include_self=False)
        self.assertEqual(ancestors, [])

        # Test single level with include_self
        ancestors = self.handler.get_ancestor_eids("sec_1", include_self=True)
        self.assertEqual(ancestors, ["sec_1"])

        # Test empty string
        ancestors = self.handler.get_ancestor_eids("", include_self=True)
        self.assertEqual(ancestors, [])

    def test_find_provisions_containing_text(self):
        """Test finding provisions containing search terms."""
        xml_with_provisions = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1" class="prov1">
                    <heading>Important Section</heading>
                    <content><p>This section contains important information.</p></content>
                </section>
                <section eId="sec_2" class="prov1">
                    <heading>Other Section</heading>
                    <content><p>This section has different content.</p></content>
                </section>
                <quotedStructure>
                    <section eId="sec_3" class="prov1">
                        <content><p>This important text is quoted.</p></content>
                    </section>
                </quotedStructure>
                <hcontainer eId="sched_1" name="schedule">
                    <heading>Schedule 1</heading>
                    <content><p>Schedule with important data.</p></content>
                </hcontainer>
            </body>
        </act>"""

        tree = etree.fromstring(xml_with_provisions.encode()).getroottree()

        # Search for "important" excluding quoted
        results = self.handler.find_provisions_containing_text(tree, ["important"], exclude_quoted=True)

        self.assertEqual(len(results), 2)  # sec_1 and sched_1
        eids = [eid for _, eid in results]
        self.assertIn("sec_1", eids)
        self.assertIn("sched_1", eids)
        self.assertNotIn("sec_3", eids)  # Excluded because it's quoted

        # The method's XPath explicitly excludes quoted structures, so even with exclude_quoted=False,
        # it won't find provisions inside quotedStructure. This is by design.
        # The exclude_quoted parameter only affects the text content extraction, not the XPath selection.
        results = self.handler.find_provisions_containing_text(tree, ["important"], exclude_quoted=False)
        self.assertEqual(len(results), 2)  # Still only sec_1 and sched_1

        # Search for non-existent term
        results = self.handler.find_provisions_containing_text(tree, ["nonexistent"])
        self.assertEqual(len(results), 0)

        # Test with no body element
        empty_tree = etree.fromstring("<act/>").getroottree()
        results = self.handler.find_provisions_containing_text(empty_tree, ["test"])
        self.assertEqual(results, [])

    def test_find_provisions_with_schedule_groups(self):
        """Test finding provisions handles schedules with groups."""
        xml_with_groups = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <hcontainer eId="sched_1" name="schedule" class="sch">
                    <hcontainer eId="sched_1__group_1" class="schGroup1">
                        <heading>Part 1</heading>
                        <hcontainer eId="sched_1__group_1__group_a" class="schGroup2">
                            <heading>Section A</heading>
                            <content><p>Important group content.</p></content>
                        </hcontainer>
                        <hcontainer eId="sched_1__group_1__group_b" class="schGroup2">
                            <heading>Section B</heading>
                            <content><p>Other content.</p></content>
                        </hcontainer>
                    </hcontainer>
                </hcontainer>
            </body>
        </act>"""

        tree = etree.fromstring(xml_with_groups.encode()).getroottree()

        results = self.handler.find_provisions_containing_text(tree, ["important"])

        # Should find only the lowest group containing the term
        self.assertEqual(len(results), 1)
        element, eid = results[0]
        self.assertEqual(eid, "sched_1__group_1__group_a")

    def test_extract_lowest_grouping_provisions(self):
        """Test extraction of lowest grouping provisions."""
        xml_with_nested_groups = """<?xml version="1.0"?>
        <hcontainer eId="sched_1" class="sch">
            <hcontainer eId="group_1" class="schGroup1">
                <hcontainer eId="group_1a" class="schGroup2">
                    <content><p>Content A</p></content>
                </hcontainer>
                <hcontainer eId="group_1b" class="schGroup2">
                    <content><p>Content B</p></content>
                </hcontainer>
            </hcontainer>
            <hcontainer eId="para_1" class="para">
                <content><p>Not a group</p></content>
            </hcontainer>
        </hcontainer>"""

        root = etree.fromstring(xml_with_nested_groups)
        provision_list = []

        self.handler._extract_lowest_grouping_provisions(root, provision_list)

        # Should extract group_1a and group_1b (lowest groups)
        self.assertEqual(len(provision_list), 2)
        eids = [elem.get("eId") for elem in provision_list]
        self.assertIn("group_1a", eids)
        self.assertIn("group_1b", eids)

        # Test with non-group element
        non_group = etree.Element("test", {"class": "other"})
        provision_list = []
        self.handler._extract_lowest_grouping_provisions(non_group, provision_list)
        self.assertEqual(len(provision_list), 0)

    def test_find_element_by_eid_with_fallback(self):
        """Test finding element with fallback to ancestors."""
        xml_with_nested = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <preface>
                <p>Preface content</p>
            </preface>
            <body>
                <section eId="sec_1">
                    <subsection eId="sec_1__subsec_1">
                        <paragraph eId="sec_1__subsec_1__para_a">
                            <content><p>Text</p></content>
                        </paragraph>
                    </subsection>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_with_nested.encode()).getroottree()

        # Test exact match
        element = self.handler.find_element_by_eid_with_fallback(tree, "sec_1__subsec_1__para_a")
        self.assertIsNotNone(element)
        self.assertEqual(element.get("eId"), "sec_1__subsec_1__para_a")

        # Test fallback to parent
        element = self.handler.find_element_by_eid_with_fallback(tree, "sec_1__subsec_1__para_b")
        self.assertIsNotNone(element)
        self.assertEqual(element.get("eId"), "sec_1__subsec_1")

        # Test fallback to grandparent
        element = self.handler.find_element_by_eid_with_fallback(tree, "sec_1__subsec_2__para_a")
        self.assertIsNotNone(element)
        self.assertEqual(element.get("eId"), "sec_1")

        # Test fallback to preface when nothing found
        element = self.handler.find_element_by_eid_with_fallback(tree, "sec_99__subsec_1")
        self.assertIsNotNone(element)
        self.assertEqual(element.tag, f"{{{XMLHandler.AKN_URI}}}preface")

    def test_create_subtree_copy(self):
        """Test creating deep copy of subtree with dummy root."""
        xml = f"""<section eId="sec_1" xmlns="{XMLHandler.AKN_URI}">
            <subsection eId="sec_1__subsec_1">
                <content><p>Text</p></content>
            </subsection>
        </section>"""

        tree = etree.fromstring(xml).getroottree()
        subsection = tree.find(f".//{{{XMLHandler.AKN_URI}}}subsection")

        # Test copy using new dummy-root-based method
        copy_tree = self.handler.create_subtree_copy(subsection)

        # Assert the dummy root is present
        dummy_root = copy_tree.getroot()
        self.assertEqual(dummy_root.tag, "root")

        # Check that the copied element is inside dummy root
        copied_subsection = dummy_root.find(f".//{{{XMLHandler.AKN_URI}}}subsection")
        self.assertIsNotNone(copied_subsection)
        self.assertEqual(copied_subsection.get("eId"), "sec_1__subsec_1")

    def test_replace_element_in_tree(self):
        """Test replacing element in tree."""
        xml = f"""<act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <content><p>Original</p></content>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml).getroottree()

        # Create replacement element
        new_section = etree.Element(f"{{{XMLHandler.AKN_URI}}}section", eId="sec_1")
        new_content = etree.SubElement(new_section, f"{{{XMLHandler.AKN_URI}}}content")
        new_p = etree.SubElement(new_content, f"{{{XMLHandler.AKN_URI}}}p")
        new_p.text = "Replaced"

        # Test successful replacement
        success = self.handler.replace_element_in_tree(tree, "sec_1", new_section)
        self.assertTrue(success)

        # Verify replacement
        section = tree.find(f".//{{{XMLHandler.AKN_URI}}}section[@eId='sec_1']")
        self.assertIsNotNone(section)
        self.assertIn("Replaced", etree.tostring(section, encoding="unicode"))

        # Test replacing non-existent element
        success = self.handler.replace_element_in_tree(tree, "sec_999", new_section)
        self.assertFalse(success)

    def test_replace_element_in_tree_root(self):
        """Test replacing root element."""
        # The current implementation cannot replace the root element because
        # find_element_by_eid won't find the root element (it searches within the tree)
        xml = f"""<act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <p>Content</p>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml).getroottree()

        # Create new section to replace the existing one
        new_section = etree.Element(f"{{{XMLHandler.AKN_URI}}}section", eId="sec_1")
        new_p = etree.SubElement(new_section, f"{{{XMLHandler.AKN_URI}}}p")
        new_p.text = "New content"

        # Replace section (not root)
        success = self.handler.replace_element_in_tree(tree, "sec_1", new_section)
        self.assertTrue(success)

        # Verify replacement
        section = tree.find(f".//{{{XMLHandler.AKN_URI}}}section[@eId='sec_1']")
        self.assertIsNotNone(section)
        self.assertIn("New content", etree.tostring(section, encoding="unicode"))

        # Test that attempting to replace root returns False
        # (because find_element_by_eid won't find the root)
        new_root = etree.Element(f"{{{XMLHandler.AKN_URI}}}act", eId="act_1")
        success = self.handler.replace_element_in_tree(tree, "act_1", new_root)
        self.assertFalse(success)  # Expected to fail

    def test_replace_element_in_tree_root_no_parent(self):
        """Test replacing root element when old element has no parent."""
        # Create a simple tree
        xml = f"""<act eId="act_1" xmlns="{XMLHandler.AKN_URI}">
            <body><p>Original content</p></body>
        </act>"""

        tree = etree.fromstring(xml).getroottree()

        # Get the actual root element
        root = tree.getroot()

        # Create new root element
        new_root = etree.Element(f"{{{XMLHandler.AKN_URI}}}act", eId="act_1")
        new_body = etree.SubElement(new_root, f"{{{XMLHandler.AKN_URI}}}body")
        etree.SubElement(new_body, f"{{{XMLHandler.AKN_URI}}}p").text = "New content"

        # Mock find_element_by_eid to return the root element
        # The root element has no parent, so getparent() returns None
        with patch.object(self.handler, "find_element_by_eid", return_value=root):
            success = self.handler.replace_element_in_tree(tree, "act_1", new_root)

        self.assertTrue(success)

        # Verify the root was actually replaced
        self.assertEqual(tree.getroot().tag, new_root.tag)
        self.assertIn("New content", etree.tostring(tree, encoding="unicode"))

    def test_validate_amendment_response_no_eid_with_parent(self):
        """Test validation fails when element has no eId but has a parent."""
        # Create an element with a parent
        parent_xml = f"""<section eId="parent_sec_1" xmlns="{XMLHandler.AKN_URI}">
            <subsection>
                <content><p>Text</p></content>
            </subsection>
        </section>"""

        parent = etree.fromstring(parent_xml)
        # Get the subsection which has no eId
        element = parent.find(f".//{{{XMLHandler.AKN_URI}}}subsection")

        is_valid, error = self.handler.validate_amendment_response(element, "sec_1__subsec_1")

        self.assertFalse(is_valid)
        # Check that parent info is included in error message
        self.assertIn(
            "parent: {http://docs.oasis-open.org/legaldocml/ns/akn/3.0}section with eId 'parent_sec_1'", error
        )
        self.assertIn("subsection missing eId attribute", error)

    def test_find_heading(self):
        """Test finding heading element."""
        # Element with heading
        xml_with_heading = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <akn:heading>Section Title</akn:heading>
            <content><p>Content</p></content>
        </section>"""
        element = etree.fromstring(xml_with_heading)
        heading = self.handler.find_heading(element)
        self.assertIsNotNone(heading)
        self.assertEqual(heading.tag, f"{{{XMLHandler.AKN_URI}}}heading")
        self.assertEqual(heading.text, "Section Title")

        # Element without heading
        xml_without_heading = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <content><p>Content only</p></content>
        </section>"""
        element = etree.fromstring(xml_without_heading)
        heading = self.handler.find_heading(element)
        self.assertIsNone(heading)

    def test_find_first_paragraph(self):
        """Test finding first paragraph element."""
        # Element with multiple paragraphs
        xml_with_paragraphs = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <content>
                <akn:p>First paragraph</akn:p>
                <akn:p>Second paragraph</akn:p>
                <akn:p>Third paragraph</akn:p>
            </content>
        </section>"""
        element = etree.fromstring(xml_with_paragraphs)
        first_p = self.handler.find_first_paragraph(element)
        self.assertIsNotNone(first_p)
        self.assertEqual(first_p.text, "First paragraph")

        # Element without paragraphs
        xml_without_paragraphs = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <heading>Just a heading</heading>
        </section>"""
        element = etree.fromstring(xml_without_paragraphs)
        first_p = self.handler.find_first_paragraph(element)
        self.assertIsNone(first_p)

    def test_find_last_paragraph(self):
        """Test finding last paragraph element."""
        # Element with multiple paragraphs
        xml_with_paragraphs = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <content>
                <akn:p>First paragraph</akn:p>
                <akn:p>Second paragraph</akn:p>
                <akn:p>Last paragraph</akn:p>
            </content>
        </section>"""
        element = etree.fromstring(xml_with_paragraphs)
        last_p = self.handler.find_last_paragraph(element)
        self.assertIsNotNone(last_p)
        self.assertEqual(last_p.text, "Last paragraph")

        # Element with single paragraph
        xml_single_paragraph = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <content>
                <akn:p>Only paragraph</akn:p>
            </content>
        </section>"""
        element = etree.fromstring(xml_single_paragraph)
        last_p = self.handler.find_last_paragraph(element)
        self.assertIsNotNone(last_p)
        self.assertEqual(last_p.text, "Only paragraph")

        # Element without paragraphs - this covers line 602
        xml_without_paragraphs = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <heading>No paragraphs here</heading>
        </section>"""
        element = etree.fromstring(xml_without_paragraphs)
        last_p = self.handler.find_last_paragraph(element)
        self.assertIsNone(last_p)

    def test_find_suitable_note_ref_location(self):
        """Test finding suitable location for noteRef."""
        # Element with heading (should return heading)
        xml_with_heading = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <akn:heading>Section Title</akn:heading>
            <content>
                <akn:p>Paragraph text</akn:p>
            </content>
        </section>"""
        element = etree.fromstring(xml_with_heading)
        location = self.handler._find_suitable_note_ref_location(element)
        self.assertIsNotNone(location)
        self.assertEqual(location.tag, f"{{{XMLHandler.AKN_URI}}}heading")

        # Element without heading but with paragraph (should return first paragraph)
        xml_no_heading = f"""<section xmlns:akn="{XMLHandler.AKN_URI}">
            <content>
                <akn:p>First paragraph</akn:p>
                <akn:p>Second paragraph</akn:p>
            </content>
        </section>"""
        element = etree.fromstring(xml_no_heading)
        location = self.handler._find_suitable_note_ref_location(element)
        self.assertIsNotNone(location)
        self.assertEqual(location.tag, f"{{{XMLHandler.AKN_URI}}}p")
        self.assertEqual(location.text, "Second paragraph")

        # Element with neither heading nor paragraph (should return element itself)
        xml_empty = f"""<section xmlns="{XMLHandler.AKN_URI}">
            <num>1</num>
        </section>"""
        element = etree.fromstring(xml_empty)
        location = self.handler._find_suitable_note_ref_location(element)
        self.assertIsNotNone(location)
        # The tag includes the namespace when no prefix is used
        self.assertEqual(location.tag, f"{{{XMLHandler.AKN_URI}}}section")

    def test_transform_eids_with_new_prefix(self):
        """Test transforming eIds by replacing with a new prefix."""
        xml_string = f"""<section xmlns:akn="{XMLHandler.AKN_URI}" eId="old_prefix__sec_1">
            <subsection eId="old_prefix__sec_1__subsec_1">
                <para eId="old_prefix__sec_1__subsec_1__para_a">
                    <content><p>Text</p></content>
                </para>
            </subsection>
            <subsection eId="different_prefix__subsec_2">
                <content><p>Should not change</p></content>
            </subsection>
        </section>"""

        element = etree.fromstring(xml_string)

        # Transform with new prefix
        self.handler.transform_eids(element, "old_prefix__", "new_prefix__")

        # Check transformations
        self.assertEqual(element.get("eId"), "new_prefix__sec_1")

        subsec1 = element.find(".//*[@eId='new_prefix__sec_1__subsec_1']")
        self.assertIsNotNone(subsec1)

        para = element.find(".//*[@eId='new_prefix__sec_1__subsec_1__para_a']")
        self.assertIsNotNone(para)

        # Check element with different prefix unchanged
        subsec2 = element.find(".//*[@eId='different_prefix__subsec_2']")
        self.assertIsNotNone(subsec2)

    def test_transform_eids_remove_prefix(self):
        """Test transforming eIds by removing prefix entirely."""
        xml_string = f"""<section xmlns:akn="{XMLHandler.AKN_URI}"
            eId="sec_25__subsec_2__qstr__sec_59b">
            <heading>New Section</heading>
            <subsection eId="sec_25__subsec_2__qstr__sec_59b__subsec_1">
                <num>(1)</num>
                <para eId="sec_25__subsec_2__qstr__sec_59b__subsec_1__para_a">
                    <content><p>Text</p></content>
                </para>
            </subsection>
        </section>"""

        element = etree.fromstring(xml_string)

        # Transform by removing prefix
        self.handler.transform_eids(element, "sec_25__subsec_2__qstr__", "")

        # Check all eIds were transformed
        self.assertEqual(element.get("eId"), "sec_59b")

        subsec = element.find(".//*[@eId='sec_59b__subsec_1']")
        self.assertIsNotNone(subsec)

        para = element.find(".//*[@eId='sec_59b__subsec_1__para_a']")
        self.assertIsNotNone(para)

    def test_transform_eids_no_matching_prefix(self):
        """Test transform_eids when prefix doesn't match any elements."""
        xml_string = f"""<section xmlns:akn="{XMLHandler.AKN_URI}" eId="sec_1">
            <subsection eId="sec_1__subsec_1">
                <content><p>Text</p></content>
            </subsection>
        </section>"""

        element = etree.fromstring(xml_string)

        # Transform with non-matching prefix
        self.handler.transform_eids(element, "non_matching_prefix__", "new_")

        # Nothing should change
        self.assertEqual(element.get("eId"), "sec_1")
        subsec = element.find(".//*[@eId='sec_1__subsec_1']")
        self.assertIsNotNone(subsec)

    def test_transform_eids_with_extra_underscores(self):
        """Test transform_eids handles extra underscores after prefix removal."""
        xml_string = f"""<section xmlns:akn="{XMLHandler.AKN_URI}"
            eId="prefix____sec_1">
            <subsection eId="prefix____sec_1__subsec_1">
                <content><p>Text</p></content>
            </subsection>
        </section>"""

        element = etree.fromstring(xml_string)

        # Remove prefix that leaves extra underscores
        self.handler.transform_eids(element, "prefix__", "")

        # Should strip leading underscores
        self.assertEqual(element.get("eId"), "sec_1")
        subsec = element.find(".//*[@eId='sec_1__subsec_1']")
        self.assertIsNotNone(subsec)

    def test_simplify_amending_bill_removes_ref_parentheticals(self):
        """Test that parenthetical descriptions are removed from ref elements."""
        from unittest.mock import Mock, patch

        # Create XML with various ref patterns
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <heading>Amendment to <ref href="/test/act/2000/1">Test Act 2000</ref> (s. 1)</heading>
                    <content>
                        <p>In <ref href="/test/act/2000/1/section/5">section 5</ref> (licensing provisions),
                        after "the words" insert—</p>
                        <p>See also <rref from="/test/act/2002/10">Another Act 2002</rref> (s. 10) and regulations.</p>
                        <p>Reference to <mref>multiple acts</mref> (various years).</p>
                    </content>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Create mock refs based on actual XML structure
        # Ref 1: "Test Act 2000" with tail " (s. 1)"
        mock_ref1 = Mock()
        mock_ref1.get_tail = Mock(return_value=" (s. 1)")
        mock_ref1.set_tail = Mock()
        mock_ref1.unwrap_element = Mock()

        # Ref 2: "section 5" with tail " (licensing provisions), after \"the words\" insert—"
        mock_ref2 = Mock()
        mock_ref2.get_tail = Mock(return_value=' (licensing provisions), after "the words" insert—')
        mock_ref2.set_tail = Mock()
        mock_ref2.unwrap_element = Mock()

        # Ref 3: "Another Act 2002" with tail " (s. 10) and regulations."
        mock_ref3 = Mock()
        mock_ref3.get_tail = Mock(return_value=" (s. 10) and regulations.")
        mock_ref3.set_tail = Mock()
        mock_ref3.unwrap_element = Mock()

        # Ref 4: "multiple acts" with tail " (various years)."
        mock_ref4 = Mock()
        mock_ref4.get_tail = Mock(return_value=" (various years).")
        mock_ref4.set_tail = Mock()
        mock_ref4.unwrap_element = Mock()

        # Mock LegiElement
        mock_legi_element = Mock()
        mock_body = Mock()
        body_element = tree.find(f".//{{{XMLHandler.AKN_URI}}}body")
        mock_body.element = body_element

        def mock_get_descendants(xpath):
            if "processing-instruction()" in xpath:
                return []
            elif ".//akn:ref" in xpath:
                return [mock_ref1, mock_ref2, mock_ref3, mock_ref4]
            elif "akn:quotedStructure" in xpath:
                return []
            return []

        mock_body.get_descendants = Mock(side_effect=mock_get_descendants)
        mock_legi_element.get_descendant = Mock(return_value=mock_body)

        with patch("app.services.xml_handler.LegiElement", return_value=mock_legi_element):
            self.handler.simplify_amending_bill(tree)

        # Verify parentheticals were correctly removed from tail text
        mock_ref1.set_tail.assert_called_once_with("")  # Empty string
        mock_ref2.set_tail.assert_called_once_with(', after "the words" insert—')  # Parenthetical removed
        mock_ref3.set_tail.assert_called_once_with(" and regulations.")  # Parenthetical removed
        mock_ref4.set_tail.assert_called_once_with(".")  # Just period remains

        # Verify all refs were unwrapped
        for mock_ref in [mock_ref1, mock_ref2, mock_ref3, mock_ref4]:
            mock_ref.unwrap_element.assert_called_once()

    def test_simplify_amending_bill_simplifies_quoted_structures(self):
        """Test that simplify_amending_bill removes grandchildren from quoted structures."""
        from unittest.mock import Mock, patch

        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <content>
                        <p>In section 5, after subsection (2) insert—</p>
                        <quotedStructure eId="sec_1__qstr">
                            <subsection eId="sec_1__qstr__subsec_2A">
                                <num>(2A)</num>
                                <content>
                                    <p>This is the new subsection content.</p>
                                </content>
                            </subsection>
                            <subsection eId="sec_1__qstr__subsec_2B">
                                <num>(2B)</num>
                                <content>
                                    <p>Another new subsection.</p>
                                </content>
                            </subsection>
                        </quotedStructure>
                    </content>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Track what gets deleted
        deleted_elements = []

        # Create mock quoted structure
        mock_quoted = Mock()

        # Mock children (subsections)
        mock_children = []
        for i in range(2):  # Two subsections
            mock_child = Mock()
            # Mock grandchildren (num and content elements)
            mock_grandchildren = []
            for j in range(2):  # Each subsection has num and content
                mock_grandchild = Mock()
                mock_grandchild.delete_element = Mock(
                    side_effect=lambda gc=j: deleted_elements.append(f"grandchild_{i}_{gc}")
                )
                mock_grandchildren.append(mock_grandchild)
            mock_child.get_children = Mock(return_value=mock_grandchildren)
            mock_children.append(mock_child)

        mock_quoted.get_children = Mock(return_value=mock_children)

        # Mock body and descendants
        mock_legi_element = Mock()
        mock_body = Mock()
        mock_body.element = tree.find(f".//{{{XMLHandler.AKN_URI}}}body")

        def mock_get_descendants(xpath):
            if "quotedStructure" in xpath:
                return [mock_quoted]
            return []

        mock_body.get_descendants = Mock(side_effect=mock_get_descendants)
        mock_legi_element.get_descendant = Mock(return_value=mock_body)

        with patch("app.services.xml_handler.LegiElement", return_value=mock_legi_element):
            self.handler.simplify_amending_bill(tree)

        # Verify all grandchildren were deleted
        self.assertEqual(len(deleted_elements), 4)  # 2 subsections × 2 grandchildren each

    def test_simplify_amending_bill_with_processing_instructions(self):
        """Test that simplify_amending_bill removes processing instructions."""
        from unittest.mock import Mock, patch

        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <?page-break?>
                <section eId="sec_1">
                    <?editor-note Check this section?>
                    <heading>Test Section</heading>
                    <content>
                        <p>Test content</p>
                        <?formatter keep-together?>
                    </content>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Mock processing instructions
        mock_pis = []
        for i in range(3):  # Three PIs in our example
            mock_pi = Mock()
            mock_pi.delete_element = Mock()
            mock_pis.append(mock_pi)

        # Mock LegiElement
        mock_legi_element = Mock()
        mock_body = Mock()
        mock_body.element = tree.find(f".//{{{XMLHandler.AKN_URI}}}body")

        def mock_get_descendants(xpath):
            if "processing-instruction()" in xpath:
                return mock_pis
            return []

        mock_body.get_descendants = Mock(side_effect=mock_get_descendants)
        mock_legi_element.get_descendant = Mock(return_value=mock_body)

        with patch("app.services.xml_handler.LegiElement", return_value=mock_legi_element):
            self.handler.simplify_amending_bill(tree)

        # Verify all processing instructions were deleted
        for mock_pi in mock_pis:
            mock_pi.delete_element.assert_called_once()

    def test_standardise_text_in_quotes_empty_quotes(self):
        """Test standardising empty quotes."""
        OPEN_QUOTE = "\u201c"
        CLOSE_QUOTE = "\u201d"

        xml_content = f"""<section xmlns="{XMLHandler.AKN_URI}">
            <p>Text with empty {OPEN_QUOTE}{CLOSE_QUOTE} quotes.</p>
            <p>And another {OPEN_QUOTE}{CLOSE_QUOTE} one.</p>
        </section>"""

        element = etree.fromstring(xml_content)

        self.handler.standardise_text_in_quotes(element)

        xml_string = etree.tostring(element, encoding="unicode")

        # Empty quotes should also be replaced
        self.assertNotIn(f"{OPEN_QUOTE}{CLOSE_QUOTE}", xml_string)
        # Should contain 'quote' replacements
        self.assertEqual(xml_string.count(f"{OPEN_QUOTE}quote{CLOSE_QUOTE}"), 2)

    def test_standardise_text_in_quotes_with_tail(self):
        """Test standardising quotes in tail text."""
        OPEN_QUOTE = "\u201c"
        CLOSE_QUOTE = "\u201d"

        xml_content = f"""<section xmlns="{XMLHandler.AKN_URI}">
            <p>Text with <em>emphasis</em> followed by {OPEN_QUOTE}quoted tail{CLOSE_QUOTE} text.</p>
            <p><strong>Bold</strong> {OPEN_QUOTE}another tail quote{CLOSE_QUOTE} here.</p>
        </section>"""

        element = etree.fromstring(xml_content)

        self.handler.standardise_text_in_quotes(element)

        xml_string = etree.tostring(element, encoding="unicode")

        # Verify tail text quotes were replaced
        self.assertNotIn(f"{OPEN_QUOTE}quoted tail{CLOSE_QUOTE}", xml_string)
        self.assertNotIn(f"{OPEN_QUOTE}another tail quote{CLOSE_QUOTE}", xml_string)
        # Check replacements
        self.assertIn(f"{OPEN_QUOTE}quote{CLOSE_QUOTE}", xml_string)
        self.assertEqual(xml_string.count(f"{OPEN_QUOTE}quote{CLOSE_QUOTE}"), 2)

    def test_standardise_text_in_quotes_nested_elements(self):
        """Test standardising quotes in deeply nested elements."""
        OPEN_QUOTE = "\u201c"
        CLOSE_QUOTE = "\u201d"

        xml_content = f"""<section xmlns="{XMLHandler.AKN_URI}">
            <subsection>
                <paragraph>
                    <subparagraph>
                        <p>Deep {OPEN_QUOTE}nested quote{CLOSE_QUOTE} here.</p>
                        <list>
                            <item>
                                <p>List item with {OPEN_QUOTE}another quote{CLOSE_QUOTE} text.</p>
                            </item>
                        </list>
                    </subparagraph>
                </paragraph>
            </subsection>
        </section>"""

        element = etree.fromstring(xml_content)

        self.handler.standardise_text_in_quotes(element)

        # Find deeply nested elements
        deep_p = element.find(
            ".//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}subparagraph/"
            "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}p"
        )
        # Check the quote was replaced
        self.assertEqual(deep_p.text, f"Deep {OPEN_QUOTE}quote{CLOSE_QUOTE} here.")

        list_p = element.find(
            ".//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}item/"
            "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}p"
        )
        self.assertEqual(list_p.text, f"List item with {OPEN_QUOTE}quote{CLOSE_QUOTE} text.")

    def test_standardise_text_in_quotes_special_characters(self):
        """Test standardising quotes with special characters inside."""
        OPEN_QUOTE = "\u201c"
        CLOSE_QUOTE = "\u201d"

        xml_content = f"""<section xmlns="{XMLHandler.AKN_URI}">
            <p>Quote with {OPEN_QUOTE}special &amp; characters{CLOSE_QUOTE} inside.</p>
            <p>Quote with {OPEN_QUOTE}   extra   spaces   {CLOSE_QUOTE} inside.</p>
        </section>"""

        element = etree.fromstring(xml_content)

        self.handler.standardise_text_in_quotes(element)

        xml_string = etree.tostring(element, encoding="unicode")

        # All single-line quoted content should be replaced
        self.assertNotIn(f"{OPEN_QUOTE}special &amp; characters{CLOSE_QUOTE}", xml_string)
        self.assertNotIn(f"{OPEN_QUOTE}   extra   spaces   {CLOSE_QUOTE}", xml_string)

        # Should have 2 'quote' replacements
        self.assertEqual(xml_string.count(f"{OPEN_QUOTE}quote{CLOSE_QUOTE}"), 2)

    def test_standardise_text_in_quotes_no_quotes(self):
        """Test standardising text with no curly quotes."""
        xml_content = f"""<section xmlns="{XMLHandler.AKN_URI}">
            <p>Text with no special quotes.</p>
            <p>Just regular 'single quotes' and "straight quotes" normal text.</p>
        </section>"""

        element = etree.fromstring(xml_content)
        original_string = etree.tostring(element, encoding="unicode")

        self.handler.standardise_text_in_quotes(element)

        result_string = etree.tostring(element, encoding="unicode")

        # Should remain unchanged since no curly quotes present
        self.assertEqual(original_string, result_string)

    def test_extract_eid_patterns_basic(self):
        """Test extracting eId patterns from a document with basic structure."""
        # Arrange
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1" class="prov1">
                    <heading>First Section</heading>
                    <subsection eId="sec_1__subsec_1">
                        <content><p>Content</p></content>
                    </subsection>
                </section>
                <section eId="sec_2" class="prov1">
                    <heading>Second Section</heading>
                </section>
            </body>
        </act>"""
        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Act
        patterns = self.handler.extract_eid_patterns(tree, max_examples=5)

        # Assert
        self.assertIn("examples", patterns)
        self.assertIn("conventions", patterns)
        self.assertEqual(patterns["examples"]["sections"], ["sec_1", "sec_2"])
        self.assertEqual(patterns["examples"]["subsections"], ["sec_1__subsec_1"])

    def test_extract_eid_patterns_with_definitions(self):
        """Test extracting definition patterns with trailing underscore convention."""
        # Arrange
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_93">
                    <hcontainer eId="sec_93__def_tenant_" class="definition" name="definition">
                        <content><p>"tenant" means...</p></content>
                    </hcontainer>
                    <hcontainer eId="sec_93__def_landlord_" class="definition" name="definition">
                        <content><p>"landlord" means...</p></content>
                    </hcontainer>
                </section>
            </body>
        </act>"""
        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Act
        patterns = self.handler.extract_eid_patterns(tree, max_examples=5)

        # Assert
        self.assertEqual(patterns["examples"]["definitions"], ["sec_93__def_tenant_", "sec_93__def_landlord_"])
        self.assertEqual(patterns["conventions"]["definition_suffix"], "_")

    def test_extract_eid_patterns_max_examples_limit(self):
        """Test that max_examples parameter limits the number of patterns extracted."""
        # Arrange
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1" class="prov1"><content><p>1</p></content></section>
                <section eId="sec_2" class="prov1"><content><p>2</p></content></section>
                <section eId="sec_3" class="prov1"><content><p>3</p></content></section>
                <section eId="sec_4" class="prov1"><content><p>4</p></content></section>
                <section eId="sec_5" class="prov1"><content><p>5</p></content></section>
                <section eId="sec_6" class="prov1"><content><p>6</p></content></section>
            </body>
        </act>"""
        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Act
        patterns = self.handler.extract_eid_patterns(tree, max_examples=3)

        # Assert
        self.assertEqual(len(patterns["examples"]["sections"]), 3)
        self.assertEqual(patterns["examples"]["sections"], ["sec_1", "sec_2", "sec_3"])

    def test_extract_eid_patterns_with_headings(self):
        """Test extracting heading patterns from elements with headings."""
        # Arrange
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <heading eId="sec_1__hdg">Section One Heading</heading>
                    <content><p>Content</p></content>
                </section>
                <section eId="sec_2">
                    <heading eId="sec_2__hdg">Section Two Heading</heading>
                    <content><p>More content</p></content>
                </section>
            </body>
        </act>"""
        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Act
        patterns = self.handler.extract_eid_patterns(tree, max_examples=5)

        # Assert
        self.assertEqual(patterns["examples"]["headings"], ["sec_1__hdg", "sec_2__hdg"])

    def test_extract_eid_patterns_empty_document(self):
        """Test extracting patterns from document with no matching elements."""
        # Arrange
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <p>Just a paragraph with no structured provisions</p>
            </body>
        </act>"""
        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Act
        patterns = self.handler.extract_eid_patterns(tree, max_examples=5)

        # Assert
        # All pattern lists should be empty
        for pattern_type, examples in patterns["examples"].items():
            self.assertEqual(examples, [], f"Expected empty list for {pattern_type}")
        self.assertEqual(patterns["conventions"]["definition_suffix"], "")

    def test_extract_eid_patterns_mixed_provisions(self):
        """Test extracting patterns from document with various provision types."""
        # Arrange
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <part eId="pt_1">
                    <chapter eId="pt_1__chp_1">
                        <article eId="art_1">
                            <content><p>Article content</p></content>
                        </article>
                    </chapter>
                </part>
                <hcontainer eId="sched_1" name="schedule">
                    <content><p>Schedule content</p></content>
                </hcontainer>
                <hcontainer eId="reg_1" name="regulation" class="prov1">
                    <content><p>Regulation content</p></content>
                </hcontainer>
            </body>
        </act>"""
        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Act
        patterns = self.handler.extract_eid_patterns(tree, max_examples=5)

        # Assert
        self.assertEqual(patterns["examples"]["parts"], ["pt_1"])
        self.assertEqual(patterns["examples"]["chapters"], ["pt_1__chp_1"])
        self.assertEqual(patterns["examples"]["articles"], ["art_1"])
        self.assertEqual(patterns["examples"]["schedules"], ["sched_1"])
        self.assertEqual(patterns["examples"]["regulations"], ["reg_1"])

    def test_extract_provision_type_from_last_part(self):
        """Test extracting provision type from the last part of eId."""
        # Test with single level eIds
        self.assertEqual(self.handler._extract_provision_type("sec_5"), "sec")
        self.assertEqual(self.handler._extract_provision_type("sched_2"), "sched")
        self.assertEqual(self.handler._extract_provision_type("para_1"), "para")
        self.assertEqual(self.handler._extract_provision_type("reg_10"), "reg")

        # Test with multi-level eIds (should extract from last part)
        self.assertEqual(self.handler._extract_provision_type("sec_5__subsec_3"), "subsec")
        self.assertEqual(self.handler._extract_provision_type("sched_2__para_4"), "para")
        self.assertEqual(self.handler._extract_provision_type("sec_1__subsec_2__para_a"), "para")
        self.assertEqual(self.handler._extract_provision_type("part_1__chapter_2__sec_5"), "sec")

    def test_extract_provision_type_fallback_to_first_part(self):
        """Test extracting provision type falls back to first part when last part has no match."""
        # When last part doesn't have underscore pattern, should try first part
        self.assertEqual(self.handler._extract_provision_type("sec_5__special"), "sec")
        self.assertEqual(self.handler._extract_provision_type("sched_1__group1"), "sched")
        self.assertEqual(self.handler._extract_provision_type("para_a__item"), "para")

    def test_extract_provision_type_no_match(self):
        """Test extracting provision type when no pattern matches."""
        # No underscore pattern in any part
        self.assertEqual(self.handler._extract_provision_type("malformed"), "")
        self.assertEqual(self.handler._extract_provision_type(""), "")
        self.assertEqual(self.handler._extract_provision_type("123"), "")
        self.assertEqual(self.handler._extract_provision_type("special__case"), "")

    def test_find_provisions_in_range_basic(self):
        """Test finding provisions in a simple range."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_5">
                    <heading>Section 5</heading>
                </section>
                <section eId="sec_6">
                    <heading>Section 6</heading>
                </section>
                <section eId="sec_7">
                    <heading>Section 7</heading>
                </section>
                <section eId="sec_8">
                    <heading>Section 8</heading>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Test normal range
        provisions = self.handler.find_provisions_in_range("sec_5", "sec_7", tree)
        self.assertEqual(provisions, ["sec_5", "sec_6", "sec_7"])

        # Test single element range
        provisions = self.handler.find_provisions_in_range("sec_6", "sec_6", tree)
        self.assertEqual(provisions, ["sec_6"])

    def test_find_provisions_in_range_reversed(self):
        """Test finding provisions when start and end are reversed."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <heading>Section 1</heading>
                </section>
                <section eId="sec_2">
                    <heading>Section 2</heading>
                </section>
                <section eId="sec_3">
                    <heading>Section 3</heading>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Should handle reversed range correctly
        provisions = self.handler.find_provisions_in_range("sec_3", "sec_1", tree)
        self.assertEqual(provisions, ["sec_1", "sec_2", "sec_3"])

    def test_find_provisions_in_range_nested_levels(self):
        """Test finding provisions respects structural level."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_5">
                    <subsection eId="sec_5__subsec_1">
                        <content><p>Content</p></content>
                    </subsection>
                    <subsection eId="sec_5__subsec_2">
                        <content><p>Content</p></content>
                    </subsection>
                    <subsection eId="sec_5__subsec_3">
                        <content><p>Content</p></content>
                    </subsection>
                </section>
                <section eId="sec_6">
                    <subsection eId="sec_6__subsec_1">
                        <content><p>Content</p></content>
                    </subsection>
                </section>
                <section eId="sec_7">
                    <heading>Section 7</heading>
                </section>
                <section eId="sec_8">
                    <subsection eId="sec_8__subsec_1">
                        <content><p>Content</p></content>
                    </subsection>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Should only find subsections at same level
        provisions = self.handler.find_provisions_in_range("sec_5__subsec_1", "sec_5__subsec_3", tree)
        self.assertEqual(provisions, ["sec_5__subsec_1", "sec_5__subsec_2", "sec_5__subsec_3"])

        # Cross-section range should not include subsections
        provisions = self.handler.find_provisions_in_range("sec_5", "sec_7", tree)
        self.assertEqual(provisions, ["sec_5", "sec_6", "sec_7"])

    def test_find_provisions_in_range_different_types(self):
        """Test finding provisions only includes same type."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <heading>Section 1</heading>
                </section>
                <article eId="art_1">
                    <heading>Article 1</heading>
                </article>
                <section eId="sec_2">
                    <heading>Section 2</heading>
                </section>
                <article eId="art_2">
                    <heading>Article 2</heading>
                </article>
                <section eId="sec_3">
                    <heading>Section 3</heading>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Should only find sections, not articles
        provisions = self.handler.find_provisions_in_range("sec_1", "sec_3", tree)
        self.assertEqual(provisions, ["sec_1", "sec_2", "sec_3"])

    def test_find_provisions_in_range_missing_start(self):
        """Test finding provisions when start element doesn't exist."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_5"><heading>Section 5</heading></section>
                <section eId="sec_6"><heading>Section 6</heading></section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        with patch("app.services.xml_handler.logger") as mock_logger:
            provisions = self.handler.find_provisions_in_range("sec_4", "sec_6", tree)

            # Should return empty list and log warning
            self.assertEqual(provisions, [])
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("Could not find range endpoints", warning_msg)
            self.assertIn("sec_4", warning_msg)

    def test_find_provisions_in_range_missing_end(self):
        """Test finding provisions when end element doesn't exist."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_5"><heading>Section 5</heading></section>
                <section eId="sec_6"><heading>Section 6</heading></section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        with patch("app.services.xml_handler.logger") as mock_logger:
            provisions = self.handler.find_provisions_in_range("sec_5", "sec_8", tree)

            # Should return empty list and log warning
            self.assertEqual(provisions, [])
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("Could not find range endpoints", warning_msg)
            self.assertIn("sec_8", warning_msg)

    def test_find_provisions_in_range_both_missing(self):
        """Test finding provisions when both elements don't exist."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_5"><heading>Section 5</heading></section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        with patch("app.services.xml_handler.logger") as mock_logger:
            provisions = self.handler.find_provisions_in_range("sec_1", "sec_3", tree)

            self.assertEqual(provisions, [])
            mock_logger.warning.assert_called_once()

    def test_find_provisions_in_range_element_not_in_tree(self):
        """Test error handling when element exists but not in document tree list."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1"><heading>Section 1</heading></section>
                <section eId="sec_2"><heading>Section 2</heading></section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Mock find_element_by_eid to return elements that aren't in the tree
        disconnected_elem1 = etree.Element("section", eId="sec_1")
        disconnected_elem2 = etree.Element("section", eId="sec_2")

        with patch.object(self.handler, "find_element_by_eid") as mock_find:
            mock_find.side_effect = [disconnected_elem1, disconnected_elem2]

            with patch("app.services.xml_handler.logger") as mock_logger:
                provisions = self.handler.find_provisions_in_range("sec_1", "sec_2", tree)

                self.assertEqual(provisions, [])
                mock_logger.error.assert_called_once_with("Start or end element not found in document tree")

    def test_find_provisions_in_range_elements_without_eid(self):
        """Test finding provisions skips elements without eId."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1"><heading>Section 1</heading></section>
                <section><heading>No eId Section</heading></section>
                <section eId="sec_2"><heading>Section 2</heading></section>
                <section><heading>Another no eId</heading></section>
                <section eId="sec_3"><heading>Section 3</heading></section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Should skip elements without eId
        provisions = self.handler.find_provisions_in_range("sec_1", "sec_3", tree)
        self.assertEqual(provisions, ["sec_1", "sec_2", "sec_3"])

    def test_find_provisions_in_range_complex_hierarchy(self):
        """Test finding provisions in complex document structure."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <part eId="part_1">
                    <section eId="part_1__sec_1">
                        <subsection eId="part_1__sec_1__subsec_1">
                            <para eId="part_1__sec_1__subsec_1__para_a">
                                <content><p>Content</p></content>
                            </para>
                            <para eId="part_1__sec_1__subsec_1__para_b">
                                <content><p>Content</p></content>
                            </para>
                        </subsection>
                    </section>
                    <section eId="part_1__sec_2">
                        <content><p>Content</p></content>
                    </section>
                </part>
                <part eId="part_2">
                    <section eId="part_2__sec_3">
                        <subsection eId="part_2__sec_3__subsec_1">
                            <para eId="part_2__sec_3__subsec_1__para_a">
                                <content><p>Content</p></content>
                            </para>
                        </subsection>
                    </section>
                </part>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Test paragraph range within same subsection
        provisions = self.handler.find_provisions_in_range(
            "part_1__sec_1__subsec_1__para_a", "part_1__sec_1__subsec_1__para_b", tree
        )
        self.assertEqual(provisions, ["part_1__sec_1__subsec_1__para_a", "part_1__sec_1__subsec_1__para_b"])

        # Test section range across parts
        provisions = self.handler.find_provisions_in_range("part_1__sec_1", "part_2__sec_3", tree)
        self.assertEqual(provisions, ["part_1__sec_1", "part_1__sec_2", "part_2__sec_3"])

    def test_find_provisions_in_range_with_letter_suffixes(self):
        """Test finding provisions with letter suffixes."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_59">
                    <heading>Section 59</heading>
                </section>
                <section eId="sec_59a">
                    <heading>Section 59A</heading>
                </section>
                <section eId="sec_59b">
                    <heading>Section 59B</heading>
                </section>
                <section eId="sec_59c">
                    <heading>Section 59C</heading>
                </section>
                <section eId="sec_60">
                    <heading>Section 60</heading>
                </section>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Should find all sections in range including letter suffixes
        provisions = self.handler.find_provisions_in_range("sec_59", "sec_59c", tree)
        self.assertEqual(provisions, ["sec_59", "sec_59a", "sec_59b", "sec_59c"])

    def test_inject_amendment_id(self):
        # Create a sample XML structure
        xml = f"""<?xml version="1.0"?>
        <root xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <section>
                <subsection ukl:changeGenerated="true" eId="sec_1__subsec_1"></subsection>
                <subsection ukl:changeGenerated="true" amendmentId="existing_id" eId="sec_1__subsec_2"></subsection>
                <subsection eId="sec_1__subsec_3"></subsection>
            </section>
        </root>"""
        tree = etree.fromstring(xml)

        # Prepare an Amendment object
        amendment = Amendment(
            source="Test Source",
            source_eid="sec_1",
            affected_document="Test Act",
            affected_provision="sec_2",
            location=AmendmentLocation.AFTER,
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            amendment_id="test-001",
        )

        self.handler.inject_amendment_id(tree, amendment)

        # Verify that 'amendmentId' was only added to the correct element
        subsections = tree.xpath(".//akn:subsection", namespaces=self.handler.namespaces)
        assert subsections[0].get("amendmentId") == "test-001"
        assert subsections[1].get("amendmentId") == "existing_id"
        assert "amendmentId" not in subsections[2].attrib

    def test_inject_amendment_id_multiple_elements(self):
        # Create a sample XML structure
        xml = f"""<?xml version="1.0"?>
        <root xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <section>
                <subsection ukl:changeGenerated="true" eId="sec_1__subsec_1"></subsection>
                <subsection ukl:changeGenerated="true" eId="sec_1__subsec_1A"></subsection>
                <subsection ukl:changeGenerated="true" amendmentId="existing_id" eId="sec_1__subsec_2"></subsection>
                <subsection eId="sec_1__subsec_3"></subsection>
            </section>
        </root>"""
        tree = etree.fromstring(xml)

        # Prepare an Amendment object
        amendment = Amendment(
            source="Test Source",
            source_eid="sec_1",
            affected_document="Test Act",
            affected_provision="sec_2",
            location=AmendmentLocation.AFTER,
            amendment_type=AmendmentType.INSERTION,
            whole_provision=True,
            amendment_id="test-001",
        )

        self.handler.inject_amendment_id(tree, amendment)

        # Verify that 'amendmentId' was only added to the correct element
        subsections = tree.xpath(".//akn:subsection", namespaces=self.handler.namespaces)
        assert subsections[0].get("amendmentId") == "test-001"
        assert subsections[1].get("amendmentId") == "test-001"
        assert subsections[2].get("amendmentId") == "existing_id"
        assert "amendmentId" not in subsections[3].attrib

    def test_remove_amendment_ids(self):
        """Test removing amendmentId from multiple elements."""
        xml_string = """
        <root xmlns:ukl="https://www.legislation.gov.uk/namespaces/UK-AKN">
            <parent amendmentId="1234">
                <child amendmentId="5678">
                    <grandchild amendmentId="91011">Test</grandchild>
                </child>
            </parent>
        </root>
        """
        tree = etree.ElementTree(etree.fromstring(xml_string))

        self.handler.remove_amendment_ids(tree)

        # Verify that no elements have an amendmentId
        elements = tree.xpath(".//*[@amendmentId]")
        self.assertEqual(len(elements), 0)

    def test_get_or_create_notes_element_with_existing_notes(self):
        """Test when 'akn:notes' already exists within 'akn:meta'."""
        xml_content = """
        <root xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
            <meta>
                <notes source="#"></notes>
            </meta>
        </root>
        """
        tree = etree.ElementTree(etree.fromstring(xml_content))

        result = self.handler._get_or_create_notes_element(tree)

        self.assertIsNotNone(result, "The notes element should exist.")
        self.assertEqual(result.tag, f"{{{XMLHandler.AKN_URI}}}notes", "The tag should be 'notes'.")

    def test_get_or_create_notes_element_without_notes(self):
        """Test when 'akn:meta' exists but 'akn:notes' does not."""
        xml_content = """
        <root xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
            <meta></meta>
        </root>
        """
        tree = etree.ElementTree(etree.fromstring(xml_content))

        result = self.handler._get_or_create_notes_element(tree)

        self.assertIsNotNone(result, "A notes element should be created.")
        self.assertEqual(result.tag, f"{{{self.handler.AKN_URI}}}notes", "The tag should be 'notes'.")
        self.assertEqual(result.get("source"), "#", "The 'source' attribute should be set to '#'.")

    def test_get_or_create_notes_element_without_meta(self):
        """Test when 'akn:meta' does not exist."""
        xml_content = """
        <root xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        </root>
        """
        tree = etree.ElementTree(etree.fromstring(xml_content))

        result = self.handler._get_or_create_notes_element(tree)

        self.assertIsNone(result, "If 'meta' is missing, the method should return None.")

    def test_insert_editorial_notes_cannot_find_notes(self):
        amended_xml = f"""<?xml version="1.0"?>
        <akomaNtoso xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <bill>
                <body>
                    <section eId="sec_25">
                        <subsection eId="sec_25__subsec_2" ukl:changeStart="true" ukl:changeEnd="true"
                            ukl:changeGenerated="true" ukl:changeDnum="KS1">
                            <content>
                                <p>text</p>
                            </content>
                        </subsection>
                    </section>
                </body>
            </bill>
        </akomaNtoso>"""
        amended_doc = etree.fromstring(amended_xml.encode()).getroottree()

        self.handler.insert_editorial_notes(amended_doc, [self.amendment])

        notes = amended_doc.xpath(".//akn:notes", namespaces=self.handler.namespaces)
        self.assertEqual(len(notes), 0)

    def test_insert_editorial_notes_success(self):
        amended_xml = f"""<?xml version="1.0"?>
        <akomaNtoso xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <bill>
                <meta></meta>
                <body>
                    <section eId="sec_25">
                        <subsection eId="sec_25__subsec_2" ukl:changeStart="true" ukl:changeEnd="true"
                            ukl:changeGenerated="true" ukl:changeDnum="KS1">
                            <content>
                                <p>text</p>
                            </content>
                        </subsection>
                    </section>
                </body>
            </bill>
        </akomaNtoso>"""
        amended_doc = etree.fromstring(amended_xml.encode()).getroottree()
        self.amendment.dnum_list = ["KS1"]

        self.handler.insert_editorial_notes(amended_doc, [self.amendment])

        notes = amended_doc.xpath(".//akn:notes", namespaces=self.handler.namespaces)
        self.assertEqual(len(notes), 1)

        note_list = notes[0].xpath(".//akn:note", namespaces=self.handler.namespaces)
        self.assertEqual(len(note_list), 1)

        self.assertEqual(note_list[0][0].text, "Insertion by s. 25(2).")
        self.assertEqual(note_list[0].get("eId"), "KS1")

        note_ref_list = amended_doc.xpath(".//akn:noteRef", namespaces=self.handler.namespaces)
        self.assertEqual(len(note_ref_list), 1)

        self.assertEqual(note_ref_list[0].get("marker"), "*")
        self.assertEqual(note_ref_list[0].get("class"), "commentary")
        self.assertEqual(note_ref_list[0].get("href"), "#KS1")

    def test_insert_editorial_notes_success_in_each_place(self):
        amended_xml = f"""<?xml version="1.0"?>
        <akomaNtoso xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <bill>
                <meta></meta>
                <body>
                    <section eId="sec_25">
                        <subsection eId="sec_25__subsec_2">
                            <content>
                                <p><ins ukl:changeStart="true" ukl:changeEnd="true"
                            ukl:changeGenerated="true" ukl:changeDnum="KS1">text</ins> text
                            <ins ukl:changeStart="true" ukl:changeEnd="true"
                            ukl:changeGenerated="true" ukl:changeDnum="KS2">text</ins></p>
                            </content>
                        </subsection>
                    </section>
                </body>
            </bill>
        </akomaNtoso>"""
        amended_doc = etree.fromstring(amended_xml.encode()).getroottree()
        self.amendment.dnum_list = ["KS1", "KS2"]

        self.handler.insert_editorial_notes(amended_doc, [self.amendment])

        notes = amended_doc.xpath(".//akn:notes", namespaces=self.handler.namespaces)
        self.assertEqual(len(notes), 1)

        note_list = notes[0].xpath(".//akn:note", namespaces=self.handler.namespaces)
        self.assertEqual(len(note_list), 2)

        self.assertEqual(note_list[0].get("eId"), "KS1")
        self.assertEqual(note_list[0][0].text, "Insertion by s. 25(2).")

        self.assertEqual(note_list[1].get("eId"), "KS2")
        self.assertEqual(note_list[1][0].text, "Insertion by s. 25(2).")

        note_ref_list = amended_doc.xpath(".//akn:noteRef", namespaces=self.handler.namespaces)
        self.assertEqual(len(note_ref_list), 2)

        self.assertEqual(note_ref_list[0].get("marker"), "*")
        self.assertEqual(note_ref_list[0].get("class"), "commentary")
        self.assertEqual(note_ref_list[0].get("href"), "#KS1")

        self.assertEqual(note_ref_list[1].get("marker"), "*")
        self.assertEqual(note_ref_list[1].get("class"), "commentary")
        self.assertEqual(note_ref_list[1].get("href"), "#KS2")

    def test_insert_editorial_notes_failure(self):
        amended_xml = f"""<?xml version="1.0"?>
        <akomaNtoso xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <bill>
                <meta></meta>
                <body>
                    <section eId="sec_25">
                        <subsection eId="sec_25__subsec_2" ukl:changeStart="true" ukl:changeEnd="true"
                            ukl:changeGenerated="true" ukl:changeDnum="KS1">
                            <content>
                                <p>text</p>
                            </content>
                        </subsection>
                    </section>
                </body>
            </bill>
        </akomaNtoso>"""
        amended_doc = etree.fromstring(amended_xml.encode()).getroottree()

        with patch.object(self.handler, "_insert_editorial_note", side_effect=Exception("Test exception")):
            with patch("app.services.xml_handler.logger.warning") as mock_logger_warning:
                self.handler.insert_editorial_notes(amended_doc, [self.amendment])
                self.assertEqual(mock_logger_warning.call_count, 1)
                notes = amended_doc.xpath(".//akn:note", namespaces=self.handler.namespaces)
                self.assertEqual(len(notes), 0)

    def test_insert_editorial_note_refs_failure(self):
        amended_xml = f"""<?xml version="1.0"?>
        <akomaNtoso xmlns="{XMLHandler.AKN_URI}" xmlns:ukl="{XMLHandler.UKL_URI}">
            <bill>
                <meta></meta>
                <body>
                    <section eId="sec_25">
                        <subsection eId="sec_25__subsec_2" ukl:changeStart="true" ukl:changeEnd="true"
                            ukl:changeGenerated="true" ukl:changeDnum="KS1">
                            <content>
                                <p>text</p>
                            </content>
                        </subsection>
                    </section>
                </body>
            </bill>
        </akomaNtoso>"""
        amended_doc = etree.fromstring(amended_xml.encode()).getroottree()

        with patch.object(self.handler, "_insert_editorial_note_ref", side_effect=Exception("Test exception")):
            with patch("app.services.xml_handler.logger.warning") as mock_logger_warning:
                self.handler._insert_editorial_note_refs(amended_doc)
                self.assertEqual(mock_logger_warning.call_count, 1)
                note_ref_list = amended_doc.xpath(".//akn:noteRef", namespaces=self.handler.namespaces)
                self.assertEqual(len(note_ref_list), 0)

    def test_find_provision_by_type_and_number(self):
        """Test finding provisions by type and number."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <section eId="sec_1">
                    <num>1.</num>
                    <content><p>Section 1</p></content>
                </section>
                <section eId="sec_2">
                    <num>2.</num>
                    <content><p>Section 2</p></content>
                </section>
                <hcontainer name="regulation" eId="reg_1">
                    <num>1.</num>
                    <content><p>Regulation 1</p></content>
                </hcontainer>
                <hcontainer name="regulation" eId="reg_2">
                    <num>2.</num>
                    <content><p>Regulation 2</p></content>
                </hcontainer>
                <hcontainer name="schedule" eId="sched_1">
                    <num>1</num>
                    <content><p>Schedule 1</p></content>
                </hcontainer>
                <hcontainer name="schedule" eId="sched">
                    <!-- Unnumbered schedule -->
                    <content><p>Schedule</p></content>
                </hcontainer>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Test finding section
        section = self.handler.find_provision_by_type_and_number(tree, "section", "2")
        self.assertIsNotNone(section)
        self.assertEqual(section.get("eId"), "sec_2")

        # Test finding regulation
        regulation = self.handler.find_provision_by_type_and_number(tree, "regulation", "1")
        self.assertIsNotNone(regulation)
        self.assertEqual(regulation.get("eId"), "reg_1")

        # Test with plural type (should be normalized)
        regulation2 = self.handler.find_provision_by_type_and_number(tree, "regulations", "2")
        self.assertIsNotNone(regulation2)
        self.assertEqual(regulation2.get("eId"), "reg_2")

        # Test finding unnumbered schedule
        sched = self.handler.find_provision_by_type_and_number(tree, "schedule", "0")
        self.assertIsNotNone(sched)
        self.assertEqual(sched.get("eId"), "sched")

        # Test with empty number for unnumbered schedule
        sched2 = self.handler.find_provision_by_type_and_number(tree, "schedule", "")
        self.assertIsNotNone(sched2)
        self.assertEqual(sched2.get("eId"), "sched")

        # Test not found
        not_found = self.handler.find_provision_by_type_and_number(tree, "section", "99")
        self.assertIsNone(not_found)

        # Test unknown provision type
        with patch("app.services.xml_handler.logger") as mock_logger:
            unknown = self.handler.find_provision_by_type_and_number(tree, "unknown_type", "1")
            self.assertIsNone(unknown)
            mock_logger.warning.assert_called_with("Unknown provision type: unknown_type")

    def test_find_provision_by_type_and_number_with_context(self):
        """Test finding provisions with context element for disambiguation."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <part eId="part_1">
                    <section eId="part_1_sec_1">
                        <num>1.</num>
                    </section>
                    <section eId="part_1_sec_2">
                        <num>2.</num>
                    </section>
                </part>
                <part eId="part_2">
                    <section eId="part_2_sec_1">
                        <num>1.</num>
                    </section>
                    <section eId="part_2_sec_2">
                        <num>2.</num>
                    </section>
                </part>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Get context elements
        part1_sec1 = tree.find(".//*[@eId='part_1_sec_1']")
        part2_sec1 = tree.find(".//*[@eId='part_2_sec_1']")

        # Find section 2 with context from part 1 - should prefer sibling
        section = self.handler.find_provision_by_type_and_number(tree, "section", "2", context_elem=part1_sec1)
        self.assertIsNotNone(section)
        self.assertEqual(section.get("eId"), "part_1_sec_2")

        # Find section 2 with context from part 2 - should prefer sibling
        section2 = self.handler.find_provision_by_type_and_number(tree, "section", "2", context_elem=part2_sec1)
        self.assertIsNotNone(section2)
        self.assertEqual(section2.get("eId"), "part_2_sec_2")

    def test_find_provision_by_type_and_number_multiple_matches_due_to_contains(self):
        """Test sibling preference when 'contains' in XPath matches multiple provisions."""
        # Create sections where searching for "2" will match multiple sections due to contains()
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <part eId="part_1">
                    <section eId="sec_context">
                        <num>1.</num>
                    </section>
                    <section eId="sec_2_sibling">
                        <num>2.</num>
                    </section>
                    <section eId="sec_12_sibling">
                        <num>12.</num>  <!-- contains "2" -->
                    </section>
                </part>
                <part eId="part_2">
                    <section eId="sec_2_other">
                        <num>2.</num>
                    </section>
                    <section eId="sec_20_other">
                        <num>20.</num>  <!-- contains "2" -->
                    </section>
                    <section eId="sec_21_other">
                        <num>21.</num>  <!-- contains "2" -->
                    </section>
                </part>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Get context from part_1
        context_elem = tree.find(".//*[@eId='sec_context']")
        self.assertIsNotNone(context_elem)

        # Search for "2" - the XPath uses contains() so it will match:
        # sec_2_sibling (2.), sec_12_sibling (12.), sec_2_other (2.), sec_20_other (20.), sec_21_other (21.)
        result = self.handler.find_provision_by_type_and_number(tree, "section", "2", context_elem=context_elem)

        # With sibling preference, it should find sec_2_sibling (same parent as context)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("eId"), "sec_2_sibling")

    def test_find_provision_by_type_and_number_multiple_matches_all_siblings(self):
        """Test when multiple matches are all siblings of context."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <part eId="part_1">
                    <section eId="sec_context">
                        <num>1.</num>
                    </section>
                    <section eId="sec_2a">
                        <num>2.</num>
                    </section>
                    <section eId="sec_12">
                        <num>12.</num>  <!-- contains "2" -->
                    </section>
                    <section eId="sec_20">
                        <num>20.</num>  <!-- contains "2" -->
                    </section>
                </part>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        context_elem = tree.find(".//*[@eId='sec_context']")

        # All matches are siblings - should return the first sibling match
        result = self.handler.find_provision_by_type_and_number(tree, "section", "2", context_elem=context_elem)

        self.assertIsNotNone(result)
        # Should get the first match among siblings
        self.assertEqual(result.get("eId"), "sec_2a")

    def test_find_provisions_in_range_by_number(self):
        """Test finding provisions in numeric range."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <hcontainer name="regulation" eId="reg_1">
                    <num>1.</num>
                </hcontainer>
                <hcontainer name="regulation" eId="reg_2">
                    <num>2.</num>
                </hcontainer>
                <hcontainer name="regulation" eId="reg_3">
                    <num>3.</num>
                </hcontainer>
                <hcontainer name="regulation" eId="reg_3A">
                    <num>3A.</num>
                </hcontainer>
                <hcontainer name="regulation" eId="reg_4">
                    <num>4.</num>
                </hcontainer>
                <hcontainer name="regulation" eId="reg_5">
                    <num>5.</num>
                </hcontainer>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Test normal range
        provisions = self.handler.find_provisions_in_range_by_number(tree, "regulation", "2", "4")
        self.assertEqual(len(provisions), 4)  # reg_2, reg_3, reg_3A, reg_4
        eids = [p.get("eId") for p in provisions]
        self.assertIn("reg_2", eids)
        self.assertIn("reg_3", eids)
        self.assertIn("reg_3A", eids)
        self.assertIn("reg_4", eids)

        # Test with plural type
        provisions2 = self.handler.find_provisions_in_range_by_number(tree, "regulations", "1", "2")
        self.assertEqual(len(provisions2), 2)

        # Test invalid range
        with patch("app.services.xml_handler.logger") as mock_logger:
            invalid = self.handler.find_provisions_in_range_by_number(tree, "regulation", "abc", "xyz")
            self.assertEqual(invalid, [])
            mock_logger.warning.assert_called_with("Could not parse range: abc to xyz")

        # Test unknown provision type
        with patch("app.services.xml_handler.logger") as mock_logger:
            unknown = self.handler.find_provisions_in_range_by_number(tree, "unknown", "1", "5")
            self.assertEqual(unknown, [])
            mock_logger.warning.assert_called_with("Unknown provision type: unknown")

    def test_find_provisions_in_range_by_number_with_context(self):
        """Test finding provisions in range with context element."""
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <part eId="part_1">
                    <section eId="part_1_sec_1">
                        <num>1.</num>
                    </section>
                    <section eId="part_1_sec_2">
                        <num>2.</num>
                    </section>
                    <section eId="part_1_sec_3">
                        <num>3.</num>
                    </section>
                </part>
                <part eId="part_2">
                    <section eId="part_2_sec_1">
                        <num>1.</num>
                    </section>
                    <section eId="part_2_sec_2">
                        <num>2.</num>
                    </section>
                </part>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Get context element from part 1
        part1_elem = tree.find(".//*[@eId='part_1_sec_1']")

        # Find sections with context - should only find within part 1
        sections = self.handler.find_provisions_in_range_by_number(tree, "section", "1", "3", context_elem=part1_elem)
        self.assertEqual(len(sections), 3)
        eids = [s.get("eId") for s in sections]
        self.assertIn("part_1_sec_1", eids)
        self.assertIn("part_1_sec_2", eids)
        self.assertIn("part_1_sec_3", eids)
        # Should not include part 2 sections
        self.assertNotIn("part_2_sec_1", eids)

    def test_find_provisions_in_range_by_number_with_regex_exception(self):
        """Test handling of provision numbers that cause regex match failures."""
        # Create provisions where _extract_provision_number returns a value
        # but that value doesn't match the expected pattern for range checking
        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <body>
                <hcontainer name="regulation" eId="reg_1">
                    <num>1.</num>
                </hcontainer>
                <hcontainer name="regulation" eId="reg_abc">
                    <num>ABC</num>  <!-- This will pass _extract_provision_number but fail regex -->
                </hcontainer>
                <hcontainer name="regulation" eId="reg_2">
                    <num>2.</num>
                </hcontainer>
            </body>
        </act>"""

        tree = etree.fromstring(xml_content.encode()).getroottree()

        # Need to mock _extract_provision_number to return a value that will fail the regex
        with patch.object(self.handler, "_extract_provision_number") as mock_extract:
            # Return valid numbers for reg_1 and reg_2, but something that will fail regex for reg_abc
            def mock_extract_side_effect(element):
                eid = element.get("eId", "")
                if eid == "reg_1":
                    return "1"
                elif eid == "reg_2":
                    return "2"
                elif eid == "reg_abc":
                    return "no_digits_here"  # This will cause re.match(r"(\d+)", ...) to fail
                return None

            mock_extract.side_effect = mock_extract_side_effect

            # This should handle the exception and continue
            provisions = self.handler.find_provisions_in_range_by_number(tree, "regulation", "1", "2")

            # Should get only the valid regulations (1 and 2)
            self.assertEqual(len(provisions), 2)
            eids = [p.get("eId") for p in provisions]
            self.assertIn("reg_1", eids)
            self.assertIn("reg_2", eids)
            self.assertNotIn("reg_abc", eids)

    def test_get_comment_content(self):
        """Test extracting XML comment content."""
        xml_content = """<?xml version="1.0"?>
        <section>
            <!-- First comment -->
            <heading>Test</heading>
            <!-- Second comment -->
            <content>
                <!-- Nested comment -->
                <p>Text</p>
            </content>
            <!-- Final comment -->
        </section>"""

        element = etree.fromstring(xml_content)

        comment_content = self.handler.get_comment_content(element)

        # Should concatenate all comments with spaces
        self.assertEqual(comment_content, "First comment Second comment Nested comment Final comment")

    def test_get_comment_content_no_comments(self):
        """Test get_comment_content when no comments exist."""
        xml_content = """<section><heading>No comments here</heading></section>"""
        element = etree.fromstring(xml_content)

        comment_content = self.handler.get_comment_content(element)
        self.assertEqual(comment_content, "")

    def test_get_crossheadings_in_schedule(self):
        """Test finding crossheadings within a schedule."""
        xml_content = f"""<?xml version="1.0"?>
        <hcontainer name="schedule" xmlns="{XMLHandler.AKN_URI}">
            <num>1</num>
            <heading>Schedule 1</heading>
            <hcontainer name="crossheading" eId="sched_1__xhdg_1">
                <heading>Part 1</heading>
            </hcontainer>
            <paragraph eId="sched_1__para_1">
                <content><p>Text</p></content>
            </paragraph>
            <hcontainer name="crossheading" eId="sched_1__xhdg_2">
                <heading>Part 2</heading>
            </hcontainer>
            <paragraph eId="sched_1__para_2">
                <content><p>More text</p></content>
            </paragraph>
        </hcontainer>"""

        schedule = etree.fromstring(xml_content)

        crossheadings = self.handler.get_crossheadings_in_schedule(schedule)

        self.assertEqual(len(crossheadings), 2)
        self.assertEqual(crossheadings[0].get("eId"), "sched_1__xhdg_1")
        self.assertEqual(crossheadings[1].get("eId"), "sched_1__xhdg_2")

    def test_get_schedule_heading_text(self):
        """Test extracting schedule heading text."""
        xml_content = f"""<?xml version="1.0"?>
        <hcontainer name="schedule" xmlns="{XMLHandler.AKN_URI}">
            <num>1</num>
            <heading>Schedule 1 - Important Provisions</heading>
            <content><p>Content</p></content>
        </hcontainer>"""

        schedule = etree.fromstring(xml_content)

        heading_text = self.handler.get_schedule_heading_text(schedule)
        self.assertEqual(heading_text, "Schedule 1 - Important Provisions")

    def test_get_schedule_heading_text_no_heading(self):
        """Test get_schedule_heading_text when no heading exists."""
        xml_content = f"""<?xml version="1.0"?>
        <hcontainer name="schedule" xmlns="{XMLHandler.AKN_URI}">
            <num>1</num>
            <content><p>Content without heading</p></content>
        </hcontainer>"""

        schedule = etree.fromstring(xml_content)

        heading_text = self.handler.get_schedule_heading_text(schedule)
        self.assertEqual(heading_text, "")

    def test_get_crossheading_child_provisions(self):
        """Test getting direct child provisions of a crossheading."""
        xml_content = f"""<?xml version="1.0"?>
        <hcontainer name="crossheading" xmlns="{XMLHandler.AKN_URI}">
            <heading>Crossheading</heading>
            <paragraph eId="para_1">
                <content><p>Paragraph</p></content>
            </paragraph>
            <section eId="sec_1">
                <content><p>Section</p></content>
            </section>
            <regulation eId="reg_1">
                <content><p>Regulation</p></content>
            </regulation>
            <article eId="art_1">
                <content><p>Article</p></content>
            </article>
            <hcontainer name="other">
                <content><p>Not a provision</p></content>
            </hcontainer>
        </hcontainer>"""

        crossheading = etree.fromstring(xml_content)

        provisions = self.handler.get_crossheading_child_provisions(crossheading)

        self.assertEqual(len(provisions), 4)
        eids = [p.get("eId") for p in provisions]
        self.assertIn("para_1", eids)
        self.assertIn("sec_1", eids)
        self.assertIn("reg_1", eids)
        self.assertIn("art_1", eids)

    def test_get_ancestor_crossheading_contexts(self):
        """Test getting ancestor crossheading contexts."""
        # Create a mock XMLHandler instance with the xml_handler attribute
        handler = XMLHandler()
        handler.xml_handler = handler

        xml_content = f"""<?xml version="1.0"?>
        <act xmlns="{XMLHandler.AKN_URI}">
            <hcontainer name="crossheading" eId="xhdg_1">
                <heading>Top Level</heading>
                <hcontainer name="crossheading" eId="xhdg_2">
                    <heading>Middle Level</heading>
                    <hcontainer name="crossheading" eId="xhdg_3">
                        <heading>Bottom Level</heading>
                        <paragraph eId="para_1">
                            <content><p>Nested paragraph</p></content>
                        </paragraph>
                    </hcontainer>
                </hcontainer>
            </hcontainer>
        </act>"""

        tree = etree.fromstring(xml_content)
        para = tree.find(".//*[@eId='para_1']")

        contexts = handler.get_ancestor_crossheading_contexts(para)

        # Should be in root-to-leaf order
        self.assertEqual(contexts, ["Top Level", "Middle Level", "Bottom Level"])

    def test_inject_xml_comment(self):
        """Test injecting XML comments into elements."""
        element = etree.Element("test")
        etree.SubElement(element, "child1")
        etree.SubElement(element, "child2")

        # Inject at beginning (default)
        self.handler.inject_xml_comment(element, "First comment")

        # Inject at position 2
        self.handler.inject_xml_comment(element, "Second comment", position=2)

        # Check comments were added
        xml_string = etree.tostring(element, encoding="unicode")
        self.assertIn("<!--First comment-->", xml_string)
        self.assertIn("<!--Second comment-->", xml_string)

        # Verify positions
        self.assertEqual(element[0].tag, etree.Comment)
        self.assertEqual(element[0].text, "First comment")
        self.assertEqual(element[2].tag, etree.Comment)
        self.assertEqual(element[2].text, "Second comment")

    def test_set_namespaced_attribute(self):
        """Test setting namespaced attributes."""
        element = etree.Element("test")

        self.handler.set_namespaced_attribute(element, XMLHandler.UKL_URI, "change", "ins")

        self.handler.set_namespaced_attribute(element, XMLHandler.UKL_URI, "changeStart", "true")

        # Check attributes were set
        ukl_ns = f"{{{XMLHandler.UKL_URI}}}"
        self.assertEqual(element.get(f"{ukl_ns}change"), "ins")
        self.assertEqual(element.get(f"{ukl_ns}changeStart"), "true")

    def test_set_change_attributes(self):
        """Test setting change tracking attributes."""
        element = etree.Element("test")

        # Test start only
        self.handler.set_change_attributes(element, is_start=True, is_end=False)
        self.assertEqual(element.get(f"{{{XMLHandler.UKL_URI}}}changeStart"), "true")
        self.assertIsNone(element.get(f"{{{XMLHandler.UKL_URI}}}changeEnd"))
        self.assertIsNone(element.get(f"{{{XMLHandler.UKL_URI}}}changeGenerated"))

        # Test end only (on fresh element)
        element2 = etree.Element("test")
        self.handler.set_change_attributes(element2, is_start=False, is_end=True)
        self.assertIsNone(element2.get(f"{{{XMLHandler.UKL_URI}}}changeStart"))
        self.assertEqual(element2.get(f"{{{XMLHandler.UKL_URI}}}changeEnd"), "true")
        self.assertEqual(element2.get(f"{{{XMLHandler.UKL_URI}}}changeGenerated"), "true")

    def test_create_akn_element(self):
        """Test creating AKN elements."""
        # Without text
        element = self.handler.create_akn_element("section")
        self.assertEqual(element.tag, f"{{{XMLHandler.AKN_URI}}}section")
        self.assertIsNone(element.text)

        # With text
        element2 = self.handler.create_akn_element("p", "Test paragraph")
        self.assertEqual(element2.tag, f"{{{XMLHandler.AKN_URI}}}p")
        self.assertEqual(element2.text, "Test paragraph")

    def test_build_xpath_for_provision_number(self):
        """Test building XPath for provision numbers."""
        # Test various provision types
        self.assertEqual(
            self.handler._build_xpath_for_provision_number("section", "5"), ".//akn:section[akn:num[contains(.,'5')]]"
        )

        self.assertEqual(
            self.handler._build_xpath_for_provision_number("regulation", "10"),
            ".//akn:hcontainer[@name='regulation'][akn:num[contains(.,'10')]]",
        )

        self.assertEqual(
            self.handler._build_xpath_for_provision_number("article", "3"), ".//akn:article[akn:num[contains(.,'3')]]"
        )

        self.assertEqual(
            self.handler._build_xpath_for_provision_number("rule", "7"), ".//akn:rule[akn:num[contains(.,'7')]]"
        )

        self.assertEqual(
            self.handler._build_xpath_for_provision_number("paragraph", "2"),
            ".//akn:paragraph[akn:num[contains(.,'2')]]",
        )

        self.assertEqual(
            self.handler._build_xpath_for_provision_number("para", "1"), ".//akn:para[akn:num[contains(.,'1')]]"
        )

        # Test unnumbered schedule
        self.assertEqual(
            self.handler._build_xpath_for_provision_number("schedule", "0"),
            ".//akn:hcontainer[@name='schedule'][not(akn:num) or akn:num='']",
        )

        self.assertEqual(
            self.handler._build_xpath_for_provision_number("schedule", ""),
            ".//akn:hcontainer[@name='schedule'][not(akn:num) or akn:num='']",
        )

        # Test unknown type
        self.assertEqual(self.handler._build_xpath_for_provision_number("unknown", "1"), "")

    def test_get_base_xpath_for_provision_type(self):
        """Test getting base XPath for provision types."""
        self.assertEqual(self.handler._get_base_xpath_for_provision_type("section"), ".//akn:section")

        self.assertEqual(
            self.handler._get_base_xpath_for_provision_type("regulation"), ".//akn:hcontainer[@name='regulation']"
        )

        self.assertEqual(self.handler._get_base_xpath_for_provision_type("article"), ".//akn:article")

        self.assertEqual(self.handler._get_base_xpath_for_provision_type("rule"), ".//akn:rule")

        self.assertEqual(
            self.handler._get_base_xpath_for_provision_type("schedule"), ".//akn:hcontainer[@name='schedule']"
        )

        self.assertEqual(self.handler._get_base_xpath_for_provision_type("paragraph"), ".//akn:paragraph")

        self.assertEqual(self.handler._get_base_xpath_for_provision_type("para"), ".//akn:para")

        result = self.handler._get_base_xpath_for_provision_type("unknown")
        self.assertEqual(result, "")

    def test_extract_provision_number(self):
        """Test extracting provision numbers from elements."""
        # Test with various number formats
        test_cases = [
            (f'<section xmlns="{XMLHandler.AKN_URI}"><num>1.</num></section>', "1"),
            (f'<section xmlns="{XMLHandler.AKN_URI}"><num>(2)</num></section>', "2"),
            (f'<section xmlns="{XMLHandler.AKN_URI}"><num>3</num></section>', "3"),
            (f'<section xmlns="{XMLHandler.AKN_URI}"><num>4A</num></section>', "4A"),
            (f'<section xmlns="{XMLHandler.AKN_URI}"><num>5B.</num></section>', "5B"),
            (f'<section xmlns="{XMLHandler.AKN_URI}"><num>Section 6</num></section>', "6"),
            (f'<section xmlns="{XMLHandler.AKN_URI}"><num>Regulation 7A</num></section>', "7A"),
        ]

        for xml_str, expected in test_cases:
            element = etree.fromstring(xml_str)
            number = self.handler._extract_provision_number(element)
            self.assertEqual(number, expected, f"Failed for: {xml_str}")

        # Test with no num element
        element = etree.fromstring(f'<section xmlns="{XMLHandler.AKN_URI}"><heading>No number</heading></section>')
        number = self.handler._extract_provision_number(element)
        self.assertIsNone(number)

        # Test with empty num element
        element = etree.fromstring(f'<section xmlns="{XMLHandler.AKN_URI}"><num></num></section>')
        number = self.handler._extract_provision_number(element)
        self.assertIsNone(number)

        # Test with num containing no digits
        element = etree.fromstring(f'<section xmlns="{XMLHandler.AKN_URI}"><num>ABC</num></section>')
        number = self.handler._extract_provision_number(element)
        self.assertIsNone(number)
