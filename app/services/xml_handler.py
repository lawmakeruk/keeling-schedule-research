# app/services/xml_handler.py
"""
Handles all XML operations for legislative documents including loading, saving,
normalisation, and element manipulation.
"""

import copy
import re
import threading
from typing import Optional, List, Tuple, Dict, Any
from lxml import etree
from ..logging.debug_logger import get_logger, event, EventType as EVT
from ..models.amendments import Amendment
from ..models.legi_element import LegiElement

logger = get_logger(__name__)


class XMLHandler:
    """Manages XML document operations with thread-safe dnum allocation."""

    # Namespace URIs
    UKL_URI = "https://www.legislation.gov.uk/namespaces/UK-AKN"
    AKN_URI = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
    XMLNS_URI = "http://www.w3.org/2000/xmlns/"

    def __init__(self):
        """Initialise the XML handler with namespace configuration."""
        # Register namespace once
        etree.register_namespace("ukl", self.UKL_URI)

        self.namespaces = {
            "akn": self.AKN_URI,
            "ukl": self.UKL_URI,
        }

        # Thread-safe dnum counter
        self._dnum_counter = 0
        self._dnum_lock = threading.Lock()

    # ==================== Core XML Operations ====================

    def load_xml(self, filepath: str) -> etree.ElementTree:
        """
        Load and normalise an XML file.

        Args:
            filepath: Path to XML file

        Returns:
            Normalised XML tree

        Raises:
            FileNotFoundError: If file cannot be found
            etree.ParseError: If XML is malformed
        """
        try:
            tree = etree.parse(filepath)
            self.normalise_namespaces(tree)
            self.normalise_eids(tree)
            return tree
        except etree.ParseError as e:
            event(logger, EVT.XML_PARSE_ERROR, "Failed to parse XML file", filepath=filepath, error=str(e))
            raise

    def save_xml(self, tree: etree.ElementTree, filepath: str) -> None:
        """
        Save XML tree to file.

        Args:
            tree: XML tree to save
            filepath: Where to save the file

        Raises:
            IOError: If file cannot be written
        """
        try:
            tree.write(filepath, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            logger.error(f"Failed to save XML to {filepath}: {e}")
            raise IOError(f"Failed to save XML: {e}")

    def parse_xml_string(self, xml_string: str, ensure_namespaces: bool = True) -> etree.Element:
        """
        Parse an XML string, optionally ensuring required namespaces are declared.

        Args:
            xml_string: XML string to parse
            ensure_namespaces: If True, wrap XML that uses ukl: without declaring it

        Returns:
            Parsed XML element

        Raises:
            etree.XMLSyntaxError: If XML is malformed
            ValueError: If XML cannot be parsed
        """
        try:
            # Check if we need to add namespace declarations
            if ensure_namespaces and "ukl:" in xml_string and "xmlns:ukl" not in xml_string:
                # Wrap with namespace declarations
                wrapped = (
                    f'<wrapper xmlns:akn="{self.namespaces["akn"]}" '
                    f'xmlns:ukl="{self.namespaces["ukl"]}">{xml_string}</wrapper>'
                )
                element = etree.fromstring(wrapped.encode("utf-8"))
                # Return the first child (unwrapped)
                if len(element) > 0:
                    return element[0]
                else:
                    raise ValueError("Wrapped XML has no child elements")
            else:
                # Parse directly
                return etree.fromstring(xml_string.encode("utf-8"))

        except etree.XMLSyntaxError as e:
            # Re-raise with original XML for debugging
            logger.debug(f"Failed to parse XML: {xml_string[:500]}...")
            event(logger, EVT.XML_PARSE_ERROR, "Failed to parse XML string", error=str(e), xml_preview=xml_string[:200])
            raise
        except Exception as e:
            event(logger, EVT.XML_PARSE_ERROR, "Unexpected error parsing XML", error=str(e))
            raise ValueError(f"Failed to parse XML: {str(e)}")

    def element_to_string(self, element: etree.Element) -> str:
        """
        Convert an element to its XML string representation.

        Args:
            element: Element to convert

        Returns:
            UTF-8 encoded XML string
        """
        return etree.tostring(element, encoding="utf-8").decode("utf-8")

    def simplify_amending_bill(self, amending_bill: etree.ElementTree) -> None:
        """
        Simplifies the amending bill so that more standardised input gets passed to the LLM
        Args:
            amending_bill: Element to convert
        """
        # Only simply the body
        root = LegiElement(amending_bill.getroot())
        body = root.get_descendant(".//akn:body")

        # Delete all processing instructions
        for pi in body.get_descendants(".//processing-instruction()"):
            pi.delete_element()

        parenthetical_descriptions_pattern = re.compile(r" \(.*?\)")
        for element in body.get_descendants(".//akn:ref | .//akn:rref | .//akn:mref"):
            # Remove all parenthetical descriptions
            element.set_tail(parenthetical_descriptions_pattern.sub("", element.get_tail()))
            # Unwrap references
            element.unwrap_element()

        # Simplify quoted structure content
        for quoted_structure in body.get_descendants(".//akn:quotedStructure"):
            for child in quoted_structure.get_children():
                for grand_child in child.get_children():
                    grand_child.delete_element()

        # Replace text in quotes with a placeholder
        self.standardise_text_in_quotes(body.element)

    def standardise_text_in_quotes(self, start_element: etree.Element):
        """
        Replaces text within curly quotes with a placeholder in the given element and its descendants.

        Args:
            start_element: Element to process
        """
        # Regular expression to match text within curly quotes
        quoted_text_pattern = re.compile(r"“.*?”")

        # Recursive function to process and clean up nodes
        def remove_text_within_quotes_recursively(element):
            placeholder = "“quote”"
            # Iterate through each node's text and tail
            if element.text:
                element.text = quoted_text_pattern.sub(placeholder, element.text)
            if element.tail:
                element.tail = quoted_text_pattern.sub(placeholder, element.tail)

            # Recursively process child elements
            for child in element:
                remove_text_within_quotes_recursively(child)

        # Start processing from the root element
        remove_text_within_quotes_recursively(start_element)

    # ==================== Normalisation Methods ====================

    def normalise_namespaces(self, tree: etree.ElementTree) -> None:
        """
        Normalise namespaces in the XML tree:
        - Remove duplicate 'uk:' prefix bound to UKL_URI
        - Rename elements using old prefix to 'ukl:'
        - Ensure canonical xmlns:ukl appears exactly once

        Args:
            tree: XML tree to normalise (modified in place)
        """
        # Register ukl as the preferred prefix for this namespace
        etree.register_namespace("ukl", self.UKL_URI)

        root = tree.getroot()
        have_dup = root.nsmap.get("uk") == self.UKL_URI

        if have_dup:
            # We need to force a complete namespace reassignment
            # The most reliable way is to serialise and reparse

            # First, change all uk: prefixed elements to use full namespace URI
            for el in root.iter():
                if el.prefix == "uk":
                    local_name = etree.QName(el).localname
                    el.tag = f"{{{self.UKL_URI}}}{local_name}"

            # Use element_to_string helper for serialisation
            xml_str = self.element_to_string(root)

            # Replace xmlns:uk with xmlns:ukl
            xml_str = xml_str.replace(f'xmlns:uk="{self.UKL_URI}"', f'xmlns:ukl="{self.UKL_URI}"')

            # Also need to update any remaining uk: prefixes in the XML
            # (though they should be gone after changing tags to full URIs)
            xml_str = xml_str.replace("uk:", "ukl:")

            # Use parse_xml_string helper for reparsing
            new_root = self.parse_xml_string(xml_str, ensure_namespaces=False)
            tree._setroot(new_root)

        # If the UK-AKN URI is not bound at all, add it
        else:
            if self.UKL_URI not in root.nsmap.values():
                # Add a temporary attribute to force namespace declaration
                dummy_attr = f"{{{self.UKL_URI}}}__fix"
                root.set(dummy_attr, "1")
                etree.cleanup_namespaces(root)
                root.attrib.pop(dummy_attr, None)

    def normalise_eids(self, tree: etree.ElementTree) -> None:
        """
        Normalise eId attributes:
        - Convert structural keywords to lowercase
        - Preserve case of alphanumeric identifiers

        Args:
            tree: XML tree to normalise (modified in place)
        """
        # Keywords ordered longest-first to prevent partial matches
        keywords = "subsubpara|subpara|subsec|sched|apndx|anx|chp|para|rule|sec|pt|st|reg|art"

        # Use lookbehind/lookahead instead of word boundaries since underscores are word characters
        eid_pattern = re.compile(rf"(?<![A-Za-z0-9])({keywords})(?![A-Za-z])", re.IGNORECASE)

        for el in tree.getroot().iter():
            eid = el.get("eId")
            if eid:
                # Single pass: lowercase only the recognised keywords
                normalised = eid_pattern.sub(lambda match: match.group(1).lower(), eid)
                el.set("eId", normalised)

    # ==================== Element Finding Methods ====================

    def find_element_by_eid(self, tree: etree.ElementTree, eid: str) -> Optional[etree.Element]:
        """
        Find element by eId attribute.

        Args:
            tree: XML tree to search
            eid: Element ID to find

        Returns:
            Found element or None
        """
        root = tree.getroot()
        return root.find(f".//*[@eId='{eid}']", self.namespaces)

    def find_element_by_eid_components(self, tree: etree.ElementTree, eid: str) -> Optional[etree.Element]:
        """
        Find element by eId attribute. First, attempt to find element by eId attribute. If that fails,
        traverse the XML structure to find the element using the provision numbers from the eId.

        Args:
            tree: XML tree to search
            eid: Element ID to find

        Returns:
            Found element or None
        """
        try:
            # Try to find the element using the whole eid
            element = self.find_element_by_eid(tree, eid)
            if element is not None:
                logger.info(f"Found element via component traversal: requested '{eid}', found '{element.get('eId')}'")
                return element
            # Otherwise, attempt to traverse through the XML structure using the provision nums to find the element
            root = tree.getroot()
            parts = eid.split("__")
            element = root.find(f".//*[@eId='{parts[0]}']", self.namespaces)
            for i, part in enumerate(parts[1:]):
                if element is None:
                    logger.debug(f"Could not find element: requested '{eid}', current portion '{parts[i]}'")
                    return None
                # Find the child of the current element whose @eId ends with the target num
                # lxml uses XPath 1.0, so it doesn't support ends-with()
                num = part.split("_")[1]
                element_list = element.xpath(
                    f"./*[@eId and substring(@eId, string-length(@eId) - string-length('_{num}') + 1) = '_{num}']",
                    namespaces=self.namespaces,
                )
                element = element_list[0] if element_list else None

            if element is not None:
                logger.info(f"Found element via component traversal: requested '{eid}', found '{element.get('eId')}'")
                return element
            else:
                logger.debug(f"Could not find element: requested '{eid}', current portion '{parts[-1]}'")
                return None
        except Exception as e:
            logger.error(f"Failed to find element with eId={eid}: {e}")
            return None

    def find_element_by_eid_with_fallback(self, tree: etree.ElementTree, eid: str) -> Optional[etree.Element]:
        """
        Find element by eId or its nearest ancestor.
        Useful for error comment placement when exact element doesn't exist.

        Args:
            tree: XML tree to search
            eid: Element ID to find

        Returns:
            Found element, nearest ancestor, or preface as fallback
        """
        # Try exact match first
        element = self.find_element_by_eid(tree, eid)
        if element is not None:
            return element

        # Use get_ancestor_eids to get all ancestors in order
        ancestors = self.get_ancestor_eids(eid, include_self=False)

        # Try each ancestor from most specific to least specific
        for i in range(len(ancestors) - 1, -1, -1):
            element = self.find_element_by_eid(tree, ancestors[i])
            if element is not None:
                return element

        # Fallback to preface
        root = tree.getroot()
        preface = root.find(".//akn:preface", namespaces=self.namespaces)
        return preface

    def find_closest_match_by_eid(self, tree: etree.ElementTree, eid: str) -> Optional[etree.Element]:
        """
        Find element by eId attribute. If it's not found, try removing the last part of the eId until it is found.

        Args:
            tree: XML tree to search
            eid: Element ID to find

        Returns:
            Found element or None
        """
        target = None
        eid_terms = eid.split("__")
        while target is None and eid_terms:
            current_eid = "__".join(eid_terms)
            target = self.find_element_by_eid(tree, current_eid)
            if target is None:
                eid_terms.pop()  # Remove the last term and retry
            else:
                return target

    def find_provisions_containing_text(
        self, tree: etree.ElementTree, search_terms: List[str], exclude_quoted: bool = True
    ) -> List[Tuple[etree.Element, str]]:
        """
        Find all provisions that contain any of the specified search terms.

        Args:
            tree: XML tree to search
            search_terms: List of terms to search for (case-insensitive)
            exclude_quoted: Whether to exclude text in quotedStructure elements

        Returns:
            List of tuples containing (element, eId) for matching provisions
        """
        root = tree.getroot()
        body = root.find(".//akn:body", namespaces=self.namespaces)

        if body is None:
            return []

        # Find all prov1 elements (sections, regulations, articles, rules)
        provision_list = body.xpath(
            ".//*[@class='prov1' and not(ancestor::akn:quotedStructure)]", namespaces=self.namespaces
        )

        # Find schedules
        schedules = body.xpath(
            ".//akn:hcontainer[@name='schedule' and not(ancestor::akn:quotedStructure)]", namespaces=self.namespaces
        )

        # Process schedules
        for schedule in schedules:
            # Check if schedule contains grouping provisions
            has_groups = any("Group" in child.get("class", "") for child in schedule)

            if has_groups:
                self._extract_lowest_grouping_provisions(schedule, provision_list)
            else:
                provision_list.append(schedule)

        # Filter provisions that contain search terms
        matching_provisions = []
        search_terms_lower = [term.lower() for term in search_terms]

        for provision in provision_list:
            text_content = self.get_text_content(provision, exclude_quoted=exclude_quoted).lower()

            if any(term in text_content for term in search_terms_lower):
                eid = provision.get("eId", "")
                matching_provisions.append((provision, eid))

        return matching_provisions

    def find_provisions_in_range(self, start_eid: str, end_eid: str, tree: etree.ElementTree) -> List[str]:
        """
        Find all provisions between start and end eIds (inclusive) in document order.

        Args:
            start_eid: Starting element ID
            end_eid: Ending element ID
            tree: XML tree to search in

        Returns:
            List of eIds for provisions in the range
        """
        logger.debug(f"Finding provisions in range: {start_eid} to {end_eid}")

        # Find elements
        start_elem = self.find_element_by_eid_components(tree, start_eid)
        end_elem = self.find_element_by_eid_components(tree, end_eid)

        if start_elem is None or end_elem is None:
            logger.warning(
                f"Could not find range endpoints: start={start_eid}"
                f"(found: {start_elem is not None}), end={end_eid} (found: {end_elem is not None})"
            )
            return []

        # Extract provision type
        start_eid_confirmed = start_elem.get("eId")
        provision_type = self._extract_provision_type(start_eid_confirmed)
        start_level = len(start_eid_confirmed.split("__"))

        logger.debug(f"Range search: provision_type={provision_type}, level={start_level}")

        # Get all elements in document order
        root = tree.getroot()
        all_elements = list(root.iter())

        # Find indices
        try:
            start_idx = all_elements.index(start_elem)
            end_idx = all_elements.index(end_elem)
            logger.debug(f"Element indices: start={start_idx}, end={end_idx}")
        except ValueError:
            logger.error("Start or end element not found in document tree")
            return []

        # Ensure correct order
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Collect provisions of same type and level
        provisions = []
        for elem in all_elements[start_idx : end_idx + 1]:
            elem_eid = elem.get("eId", "")
            if not elem_eid:
                continue

            elem_type = self._extract_provision_type(elem_eid)
            elem_level = len(elem_eid.split("__"))

            if elem_level == start_level and elem_type == provision_type:
                provisions.append(elem_eid)

        return provisions

    def extract_eid_patterns(self, tree: etree.ElementTree, max_examples: int = 5) -> Dict[str, Any]:
        """
        Extract example eId patterns and naming conventions from the document.

        Analyses the target Act to find real examples of how different provision
        types are identified, helping the LLM generate consistent eIds that match
        the document's actual conventions.

        Args:
            tree: XML tree to extract patterns from
            max_examples: Maximum number of examples to extract per type (default: 5)

        Returns:
            Dictionary containing:
            - examples: Dict mapping provision types to lists of example eIds
            - conventions: Dict of detected naming conventions (e.g., definition suffixes)
        """
        patterns = {
            "examples": {},
            "conventions": {
                "definition_suffix": "",
                "uses_number_prefix": False,
            },
        }

        # Define UK/Scottish-relevant element types with their XPath
        element_queries = {
            # Primary provisions
            "sections": ".//akn:section",
            "subsections": ".//akn:subsection",
            "paragraphs": ".//akn:paragraph | .//akn:level[@class='para1']",
            "subparagraphs": ".//akn:subparagraph | .//akn:level[@class='para2']",
            # Structural divisions
            "parts": ".//akn:part",
            "chapters": ".//akn:chapter",
            "schedules": ".//akn:hcontainer[@name='schedule']",
            # Regulations/Rules specific
            "regulations": ".//akn:hcontainer[@name='regulation']",
            "articles": ".//akn:article",
            "rules": ".//akn:rule",
            # Special elements
            "definitions": ".//akn:hcontainer[@class='definition']",
            "headings": ".//*[akn:heading]",
            "transitional": ".//akn:transitional",
            # Generic
            "levels": ".//akn:level[not(@class='para1' or @class='para2')]",
        }

        # Track counts for logging
        pattern_counts = {}

        # Extract patterns for each type
        for pattern_type, xpath in element_queries.items():
            patterns["examples"][pattern_type] = []
            elements = tree.xpath(xpath, namespaces=self.namespaces)

            for elem in elements[:max_examples]:
                eid = elem.get("eId")
                if eid:
                    patterns["examples"][pattern_type].append(eid)

                    # Special handling for definitions to detect suffix
                    if pattern_type == "definitions" and eid.endswith("_"):
                        patterns["conventions"]["definition_suffix"] = "_"

            # Track non-empty pattern types
            if patterns["examples"][pattern_type]:
                pattern_counts[pattern_type] = len(patterns["examples"][pattern_type])

        # Also check for heading patterns
        heading_examples = []
        for elem_with_heading in tree.xpath(".//*[@eId and akn:heading]", namespaces=self.namespaces)[:max_examples]:
            base_eid = elem_with_heading.get("eId")
            heading_eid = f"{base_eid}__hdg"
            if tree.xpath(f".//*[@eId='{heading_eid}']", namespaces=self.namespaces):
                heading_examples.append(heading_eid)

        if heading_examples:
            patterns["examples"]["headings"] = heading_examples
            pattern_counts["headings"] = len(heading_examples)

        # Log summary of what was found
        if pattern_counts:
            logger.info(f"Extracted eId patterns: {pattern_counts}")
            if patterns["conventions"]["definition_suffix"]:
                logger.info(f"Detected definition suffix convention: '{patterns['conventions']['definition_suffix']}'")
        else:
            logger.warning("No eId patterns found in document")

        return patterns

    def find_provision_by_type_and_number(
        self, tree: etree.ElementTree, prov_type: str, number: str, context_elem: etree.Element = None
    ) -> Optional[etree.Element]:
        """
        Find a provision by its type and number, using context for hints.

        Args:
            tree: XML tree to search
            prov_type: Type like 'regulation', 'section', etc.
            number: The provision number (e.g., '3', '15A')
            context_elem: Optional context element to search near first

        Returns:
            The found element or None
        """
        # Normalise provision type
        prov_type = prov_type.lower().rstrip("s")

        # Convert to LegiElement for cleaner XPath
        if context_elem is not None and context_elem.getparent() is not None:
            search_root = LegiElement(context_elem.getparent())
        else:
            search_root = LegiElement(tree.getroot())

        # Build XPath based on provision type
        xpath = self._build_xpath_for_provision_number(prov_type, number)
        if not xpath:
            logger.warning(f"Unknown provision type: {prov_type}")
            return None

        # Search for matching elements
        matches = search_root.get_descendants(xpath)

        # If multiple matches and we have context, prefer siblings
        if len(matches) > 1 and context_elem is not None:
            parent = context_elem.getparent()
            if parent is not None:
                for match in matches:
                    if match.element.getparent() == parent:
                        return match.element

        return matches[0].element if matches else None

    def find_provisions_in_range_by_number(
        self, tree: etree.ElementTree, prov_type: str, start_num: str, end_num: str, context_elem: etree.Element = None
    ) -> List[etree.Element]:
        """
        Find all provisions of given type in numeric range.

        Args:
            tree: XML tree to search
            prov_type: Type like 'regulation', 'section'
            start_num: Starting number (e.g., '3')
            end_num: Ending number (e.g., '7')
            context_elem: Optional context to search near first

        Returns:
            List of found elements in document order
        """
        provisions = []

        # Parse numbers for range
        try:
            start_base = int(re.match(r"(\d+)", start_num).group(1))
            end_base = int(re.match(r"(\d+)", end_num).group(1))
        except Exception:
            logger.warning(f"Could not parse range: {start_num} to {end_num}")
            return []

        # Find all provisions of this type
        prov_type = prov_type.lower().rstrip("s")

        # Use context if provided
        if context_elem is not None and context_elem.getparent() is not None:
            search_root = LegiElement(context_elem.getparent())
        else:
            search_root = LegiElement(tree.getroot())

        # Get base XPath for provision type
        base_xpath = self._get_base_xpath_for_provision_type(prov_type)
        if not base_xpath:
            logger.warning(f"Unknown provision type: {prov_type}")
            return []

        # Find all provisions of this type
        all_provisions = search_root.get_descendants(base_xpath)

        # Filter by number range
        for legi_elem in all_provisions:
            elem_num = self._extract_provision_number(legi_elem.element)
            if elem_num:
                try:
                    elem_base = int(re.match(r"(\d+)", elem_num).group(1))
                    if start_base <= elem_base <= end_base:
                        provisions.append(legi_elem.element)
                except Exception:
                    continue

        # Sort by document order
        return sorted(provisions, key=lambda e: list(tree.getroot().iter()).index(e))

    # ==================== Change Tracking Methods ====================

    def add_change_markup(
        self,
        element: etree.Element,
        change_type: str,
        is_start: bool = False,
        is_end: bool = False,
        add_dnum: bool = False,
    ) -> None:
        """
        Add ukl:change attributes to an element.

        Args:
            element: Element to modify
            change_type: Type of change (ins, del, insReplace, delReplace)
            is_start: Whether this starts a change span
            is_end: Whether this ends a change span
            add_dnum: Whether to add a dnum to this element
        """
        ukl_ns = f"{{{self.UKL_URI}}}"

        element.set(f"{ukl_ns}change", change_type)

        if is_start:
            element.set(f"{ukl_ns}changeStart", "true")

        if is_end:
            element.set(f"{ukl_ns}changeEnd", "true")
            element.set(f"{ukl_ns}changeGenerated", "true")

        if add_dnum:
            dnum = self.allocate_dnum()
            element.set(f"{ukl_ns}changeDnum", f"KS{dnum}")

    def set_change_attributes(self, element: etree.Element, is_start: bool, is_end: bool) -> None:
        """
        Set change tracking attributes on an element.

        Args:
            element: Element to set attributes on
            is_start: Whether this starts a change
            is_end: Whether this ends a change
        """
        if is_start:
            element.set(f"{{{self.UKL_URI}}}changeStart", "true")

        if is_end:
            element.set(f"{{{self.UKL_URI}}}changeEnd", "true")
            element.set(f"{{{self.UKL_URI}}}changeGenerated", "true")

    def create_akn_element(self, tag: str, text: str = None) -> etree.Element:
        """
        Create an Akoma Ntoso element.

        Args:
            tag: Element tag name (without namespace)
            text: Optional text content

        Returns:
            Created element with AKN namespace
        """
        elem = etree.Element(f"{{{self.AKN_URI}}}{tag}")
        if text:
            elem.text = text
        return elem

    def allocate_dnum(self) -> int:
        """
        Allocate next dnum number (thread-safe).

        Returns:
            Next unique dnum number
        """
        with self._dnum_lock:
            self._dnum_counter += 1
            return self._dnum_counter

    def set_dnum_counter(self, value: int) -> None:
        """
        Set the dnum counter to a specific value (thread-safe).

        Args:
            value: New counter value
        """
        with self._dnum_lock:
            self._dnum_counter = value

    def find_existing_dnums(self, tree: etree.ElementTree) -> int:
        """
        Find the highest existing dnum in the document.

        Args:
            tree: XML tree to search

        Returns:
            Highest dnum number found, or 0 if none
        """
        dnum_elements = tree.xpath(".//*[@ukl:changeGenerated='true' and @ukl:changeDnum]", namespaces=self.namespaces)

        highest = 0
        for element in dnum_elements:
            dnum_attr = element.get(f"{{{self.UKL_URI}}}changeDnum", "")
            match = re.match(r"KS(\d+)", dnum_attr)
            if match:
                num = int(match.group(1))
                highest = max(highest, num)

        return highest

    def renumber_dnums(self, tree: etree.ElementTree, amendments: List["Amendment"]) -> None:
        """
        Renumber all dnums in document order.

        Args:
            tree: XML tree to process (modified in place)
            amendments: The list of amendments applied to the document
        """
        # Find all elements with changeGenerated
        elements = tree.xpath("descendant-or-self::*[@ukl:changeGenerated='true']", namespaces=self.namespaces)

        logger.debug(f"Found {len(elements)} elements to renumber")
        id_to_amendment = {}
        for amendment in amendments:
            id_to_amendment[amendment.amendment_id] = amendment

        # Renumber sequentially
        for i, element in enumerate(elements, 1):
            new_dnum = f"KS{i}"

            # Update the amended element in the XML
            element.set(f"{{{self.UKL_URI}}}changeDnum", new_dnum)

            # Update the amendment objects's dnum
            amendment_id = element.get("amendmentId")
            if amendment_id is not None:
                id_to_amendment[amendment_id].dnum_list.append(new_dnum)

            logger.debug(f"Assigned dnum: {new_dnum} amendment_id={amendment_id}")

        # Update counter for future allocations
        self.set_dnum_counter(len(elements))

    def element_has_change_markup(self, element: etree.Element) -> bool:
        """
        Check if an element or its descendants have change markup.

        Args:
            element: Element to check

        Returns:
            True if change markup is present
        """
        has_change_elements = bool(
            element.xpath("descendant-or-self::akn:ins | descendant-or-self::akn:del", namespaces=self.namespaces)
        )
        has_change_attrs = bool(element.xpath("descendant-or-self::*[@ukl:change]", namespaces=self.namespaces))
        return has_change_elements or has_change_attrs

    def get_change_tracking_attributes(self, element: etree.Element) -> Dict[str, bool]:
        """
        Get the presence of change tracking attributes in an element tree.

        Args:
            element: Root element to check

        Returns:
            Dictionary with keys 'changeStart', 'changeEnd', 'changeGenerated' and boolean values
        """
        return {
            "changeStart": bool(
                element.xpath("descendant-or-self::*[@ukl:changeStart='true']", namespaces=self.namespaces)
            ),
            "changeEnd": bool(
                element.xpath("descendant-or-self::*[@ukl:changeEnd='true']", namespaces=self.namespaces)
            ),
            "changeGenerated": bool(
                element.xpath("descendant-or-self::*[@ukl:changeGenerated='true']", namespaces=self.namespaces)
            ),
        }

    def inject_amendment_id(self, target: etree.Element, amendment: Amendment) -> None:
        """
        Inserts the amendment ID into the XML of the XML change tracking of the
        amendment as a temporary @amendmentId attribute.

        Args:
            target: The XML element containing the change tracking of the amendment.
            amendment: The amendment.
        """
        change_element_list = target.xpath(
            "descendant-or-self::*[@ukl:changeGenerated='true' and not(@amendmentId)]", namespaces=self.namespaces
        )
        if len(change_element_list) > 0:
            for change_element in change_element_list:
                change_element.set("amendmentId", amendment.amendment_id)

    def remove_amendment_ids(self, tree: etree.ElementTree) -> None:
        """
        Delete all instances of the @amendmentId attribute in the XMl document.

        Args:
           tree: The XML document.
        """
        elements = tree.xpath(".//*[@amendmentId]", namespaces=self.namespaces)
        for element in elements:
            del element.attrib["amendmentId"]

    # ==================== Content Extraction Methods ====================

    def get_text_content(self, element: etree.Element, exclude_quoted: bool = True) -> str:
        """
        Get text content from an element, optionally excluding quoted structures.

        Args:
            element: Element to extract text from
            exclude_quoted: Whether to exclude text in quotedStructure elements

        Returns:
            Concatenated text content
        """
        if exclude_quoted:
            text_nodes = element.xpath(".//text()[not(ancestor::akn:quotedStructure)]", namespaces=self.namespaces)
        else:
            text_nodes = element.xpath(".//text()")

        return "".join(text_nodes)

    def get_comment_content(self, element: etree.Element) -> str:
        """
        Get all XML comment content from an element and its descendants.

        Args:
            element: Element to extract comments from

        Returns:
            Concatenated comment text
        """
        comment_nodes = element.xpath(".//comment()")
        return " ".join(node.text.strip() for node in comment_nodes if node.text)

    def get_ancestor_eids(self, eid: str, include_self: bool = False) -> List[str]:
        """
        Get all ancestor eIds from a given eId.

        Args:
            eid: Element ID
            include_self: Whether to include the original eId in the result

        Returns:
            List of ancestor eIds, ordered from root to immediate parent
        """
        parts = eid.split("__") if eid else []
        ancestors = []

        # Build ancestors from root down
        for i in range(1, len(parts)):
            ancestors.append("__".join(parts[:i]))

        if include_self and parts:
            ancestors.append(eid)

        return ancestors

    def get_crossheadings_in_schedule(self, schedule: etree.Element) -> List[etree.Element]:
        """
        Find all crossheading containers within a schedule.

        Args:
            schedule: Schedule element to search within

        Returns:
            List of crossheading hcontainer elements
        """
        return schedule.xpath(".//akn:hcontainer[@name='crossheading']", namespaces=self.namespaces)

    def get_schedule_heading_text(self, schedule: etree.Element) -> str:
        """
        Extract the heading text from a schedule element.

        Args:
            schedule: Schedule element

        Returns:
            Heading text or empty string if no heading found
        """
        heading = schedule.find(".//akn:heading", self.namespaces)
        if heading is not None:
            return self.get_text_content(heading)
        return ""

    def get_crossheading_child_provisions(self, crossheading: etree.Element) -> List[etree.Element]:
        """
        Get direct child provisions of a crossheading.

        Args:
            crossheading: Crossheading hcontainer element

        Returns:
            List of child provision elements (paragraph, section, regulation, article)
        """
        return crossheading.xpath(
            "./akn:paragraph | ./akn:section | ./akn:regulation | ./akn:article", namespaces=self.namespaces
        )

    def get_ancestor_crossheading_contexts(self, element: etree.Element) -> List[str]:
        """
        Get crossheading texts from all ancestor crossheadings.

        Traverses up the XML tree to find all ancestor crossheading containers
        and extracts their heading text. Used to build hierarchical context.

        Args:
            element: Starting element to traverse from

        Returns:
            List of ancestor crossheading texts, ordered from root to immediate parent
        """
        contexts = []
        current = element.getparent()

        while current is not None:
            # Check if this is a crossheading container
            if current.tag.endswith("hcontainer") and current.get("name") == "crossheading":
                heading_elem = current.find(".//akn:heading", self.xml_handler.namespaces)
                if heading_elem is not None:
                    heading_text = self.xml_handler.get_text_content(heading_elem)
                    contexts.append(heading_text)

            current = current.getparent()

        # Reverse to get root-to-leaf order
        contexts.reverse()
        return contexts

    # ==================== Validation Methods ====================

    def validate_element_has_eid(self, element: etree.Element, context: str = "") -> None:
        """
        Validate that an element has an eId attribute.

        Args:
            element: Element to validate
            context: Additional context for error message

        Raises:
            ValueError: If element lacks eId
        """
        if not element.get("eId"):
            parent = element.getparent()
            parent_info = ""
            if parent is not None:
                parent_eid = parent.get("eId", "NO-EID")
                parent_info = f" (parent: {parent.tag} with eId '{parent_eid}')"

            raise ValueError(
                f"Element {element.tag} missing eId attribute{parent_info}" f"{' - ' + context if context else ''}"
            )

    def validate_amendment_response(self, element: etree.Element, original_eid: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that an element has required structure for an amendment response.

        Args:
            element: The parsed element to validate
            original_eid: The expected eId that should be preserved

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check element has eId
        if not element.get("eId"):
            parent = element.getparent()
            parent_info = ""
            if parent is not None:
                parent_eid = parent.get("eId", "NO-EID")
                parent_info = f" (parent: {parent.tag} with eId '{parent_eid}')"
            error_msg = f"Element {element.tag} missing eId attribute{parent_info}"
            event(logger, EVT.XML_VALIDATION_ERROR, error_msg, element_tag=element.tag, expected_eid=original_eid)
            return False, error_msg

        # Check eId hasn't changed
        if element.get("eId") != original_eid:
            error_msg = f"eId changed from {original_eid} to {element.get('eId')}"
            event(logger, EVT.XML_VALIDATION_ERROR, error_msg, original_eid=original_eid, actual_eid=element.get("eId"))
            return False, error_msg

        return True, None

    # ==================== Editorial Note Methods ====================

    def insert_editorial_notes(self, amended_doc: etree.ElementTree, amendments: List[Amendment]) -> None:
        """
        Insert an editorial note for each amendment into the //meta element of the amended XML document.

        Args:
            amended_doc: The amended XML document.
            amendments: The list of amendments identified.
        """

        try:
            # Find or create the //notes element in the //meta
            notes_element = self._get_or_create_notes_element(amended_doc)
            if notes_element is None:
                raise ValueError("The //notes element could not be found or created in the amended XML document.")

            # Insert a //note element for each amendment into the //notes element
            for amendment in amendments:
                try:
                    self._insert_editorial_note(notes_element, amendment)
                except Exception as e:
                    logger.warning(f"Failed to insert editorial note: amendment_id={amendment.amendment_id}: {e}")

            # Indent the //notes element for readability
            etree.indent(notes_element, level=1)

            self._insert_editorial_note_refs(amended_doc)
        except Exception as e:
            logger.warning(f"Failed to insert editorial notes: {e}")

    def _insert_editorial_note(self, notes_element: etree.Element, amendment: Amendment) -> None:
        """
        Insert a //note element that contains amendment source information into the //notes element

        Args:
            notes_element: The //notes element in the //meta element
            amendment: The Amendment object containing amendment information
        """
        for dnum in amendment.dnum_list:
            note_element = etree.Element(f"{{{XMLHandler.AKN_URI}}}note")
            note_element.set("eId", dnum)

            p_element = etree.SubElement(note_element, f"{{{XMLHandler.AKN_URI}}}p")
            p_element.text = f"{amendment.amendment_type.value.capitalize()} by {amendment.source}."

            notes_element.append(note_element)
            event(
                logger,
                EVT.EDITORIAL_NOTE_INSERTED,
                "Editorial note inserted",
                amendment_id=amendment.amendment_id,
                dnum=dnum,
            )

    def _insert_editorial_note_refs(self, amended_doc: etree.ElementTree) -> None:
        """
        For each change with a Dnum, inserts a //noteRef element into the amended XML document.

        Args:
            amended_doc: The amended XML document.
        """
        elements = amended_doc.xpath("//*[@ukl:changeDnum]", namespaces=self.namespaces)
        for element in elements:
            amendment_id = element.get("amendmentId", "unknown")
            try:
                insertion_point = self._find_suitable_note_ref_location(element)
                dnum = element.get(f"{{{self.UKL_URI}}}changeDnum", "")
                self._insert_editorial_note_ref(insertion_point, dnum)
                event(
                    logger,
                    EVT.EDITORIAL_NOTE_REF_INSERTED,
                    "Editorial note ref inserted",
                    amendment_id=amendment_id,
                    dnum=dnum,
                )
            except Exception as e:
                logger.warning(f"Failed to insert editorial note reference: amendment_id={amendment_id}: {e}")

    def _insert_editorial_note_ref(self, element: etree.Element, dnum: str) -> etree.Element:
        """
        Inserts a //noteRef into the provided element.

        Args:
            element: Element to add note to.
            dnum: The Dnum of the amendment the note refers to.

        Returns:
            The created //noteRef element
        """
        akn_ns = f"{{{self.AKN_URI}}}"
        note_ref = etree.SubElement(
            element, f"{akn_ns}noteRef", {"class": "commentary", "marker": "*", "href": f"#{dnum}"}
        )
        return note_ref

    def _find_suitable_note_ref_location(self, element: etree.Element) -> etree.Element:
        """
        Find the best location to add a //noteRef within an element.

        Args:
            element: Element to search within

        Returns:
            Best element for //noteRef placement (heading, last paragraph, or the element itself)
        """
        # Try heading first
        heading = self.find_heading(element)
        if heading is not None:
            return heading

        # Then try last paragraph
        last_p = self.find_last_paragraph(element)
        if last_p is not None:
            return last_p

        # Default to the element itself
        return element

    def find_heading(self, element: etree.Element) -> Optional[etree.Element]:
        """
        Find heading element within given element.

        Args:
            element: Element to search within

        Returns:
            Heading element or None
        """
        return element.find(".//akn:heading", self.namespaces)

    def find_first_paragraph(self, element: etree.Element) -> Optional[etree.Element]:
        """
        Find first paragraph element within given element.

        Args:
            element: Element to search within

        Returns:
            First paragraph element or None
        """
        return element.find(".//akn:p", self.namespaces)

    def find_last_paragraph(self, element: etree.Element) -> Optional[etree.Element]:
        """
        Find last paragraph element within given element.

        Args:
            element: Element to search within

        Returns:
            Last paragraph element or None
        """
        p_elements = element.xpath(".//akn:p[last()]", namespaces=self.namespaces)
        return p_elements[-1] if p_elements else None

    def _get_or_create_notes_element(self, tree: etree.ElementTree) -> Optional[etree.Element]:
        """
        Get the notes element from the meta section, creating it if it doesn't exist.

        Args:
            tree: The XML element tree to search within

        Returns:
            The notes element from the meta section, or None if meta element not found
        """
        try:
            meta_element = tree.find(".//akn:meta", self.namespaces)
            notes_element = meta_element.find("./akn:notes", self.namespaces)
            if notes_element is None:
                notes_element = etree.SubElement(meta_element, f"{{{XMLHandler.AKN_URI}}}notes")
                notes_element.set("source", "#")
                notes_element.tail = "\n"
            return notes_element
        except Exception as e:
            logger.warning(f"Failed to get or create notes element in meta section: {e}")
            return None

    # ==================== Transformation Methods ====================

    def transform_eids(self, element: etree.Element, old_prefix: str, new_prefix: str = "") -> None:
        """
        Transform eIds by replacing a prefix pattern.

        Recursively updates eId attributes in an element tree by replacing
        an old prefix with a new prefix (or removing it if new_prefix is empty).

        Args:
            element: Root element to transform
            old_prefix: The prefix to replace (e.g., "sec_21__subsec_2__qstr")
            new_prefix: The new prefix to use (empty string to remove prefix)

        Example:
            Transform "sec_21__subsec_2__qstr__sec_59b" to "sec_59b"
            by calling transform_eids(elem, "sec_21__subsec_2__qstr__", "")
        """
        # Update the element's own eId
        current_eid = element.get("eId")
        if current_eid and current_eid.startswith(old_prefix):
            if new_prefix:
                new_eid = current_eid.replace(old_prefix, new_prefix, 1)
            else:
                # Remove the prefix
                new_eid = current_eid[len(old_prefix) :]
                # Remove leading underscores if any
                new_eid = new_eid.lstrip("_")
            element.set("eId", new_eid)

        # Update all descendant elements with eIds
        for descendant in element.xpath(".//*[@eId]"):
            desc_eid = descendant.get("eId")
            if desc_eid and desc_eid.startswith(old_prefix):
                if new_prefix:
                    new_eid = desc_eid.replace(old_prefix, new_prefix, 1)
                else:
                    # Remove the prefix
                    new_eid = desc_eid[len(old_prefix) :]
                    # Remove leading underscores if any
                    new_eid = new_eid.lstrip("_")
                descendant.set("eId", new_eid)

    # ==================== Tree Manipulation Methods ====================

    def create_subtree_copy(self, element: etree.Element) -> etree.ElementTree:
        """
        Create a deep copy of an element subtree wrapped in a dummy root element.

        This is used to ensure XPath queries like `.//*[@eId='...']` still work
        even when operating on subtrees that are not rooted at the act-level root.

        Args:
            element: Element to copy

        Returns:
            ElementTree rooted at a dummy element with the copied subtree inside
        """
        dummy_root = etree.Element("root")
        dummy_root.append(copy.deepcopy(element))
        return etree.ElementTree(dummy_root)

    def replace_element_in_tree(self, tree: etree.ElementTree, old_eid: str, new_element: etree.Element) -> bool:
        """
        Replace an element in the tree with a new element.

        Args:
            tree: Target tree
            old_eid: eId of element to replace
            new_element: New element to insert

        Returns:
            True if replacement successful, False otherwise
        """
        old_element = self.find_element_by_eid(tree, old_eid)
        if old_element is None:
            return False

        parent = old_element.getparent()
        if parent is None:
            # Root element replacement
            tree._setroot(new_element)
        else:
            parent.replace(old_element, new_element)

        return True

    def inject_xml_comment(self, element: etree.Element, comment_text: str, position: int = 0) -> None:
        """
        Inject an XML comment into an element at specified position.

        Args:
            element: Element to modify
            comment_text: Text for the comment
            position: Where to insert (0 = beginning)
        """
        comment = etree.Comment(comment_text)
        element.insert(position, comment)

    def set_namespaced_attribute(self, element: etree.Element, namespace_uri: str, local_name: str, value: str) -> None:
        """
        Set a namespaced attribute on an element.

        Args:
            element: Element to modify
            namespace_uri: Namespace URI
            local_name: Local name of attribute
            value: Attribute value
        """
        ns_attr = f"{{{namespace_uri}}}{local_name}"
        element.set(ns_attr, value)

    # ==================== Internal Helper Methods ====================

    def _extract_lowest_grouping_provisions(self, element: etree.Element, provision_list: List[etree.Element]) -> None:
        """
        Extract the lowest-level grouping provisions from a schedule.

        Args:
            element: Schedule element to process
            provision_list: List to append found provisions to
        """
        class_name = element.get("class", "")

        if "Group" not in class_name and class_name != "sch":
            return

        # Check if this element contains further grouping
        has_child_groups = any("Group" in child.get("class", "") for child in element)

        if has_child_groups:
            # Recurse into children
            for child in element:
                self._extract_lowest_grouping_provisions(child, provision_list)
        else:
            # This is a lowest-level grouping
            provision_list.append(element)

    def _extract_provision_type(self, eid: str) -> str:
        """
        Extract the provision type from an eId.

        Args:
            eid: Element ID like "sec_5__subsec_3" or "sched_2__para_4"

        Returns:
            Provision type (e.g., "sec", "subsec", "para", "sched")
        """
        # Get the last component of the eId
        parts = eid.split("__")
        last_part = parts[-1]

        # Extract the type prefix (everything before the first underscore in the last part)
        match = re.match(r"^([a-zA-Z]+)_", last_part)
        if match:
            return match.group(1)

        # If no match, try the first part
        first_part = parts[0]
        match = re.match(r"^([a-zA-Z]+)_", first_part)
        if match:
            return match.group(1)

        return ""

    def _build_xpath_for_provision_number(self, prov_type: str, number: str) -> str:
        """
        Build XPath to find a provision by type and number.

        Args:
            prov_type: Normalized provision type
            number: Provision number to search for

        Returns:
            XPath string or empty string if type unknown
        """
        # Map provision types to XPath patterns
        xpath_templates = {
            "section": f".//akn:section[akn:num[contains(.,'{number}')]]",
            "regulation": f".//akn:hcontainer[@name='regulation'][akn:num[contains(.,'{number}')]]",
            "article": f".//akn:article[akn:num[contains(.,'{number}')]]",
            "rule": f".//akn:rule[akn:num[contains(.,'{number}')]]",
            "schedule": f".//akn:hcontainer[@name='schedule'][akn:num[contains(.,'{number}')]]",
            "paragraph": f".//akn:paragraph[akn:num[contains(.,'{number}')]]",
            "para": f".//akn:para[akn:num[contains(.,'{number}')]]",
        }

        # Handle unnumbered schedule
        if prov_type == "schedule" and (not number or number == "0"):
            return ".//akn:hcontainer[@name='schedule'][not(akn:num) or akn:num='']"

        return xpath_templates.get(prov_type, "")

    def _get_base_xpath_for_provision_type(self, prov_type: str) -> str:
        """
        Get base XPath for finding all provisions of a type.

        Args:
            prov_type: Normalized provision type

        Returns:
            XPath string or empty string if type unknown
        """
        xpath_map = {
            "section": ".//akn:section",
            "regulation": ".//akn:hcontainer[@name='regulation']",
            "article": ".//akn:article",
            "rule": ".//akn:rule",
            "schedule": ".//akn:hcontainer[@name='schedule']",
            "paragraph": ".//akn:paragraph",
            "para": ".//akn:para",
        }

        return xpath_map.get(prov_type, "")

    def _extract_provision_number(self, element: etree.Element) -> Optional[str]:
        """
        Extract the number from a provision's num element.

        Args:
            element: Element to extract number from

        Returns:
            The provision number or None if not found
        """
        # Convert to LegiElement to use its methods
        legi_elem = LegiElement(element)
        num_elem = legi_elem.get_descendant(".//akn:num")

        if num_elem is not None and num_elem.element.text:
            # Extract just the number part (handles '3.', '(3)', '3', etc.)
            match = re.search(r"(\d+[A-Za-z]*)", num_elem.element.text)
            if match:
                return match.group(1)
        return None
