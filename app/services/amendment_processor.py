# app/services/amendment_processor.py
"""
Handles the application of amendments to XML documents.
Processes insertions, deletions, and substitutions while maintaining
proper change tracking markup.

This processor is responsible for:
1. Applying amendments to a working copy of an XML subtree
2. Coordinating with LLM for complex text-based amendments
3. Handling whole-provision structural amendments directly
4. Ensuring proper change tracking markup is applied
5. Inserting error comments for failed amendments
"""

import copy
import re

import tiktoken
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from lxml import etree

from ..models.amendments import Amendment, AmendmentType, AmendmentLocation
from .xml_handler import XMLHandler
from ..logging.debug_logger import get_logger, event, EventType as EVT
from ..models.legi_element import LegiElement

logger = get_logger(__name__)


class AmendmentProcessor:
    """
    Processes individual amendments on XML subtrees.

    This class is stateless and designed to work on copies of XML subtrees,
    making it safe for concurrent use across multiple threads.
    """

    def __init__(self, xml_handler: XMLHandler, llm_kernel=None):
        """
        Initialise the processor.

        Args:
            xml_handler: XMLHandler instance for XML operations
            llm_kernel: Optional LLM kernel for complex amendments
        """
        self.xml_handler = xml_handler
        self.llm_kernel = llm_kernel
        self.namespaces = xml_handler.namespaces

        # Initialise tokenizer based on the active LLM service
        self.tokenizer = None
        if llm_kernel:
            if llm_kernel.llm_config.enable_azure_openai:
                # Use OpenAI tokenizer for GPT models
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.debug("Initialised OpenAI tokenizer (cl100k_base)")
                except Exception as e:
                    logger.warning(f"Failed to initialise OpenAI tokenizer: {e}. Using character-based estimation.")
            elif llm_kernel.llm_config.enable_aws_bedrock:
                # Claude models don't have a public tokenizer
                # Use character-based estimation instead
                logger.debug("Using character-based token estimation for Claude models")

    # ==================== Public Interface Methods ====================

    def apply_amendment(
        self,
        amendment: Amendment,
        working_tree: etree.ElementTree,
        amending_bill: etree.ElementTree,
        schedule_id: str = None,
        llm_response: str = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply a single amendment to a working copy of the XML tree.

        This method modifies the working_tree in place. It should only
        be called on a copy of the original tree, not the original itself.

        For partial amendments, a pre-fetched LLM response must be provided.
        Whole provision amendments are applied directly without LLM.

        Args:
            amendment: Amendment to apply
            working_tree: Working copy of the target XML tree (will be modified)
            amending_bill: Source bill containing amendment text
            schedule_id: Schedule ID for logging (optional, used for tracking)
            llm_response: Pre-fetched LLM response (required for partial amendments)

        Returns:
            Tuple of (success, error_message)
        """
        try:

            # Log amendment for debugging
            logger.debug(
                f"Applying {amendment.amendment_type.value} amendment "
                f"{getattr(amendment, 'amendment_id', 'unknown')} "
                f"to {amendment.affected_provision} "
                f"(whole_provision={amendment.whole_provision})"
            )

            # Find target element in working tree
            target = self.xml_handler.find_element_by_eid_components(working_tree, amendment.affected_provision)

            if target is None:
                return False, f"Target element {amendment.affected_provision} not found"

            # Validate target has eId
            try:
                self.xml_handler.validate_element_has_eid(
                    target, f"for amendment {getattr(amendment, 'amendment_id', 'unknown')}"
                )
            except ValueError as e:
                return False, str(e)

            # Get source element from amending bill
            source_element = self.xml_handler.find_element_by_eid(amending_bill, amendment.source_eid)

            if source_element is None:
                return False, f"Source element {amendment.source_eid} not found"

            # Handle based on whether it's whole provision or partial
            if amendment.whole_provision:
                # Apply structural amendment directly (no LLM needed)
                if amendment.amendment_type == AmendmentType.INSERTION:
                    success = self._apply_whole_provision_insertion(amendment, target, source_element)
                elif amendment.amendment_type == AmendmentType.DELETION:
                    success = self._apply_whole_provision_deletion(amendment, target, source_element)
                elif amendment.amendment_type == AmendmentType.SUBSTITUTION:
                    success = self._apply_whole_provision_substitution(amendment, target, source_element)
                else:
                    return False, f"Unknown amendment type: {amendment.amendment_type}"

                if success:
                    self.xml_handler.inject_amendment_id(target.getparent(), amendment)
                    return True, None
                else:
                    return False, "Whole provision amendment application failed"

            else:
                # Partial amendment - requires pre-fetched LLM response
                if not llm_response:
                    return False, (
                        f"Pre-fetched LLM response required for partial {amendment.amendment_type.value} amendment. "
                        "Ensure LLM responses are fetched before calling apply_amendment."
                    )

                # Apply using the pre-fetched response
                success = self._apply_llm_response(llm_response, target, amendment, working_tree)

                if success:
                    return True, None
                else:
                    return False, "Failed to apply partial amendment using LLM response"

        except Exception as e:
            logger.error(f"Failed to apply amendment {getattr(amendment, 'amendment_id', 'unknown')}: {str(e)}")
            logger.exception("Full traceback:")
            return False, f"Unexpected error: {str(e)}"

    def apply_each_place_amendment(
        self,
        amendment: Amendment,
        working_tree: etree.ElementTree,
        amending_bill: etree.ElementTree,
        pattern_data: Dict[str, str],
        schedule_id: str = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply an "each place" amendment algorithmically using extracted patterns.

        Args:
            amendment: Amendment to apply
            working_tree: Working copy of the target XML tree
            amending_bill: Source bill containing amendment text
            pattern_data: Dictionary with 'find_text' and 'replace_text'
            schedule_id: Schedule ID for logging

        Returns:
            Tuple of (success, error_message)
        """
        try:
            find_text = pattern_data.get("find_text", "")
            replace_text = pattern_data.get("replace_text", "")

            if not find_text:
                return False, "No find_text in pattern data"

            # Log the application attempt
            event(
                logger,
                EVT.EACH_PLACE_APPLICATION,
                f"Applying each place amendment {getattr(amendment, 'amendment_id', 'unknown')}",
                find_text=find_text,
                replace_text=replace_text,
                amendment_type=amendment.amendment_type.value,
            )

            # Find all text nodes within the affected provision
            affected_element = self.xml_handler.find_element_by_eid(working_tree, amendment.affected_provision)

            if affected_element is None:
                return False, f"Affected provision {amendment.affected_provision} not found"

            # Track changes for proper markup
            changes_made = []

            # Process all text nodes in the affected element using iterative approach
            self._replace_text_occurrences_iteratively(
                affected_element, find_text, replace_text, amendment.amendment_type, changes_made
            )

            if not changes_made:
                return False, f"Pattern '{find_text}' not found in {amendment.affected_provision}"

            # Inject amendment ID into the affected element
            self.xml_handler.inject_amendment_id(affected_element, amendment)

            # Count total occurrences for logging
            total_occurrences = sum(change.get("occurrences", 0) for change in changes_made)

            # Verify tracking was applied by counting elements with changeGenerated
            tracked_elements = affected_element.xpath(
                ".//*[@ukl:changeGenerated='true']", namespaces=self.xml_handler.namespaces
            )

            logger.info(
                f"Each place amendment {amendment.amendment_id}: "
                f"processed {total_occurrences} occurrences, "
                f"created {len(tracked_elements)} tracked changes"
            )

            return True, None

        except Exception as e:
            logger.error(f"Failed to apply each place amendment: {str(e)}")
            return False, f"Unexpected error: {str(e)}"

    def prepare_llm_amendment(
        self,
        amendment: Amendment,
        target: etree.Element,
        source_element: etree.Element,
        schedule_id: str,
    ) -> Dict[str, Any]:
        """
        Prepare data for LLM call without applying the amendment.

        Used when pre-fetching LLM responses in parallel before sequential application.

        Args:
            amendment: Amendment to prepare
            target: Target element
            source_element: Source element with amendment instructions
            schedule_id: Schedule ID for logging

        Returns:
            Dictionary with prompt data and metadata
        """
        # Determine prompt name based on amendment type
        prompt_map = {
            AmendmentType.INSERTION: "ApplyInsertionAmendment",
            AmendmentType.DELETION: "ApplyDeletionAmendment",
            AmendmentType.SUBSTITUTION: "ApplySubstitutionAmendment",
        }

        prompt_name = prompt_map.get(amendment.amendment_type)
        if not prompt_name:
            raise ValueError(f"Unknown amendment type: {amendment.amendment_type}")

        # Prepare XML strings for LLM
        target_xml = etree.tostring(target, encoding="unicode", pretty_print=True)
        source_xml = etree.tostring(source_element, encoding="unicode", pretty_print=True)

        return {
            "prompt_name": prompt_name,
            "schedule_id": schedule_id,
            "amendment_id": getattr(amendment, "amendment_id", None),
            "kwargs": {
                "source": amendment.source,
                "amendment_xml": source_xml,
                "original_xml": target_xml,
            },
        }

    def insert_all_error_comments(self, output_act: etree.ElementTree, tracker) -> None:
        """
        Insert error comments for all failed amendments.

        This method orchestrates the insertion of error comments for all amendments
        that couldn't be applied successfully. It coordinates with the tracker to
        identify which amendments need comments and marks them as processed.

        Args:
            output_act: The output XML tree to add comments to
            tracker: AmendmentTracker instance for state management
        """
        # Import here to avoid circular dependency
        from ..models.amendments import Amendment, AmendmentType, AmendmentLocation

        for record in tracker.get_all_requiring_comments():
            # Create Amendment object from record
            amendment = Amendment(
                source_eid=record.source_eid,
                source=record.source,
                amendment_type=AmendmentType[record.amendment_type.upper()],
                whole_provision=record.whole_provision,
                location=AmendmentLocation[record.location.upper()],
                affected_document=record.affected_document,
                affected_provision=record.affected_provision,
            )

            # Find target element or ancestor
            target = self.xml_handler.find_element_by_eid_with_fallback(output_act, record.affected_provision)

            if target is not None:
                self.insert_error_comment(target, amendment, record.error_message or "Amendment could not be applied")
                tracker.mark_error_commented(record.amendment_id)
            else:
                logger.error(f"Could not find target for error comment: {record.affected_provision}")

    def insert_error_comment(self, element: etree.Element, amendment: Amendment, error_message: str) -> None:
        """
        Insert an oxygen comment for a failed amendment.

        This modifies the element in place and is used to document amendments
        that could not be applied successfully.

        Args:
            element: Element to add comment to (or its ancestor)
            amendment: Failed amendment
            error_message: Error description
        """
        author = "Keeling AI - Applied Amendment Error"
        comment = (
            f"Unable to apply the {amendment.amendment_type.value} amendment from "
            f"{amendment.source_eid} of the source document to "
            f"{amendment.affected_provision} of the target document. "
            f"\nError: {error_message}"
        )

        self._insert_oxy_comment(element, author, comment)

    def check_token_limits(
        self, amendment: Amendment, target: etree.Element, source_element: etree.Element
    ) -> Tuple[bool, int]:
        """
        Check if an amendment would exceed token limits before calling LLM.

        Args:
            amendment: Amendment to check
            target: Target element (the affected provision)
            source_element: Source element with amendment instructions

        Returns:
            Tuple of (is_within_limits, estimated_output_tokens)
        """
        try:
            # Get the maximum completion tokens for the active service
            max_completion_tokens = self.llm_kernel.llm_config.get_max_completion_tokens()

            # Get the XML string of the target element
            target_xml = etree.tostring(target, encoding="unicode", pretty_print=True)

            # Estimate tokens for the target element only
            # Assume that the LLM will return roughly the same amount of XML
            estimated_output_tokens = self._estimate_tokens(target_xml)

            # Log the estimation for debugging
            logger.debug(
                f"Token estimation for amendment {getattr(amendment, 'amendment_id', 'unknown')}: "
                f"target provision tokens={estimated_output_tokens}, "
                f"max_completion={max_completion_tokens}"
            )

            # Check if the target provision itself would exceed limits
            is_within_limits = estimated_output_tokens <= max_completion_tokens

            return is_within_limits, estimated_output_tokens

        except Exception as e:
            logger.warning(f"Error estimating tokens: {e}. Allowing amendment to proceed.")
            # On error, allow the amendment to proceed (fail open)
            return True, 0

    def correct_amendments(
        self, amendments: List[Amendment], target_act: etree.ElementTree, amending_bill: etree.ElementTree
    ) -> None:
        """
        Correct the AI-generated fields of each amendment algorithmically if necessary.

        Args:
            amendments: List of amendments to correct.
            target_act: The act in which the amendments will be applied.
            amending_bill: The bill from which the amendments were extracted.
        """
        for amendment in amendments:
            # Correct the 'whole_provision' field of the amendment
            self.correct_whole_provision_field(amendment, amending_bill)

            # Correct the 'affected_provision' field of the amendment (Do not correct amendments that will be manually
            # inserted, because there may be dependency on another amendment being applied)
            self.correct_affected_provision_field(amendment, target_act)

    # ==================== Amendment Correction Methods ====================

    def correct_affected_provision_field(self, amendment: Amendment, target_act: etree.ElementTree) -> None:
        """
        Correct the affected_provision field of the amendment by finding the closest matching element in the target_act
        to the affected_provision.

        Args:
            amendment: The amendment to correct.
            target_act: The act in which the amendment will be applied.
        """
        try:
            # Only correct the affected_provision field if the amendment will be applied by the LLM
            if amendment.whole_provision:
                return

            # Find the eid of the affected_provision in the target_act or the nearest ancestor that can be identified
            target_element = self.xml_handler.find_closest_match_by_eid(target_act, amendment.affected_provision)
            target_eid = target_element.get("eId")

            # Make the correction to the affected_provision field
            if target_eid is not None and target_eid != amendment.affected_provision:
                logger.info(
                    f"Corrected affected_provision from {amendment.affected_provision} to {target_eid}"
                    f" amendment_id={amendment.amendment_id}"
                )
                amendment.affected_provision = target_eid
        except Exception as e:
            logger.error(f"Failed to correct affected_provision: {e} amendment_id={amendment.amendment_id}")

    def correct_whole_provision_field(self, amendment: Amendment, amending_bill: etree.ElementTree) -> None:
        """
        Correct the affected_provision field of the amendment.

        Args:
            amendment: The amendment to correct.
            amending_bill: The bill from which the amendment was extracted.
        """
        should_manually_apply = self.should_manually_apply(amendment, amending_bill)
        if should_manually_apply is None:
            return
        elif should_manually_apply != amendment.whole_provision:
            logger.info(
                f"Corrected whole_provision from {amendment.whole_provision} to {should_manually_apply}"
                f" amendment_id={amendment.amendment_id}"
            )
            amendment.whole_provision = should_manually_apply

    def should_manually_apply(self, amendment: Amendment, amending_bill: etree.ElementTree) -> bool | None:
        """
        Determine if an amendment should be applied manually (without LLM) or using LLM.
        Whole provision amendments are applied manually, while partial amendments require LLM.

        Args:
            amendment: Amendment to check
            amending_bill: The document from which the amendment was derived

        Returns:
            True if the amendment should be applied manually, False if it requires LLM, None if inconclusive.
        """
        try:
            # If the amendment is already set to use the LLM, do not override that decision
            if not amendment.whole_provision:
                return None

            # For manually applied insertions and substitutions, the amendment quoted structure must contain top-level
            # or grouping provisions
            if not amendment.is_deletion():
                source_element = LegiElement(self.xml_handler.find_element_by_eid(amending_bill, amendment.source_eid))
                quoted_structure = source_element.get_descendant(".//akn:quotedStructure")
                if quoted_structure is None:
                    return False
                qs_provision = quoted_structure.get_children()[0]
                if re.match("^(prov|sch|group|para)", qs_provision.get_attribute("class")) is None:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to correct whole_provision: {e} amendment_id={amendment.amendment_id}")
            return None

    # ==================== Whole Provision Amendment Methods ====================

    def _apply_whole_provision_insertion(
        self, amendment: Amendment, target: etree.Element, source_element: etree.Element, is_substitution: bool = False
    ) -> bool:
        """
        Apply insertion of complete structural provisions.

        Args:
            amendment: Amendment details
            target: Target element for positioning
            source_element: Source element containing content
            is_substitution: Whether this is part of a substitution

        Returns:
            True if successful, False otherwise
        """
        # Find quoted structure
        quoted_structure = source_element.find(".//akn:quotedStructure", self.namespaces)
        if quoted_structure is None:
            # Check if parent is quotedStructure
            parent = source_element.getparent()
            if parent is not None and parent.tag.endswith("quotedStructure"):
                quoted_structure = parent
            else:
                logger.error("No quoted structure found for whole provision insertion")
                return False

        # Extract elements to insert
        elements_to_insert = list(quoted_structure)
        if not elements_to_insert:
            logger.error("No elements in quoted structure")
            return False

        # Prepare elements for insertion (fix eIds)
        self._prepare_quoted_elements_for_insertion(elements_to_insert)

        # Get parent and position
        parent = target.getparent()
        if parent is None:
            logger.error("Target element has no parent for insertion")
            return False

        try:
            index = list(parent).index(target)
        except ValueError:
            logger.error("Target element not found in parent's children")
            return False

        # Apply insertion based on location
        if amendment.location in {AmendmentLocation.AFTER, AmendmentLocation.BEFORE}:
            self._insert_elements(parent, index, elements_to_insert, amendment, is_substitution, amendment.location)
            return True
        else:
            logger.error(f"Invalid location for insertion: {amendment.location}")
            return False

    def _apply_whole_provision_deletion(
        self, amendment: Amendment, target: etree.Element, source_element: etree.Element
    ) -> bool:
        """
        Apply a whole provision deletion.

        Args:
            amendment: Deletion amendment
            target: Target element to delete
            source_element: Source element (unused for deletions)

        Returns:
            True (deletions always succeed for whole provisions)
        """
        # Mark the element for deletion
        self.xml_handler.add_change_markup(target, "del", is_start=True, is_end=True, add_dnum=True)
        return True

    def _apply_whole_provision_substitution(
        self, amendment: Amendment, target: etree.Element, source_element: etree.Element
    ) -> bool:
        """
        Apply a whole provision substitution (deletion + insertion).

        Args:
            amendment: Substitution amendment
            target: Target element to replace
            source_element: Source element containing replacement

        Returns:
            True if successful, False otherwise
        """
        # Mark original for deletion (without end marker)
        self.xml_handler.add_change_markup(target, "delReplace", is_start=True, is_end=False, add_dnum=False)

        # Apply insertion after the deleted element
        # Temporarily change location for insertion logic
        original_location = amendment.location
        amendment.location = AmendmentLocation.AFTER

        success = self._apply_whole_provision_insertion(amendment, target, source_element, is_substitution=True)

        # Restore original location
        amendment.location = original_location

        return success

    # ==================== Partial Amendment Methods (LLM-based) ====================

    def _apply_llm_response(
        self, response: str, target: etree.Element, amendment: Amendment, working_tree: etree.ElementTree
    ) -> bool:
        """
        Process and apply the LLM's XML response.

        Args:
            response: Raw XML string from LLM
            target: Original target element
            amendment: Amendment being applied
            working_tree: The working tree being modified

        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse the response using xml_handler. The response will be wrapped in a //wrapper element incase the
            # response contains more than one root element.
            amended = self.xml_handler.parse_xml_string(f"<wrapper>{response}</wrapper>", ensure_namespaces=True)

            # Validate the response
            is_valid, error_msg = self._validate_llm_amendment_response(
                amended, target.get("eId"), amendment.amendment_type.value
            )

            if not is_valid:
                logger.error(
                    f"Invalid LLM response for amendment "
                    f"{getattr(amendment, 'amendment_id', 'unknown')}: {error_msg}"
                )
                return False

            # Replace in tree
            parent = target.getparent()
            if parent is not None:
                parent.replace(target, amended)
                # unwrap the wrapper
                wrapper = LegiElement(parent.find(".//wrapper"))
                wrapper.unwrap_element()
                return True
            else:
                logger.error("Target has no parent for replacement")
                return False

        except etree.XMLSyntaxError as e:
            logger.error(f"Invalid XML from LLM: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            return False

    # ==================== Helper Methods ====================

    def _insert_elements(
        self,
        parent: etree.Element,
        index: int,
        elements: List[etree.Element],
        amendment: Amendment,
        is_substitution: bool,
        location: AmendmentLocation,
    ) -> None:
        """
        Insert elements at the specified location relative to target position.

        Args:
            parent: Parent element to insert into
            index: Index of target element
            elements: Elements to insert
            amendment: Amendment being applied
            is_substitution: Whether this is part of a substitution
            location: Where to insert (BEFORE or AFTER)
        """
        if location == AmendmentLocation.AFTER:
            # Insert elements after the target in normal order
            for pos, elem in enumerate(elements):
                new_elem = copy.deepcopy(elem)

                # Determine change markup
                change_type = "insReplace" if is_substitution else "ins"
                is_first_elem = pos == 0
                is_last_elem = pos == len(elements) - 1
                is_start = is_first_elem and not is_substitution
                is_end = is_last_elem

                self.xml_handler.add_change_markup(
                    new_elem, change_type, is_start=is_start, is_end=is_end, add_dnum=is_end
                )

                # Insert after target at incrementing positions
                parent.insert(index + pos + 1, new_elem)

        else:  # BEFORE
            # Insert elements before the target in reverse order
            for pos, elem in enumerate(reversed(elements)):
                new_elem = copy.deepcopy(elem)

                # Determine change markup
                change_type = "insReplace" if is_substitution else "ins"
                is_first_elem = pos == len(elements) - 1  # Last in reversed is first
                is_last_elem = pos == 0  # First in reversed is last
                is_start = is_first_elem and not is_substitution
                is_end = is_last_elem

                self.xml_handler.add_change_markup(
                    new_elem, change_type, is_start=is_start, is_end=is_end, add_dnum=is_end
                )

                # Always insert at the same position when going before
                parent.insert(index, new_elem)

    def _prepare_quoted_elements_for_insertion(self, elements: List[etree.Element]) -> None:
        """
        Prepare quoted structure elements for insertion by fixing their eIds.

        Quoted structures from amending bills often have eIds like:
        "sec_21__subsec_2__qstr__sec_59b" which need to become "sec_59b"
        in the target document.

        Args:
            elements: List of elements from quoted structure to prepare
        """
        for elem in elements:
            current_eid = elem.get("eId", "")
            if "__qstr__" in current_eid:
                # Find the prefix to remove (everything up to and including __qstr__)
                parts = current_eid.split("__qstr__")
                if len(parts) == 2:
                    prefix_to_remove = parts[0] + "__qstr__"
                    logger.debug(f"Transforming eIds: removing prefix '{prefix_to_remove}' from element {current_eid}")
                    # Transform all eIds in this element tree
                    self.xml_handler.transform_eids(elem, prefix_to_remove, "")

    def _validate_llm_amendment_response(
        self, element: etree.Element, original_eid: str, amendment_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an LLM-generated amendment response.

        Args:
            element: The parsed element from LLM
            original_eid: The expected eId that should be preserved
            amendment_type: Type of amendment for context

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic structural validation
        is_valid, error_msg = self.xml_handler.validate_amendment_response(element[0], original_eid)
        if not is_valid:
            return False, f"in LLM response for {amendment_type} amendment: {error_msg}"

        # Check for change markup
        if not self.xml_handler.element_has_change_markup(element):
            error_msg = "No change markup found (no ins/del elements or ukl:change attributes)"
            event(logger, EVT.XML_VALIDATION_ERROR, error_msg, original_eid=original_eid, amendment_type=amendment_type)
            return False, error_msg

        # Check for required tracking attributes
        tracking_attrs = self.xml_handler.get_change_tracking_attributes(element)

        missing = []
        if not tracking_attrs["changeStart"]:
            missing.append("ukl:changeStart")
        if not tracking_attrs["changeEnd"]:
            missing.append("ukl:changeEnd")
        if not tracking_attrs["changeGenerated"]:
            missing.append("ukl:changeGenerated")

        if missing:
            error_msg = f"Missing required attributes: {', '.join(missing)}"
            event(
                logger,
                EVT.XML_VALIDATION_ERROR,
                error_msg,
                original_eid=original_eid,
                amendment_type=amendment_type,
                missing_attributes=missing,
            )
            return False, error_msg

        return True, None

    def _insert_oxy_comment(self, element: etree.Element, author: str, comment: str) -> None:
        """
        Insert oxygen comment at beginning of element, after any existing comments.

        Args:
            element: Element to add comment to
            author: Comment author
            comment: Comment text
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Escape quotes in the comment
        comment = comment.replace('"', "&quot;")

        data = f'author="{author}" timestamp="{current_time}" comment="{comment}"'

        start_pi = etree.ProcessingInstruction("oxy_comment_start", data)
        end_pi = etree.ProcessingInstruction("oxy_comment_end", "")

        # Find position after any existing PIs
        insert_pos = 0
        for i, child in enumerate(element):
            if isinstance(child, etree._ProcessingInstruction):
                # Skip past existing processing instructions
                insert_pos = i + 1
            else:
                # Found first non-PI child, insert here
                break
        else:
            # All children are PIs or no children
            insert_pos = len(element)

        element.insert(insert_pos, start_pi)
        element.insert(insert_pos + 1, end_pi)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Uses tiktoken for accurate counting when available, otherwise
        falls back to empirically-measured character ratio.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        if self.tokenizer:
            try:
                # Use tiktoken for accurate token counting (OpenAI models)
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.debug(f"Tokenizer error: {e}. Using fallback estimation.")

        # Fallback: character-based estimation
        # Empirical testing with legislative XML shows ~2.92 characters per token
        # (measured using OpenAI's tokenizer on 100,000+ characters of UK legislative XML)
        return int(len(text) / 2.92)

    def _replace_text_occurrences_iteratively(
        self,
        element: etree.Element,
        find_text: str,
        replace_text: str,
        amendment_type: AmendmentType,
        changes_made: List[Dict[str, Any]],
    ) -> bool:
        """
        Iteratively replace all text occurrences in an element with change tracking elements.
        Creates proper XML elements with change tracking for each occurrence found.
        This ensures each occurrence gets its own DNUM when processing "each place" amendments.
        Args:
            element: Element to process
            find_text: Text to find
            replace_text: Replacement text
            amendment_type: Type of amendment (INSERTION, DELETION, SUBSTITUTION)
            changes_made: List to track changes for logging/verification
        Returns:
            True if any replacements were made
        """
        made_changes = False

        # First collect all text nodes to process to avoid modifying while iterating
        text_nodes_to_process = []

        # Use a stack to collect all elements
        stack = [element]

        while stack:
            current_element = stack.pop()

            # Skip quoted structures
            tag = getattr(current_element, "tag", None)
            if tag and "quotedStructure" in str(tag):
                continue

            # Collect text node if it contains find_text
            if current_element.text and find_text in current_element.text:
                text_nodes_to_process.append({"element": current_element, "attr": "text", "text": current_element.text})

            # Process children, skipping nodes with change markup
            for child in current_element:
                # Check if child is ins/del element by examining the tag directly
                child_tag = getattr(child, "tag", None)
                is_ins_or_del = False
                if child_tag and isinstance(child_tag, str):
                    # Check if tag ends with }ins or }del (namespace-aware)
                    is_ins_or_del = child_tag.endswith("}ins") or child_tag.endswith("}del")

                if not is_ins_or_del:
                    # Check for wholly changed structural elements
                    child_ukl_change = child.get(f"{{{self.namespaces['ukl']}}}change")
                    skip_child = False

                    if child_ukl_change in ["ins", "del"]:
                        # For ins/del, require both start and end
                        child_has_start = child.get(f"{{{self.namespaces['ukl']}}}changeStart") == "true"
                        child_has_end = child.get(f"{{{self.namespaces['ukl']}}}changeEnd") == "true"
                        if child_has_start and child_has_end:
                            skip_child = True

                    elif child_ukl_change == "delReplace":
                        # For delReplace, only requires changeStart
                        if child.get(f"{{{self.namespaces['ukl']}}}changeStart") == "true":
                            skip_child = True

                    elif child_ukl_change == "insReplace":
                        # For insReplace, only requires changeEnd
                        if child.get(f"{{{self.namespaces['ukl']}}}changeEnd") == "true":
                            skip_child = True

                    if not skip_child:
                        stack.append(child)

                # Always collect tail text, regardless of element type
                if child.tail and find_text in child.tail:
                    text_nodes_to_process.append({"element": child, "attr": "tail", "text": child.tail})

        # Now process all collected text nodes and replace text
        for node_info in text_nodes_to_process:
            occurrences = self._replace_occurrences_in_text_node(
                node_info["element"], node_info["attr"], node_info["text"], find_text, replace_text, amendment_type
            )

            if occurrences > 0:
                changes_made.append(
                    {
                        "element": node_info["element"],
                        "attr": node_info["attr"],
                        "occurrences": occurrences,
                    }
                )
                made_changes = True

        return made_changes

    def _replace_occurrences_in_text_node(
        self,
        element: etree.Element,
        attr: str,
        text: str,
        find_text: str,
        replace_text: str,
        amendment_type: AmendmentType,
    ) -> int:
        """
        Replace all occurrences of find_text in a single text node with change tracking elements.

        Creates proper XML elements with change tracking for each individual occurrence.
        This is the key method that ensures each occurrence gets its own DNUM.

        Args:
            element: Element containing the text
            attr: Either "text" or "tail" indicating which text attribute
            text: The text content to process
            find_text: Text to find
            replace_text: Replacement text
            amendment_type: Type of amendment

        Returns:
            Number of occurrences replaced
        """
        # Use word boundaries to find only whole word matches
        escaped_find = re.escape(find_text)
        pattern = rf"(?<!\w){escaped_find}(?!\w)"

        # Check if pattern exists
        if not re.search(pattern, text):
            return 0

        # Split text by pattern occurrences, returning text segments between matches
        text_parts = re.split(pattern, text)

        if attr == "tail":
            return self._replace_occurrences_in_tail(element, text_parts, find_text, replace_text, amendment_type)
        else:  # attr == "text"
            return self._replace_occurrences_in_element_text(
                element, text_parts, find_text, replace_text, amendment_type
            )

    def _replace_occurrences_in_tail(
        self, element: etree.Element, parts: List[str], find_text: str, replace_text: str, amendment_type: AmendmentType
    ) -> int:
        """
        Replace occurrences in an element's tail text with change tracking elements.

        Args:
            element: Element whose tail is being processed
            parts: Text parts split by find_text
            find_text: Text being replaced/deleted
            replace_text: New text for substitutions/insertions
            amendment_type: Type of amendment

        Returns:
            Number of occurrences replaced
        """
        parent = element.getparent()
        if parent is None:
            logger.warning("Cannot process tail text without parent")
            return 0

        element_index = list(parent).index(element)
        element.tail = None
        insert_position = element_index + 1

        for i, part in enumerate(parts[:-1]):
            # Add any text before this occurrence
            if part:
                # Create span
                text_span = self.xml_handler.create_akn_element("span", part)
                parent.insert(insert_position, text_span)
                insert_position += 1

            # Create the change elements for this occurrence
            elements_created = self._create_tracked_change_elements(
                parent, insert_position, find_text, replace_text, amendment_type
            )
            insert_position += elements_created

        # Add any remaining text
        if parts[-1]:
            self._add_tail_text(parent, insert_position - 1, parts[-1])

        return len(parts) - 1

    def _replace_occurrences_in_element_text(
        self, element: etree.Element, parts: List[str], find_text: str, replace_text: str, amendment_type: AmendmentType
    ) -> int:
        """
        Replace occurrences in an element's text content with change tracking elements.

        Args:
            element: Element whose text is being processed
            parts: Text parts split by find_text
            find_text: Text being replaced/deleted
            replace_text: New text for substitutions/insertions
            amendment_type: Type of amendment

        Returns:
            Number of occurrences replaced
        """
        if len(parts) <= 1:
            return 0

        # Handle elements that already have children (from previous amendments)
        if len(element) > 0:
            return self._replace_in_element_with_children(element, find_text, replace_text, amendment_type)

        # Handle elements without children (standard case)
        return self._replace_in_empty_element(element, parts, find_text, replace_text, amendment_type)

    def _replace_in_element_with_children(
        self, element: etree.Element, find_text: str, replace_text: str, amendment_type: AmendmentType
    ) -> int:
        """
        Replace text in elements that already have child elements from previous amendments.

        Args:
            element: Element containing existing child elements
            find_text: Text to find and replace
            replace_text: Replacement text
            amendment_type: Type of amendment (insertion, deletion, substitution)

        Returns:
            Number of replacements made
        """
        if not element.text or find_text not in element.text:
            return 0

        # Use word boundaries to avoid matching partial words
        escaped_find = re.escape(find_text)
        pattern = rf"(?<!\w){escaped_find}(?!\w)"

        if not re.search(pattern, element.text):
            return 0

        # Split text using word boundary pattern
        text_parts = re.split(pattern, element.text)
        element.text = text_parts[0]

        # Insert new elements at the beginning
        insert_pos = 0
        for i in range(len(text_parts) - 1):
            elements_created = self._create_tracked_change_elements(
                element, insert_pos, find_text, replace_text, amendment_type
            )
            insert_pos += elements_created

            # Add remaining text after the inserted elements
            if i < len(text_parts) - 1 and text_parts[i + 1]:
                element[insert_pos - 1].tail = text_parts[i + 1]

        return len(text_parts) - 1

    def _replace_in_empty_element(
        self, element: etree.Element, parts: List[str], find_text: str, replace_text: str, amendment_type: AmendmentType
    ) -> int:
        """
        Replace text in elements without any child elements.

        Args:
            element: Element containing only text (no children)
            parts: Pre-split text parts (split by find_text)
            find_text: Text that was found
            replace_text: Replacement text
            amendment_type: Type of amendment (insertion, deletion, substitution)

        Returns:
            Number of replacements made
        """
        # Set initial text
        element.text = parts[0] if parts[0] else None

        # Create tracked changes for each occurrence
        for i in range(len(parts) - 1):
            self._create_tracked_change_elements(element, len(element), find_text, replace_text, amendment_type)

            # Add intermediate text parts
            if i + 1 < len(parts) - 1 and parts[i + 1]:
                element[-1].tail = parts[i + 1]

        # Add final text part
        if parts[-1]:
            if len(element) > 0:
                element[-1].tail = parts[-1]
            else:
                element.text = parts[-1]

        return len(parts) - 1

    def _insert_text_content(self, parent: etree.Element, position: int, text: str) -> int:
        """
        Insert text content at the specified position.

        Args:
            parent: Parent element
            position: Position to insert at
            text: Text to insert

        Returns:
            Updated position after insertion
        """
        if position > 0 and position - 1 < len(parent) and parent[position - 1].tail is None:
            parent[position - 1].tail = text
            return position
        else:
            # Need to create a text-holding element
            text_span = self.xml_handler.create_akn_element("span", text)
            parent.insert(position, text_span)
            return position + 1

    def _add_tail_text(self, parent: etree.Element, index: int, text: str) -> None:
        """
        Add tail text to element at specified index.

        Args:
            parent: Parent element
            index: Index of element to add tail to
            text: Text to add as tail
        """
        if index < len(parent):
            parent[index].tail = text

    def _append_text_to_last_child(self, element: etree.Element, text: str) -> None:
        """
        Append text to last child's tail or element's text.

        Args:
            element: Element to append text to
            text: Text to append
        """
        if len(element) > 0:
            element[-1].tail = (element[-1].tail or "") + text
        else:
            element.text = (element.text or "") + text

    def _create_tracked_change_elements(
        self,
        parent: etree.Element,
        insert_position: int,
        find_text: str,
        replace_text: str,
        amendment_type: AmendmentType,
    ) -> int:
        """
        Create properly tracked change elements for a single occurrence.

        Based on the amendment type, creates the appropriate change tracking
        elements (del, ins, or both) at the specified position in the parent.

        Args:
            parent: Parent element to insert change elements into
            insert_position: Position in parent where to insert elements
            find_text: Text being found/replaced
            replace_text: Replacement text (for insertions/substitutions)
            amendment_type: Type of amendment (DELETION, INSERTION, SUBSTITUTION)

        Returns:
            Number of elements created (1 for deletion, 1-2 for insertion, 2 for substitution)
        """
        if amendment_type == AmendmentType.DELETION:
            return self._create_deletion_change(parent, insert_position, find_text)
        elif amendment_type == AmendmentType.SUBSTITUTION:
            return self._create_substitution_change(parent, insert_position, find_text, replace_text)
        elif amendment_type == AmendmentType.INSERTION:
            return self._create_insertion_change(parent, insert_position, find_text, replace_text)

        # This should never happen as AmendmentType enum only has these three values
        logger.error(f"Unexpected amendment type '{amendment_type}' in _create_tracked_change_elements.")
        return 0

    def _create_deletion_change(self, parent: etree.Element, position: int, text: str) -> int:
        """
        Create a deletion change element.

        Args:
            parent: Parent element to insert into
            position: Position to insert at
            text: Text being deleted

        Returns:
            Number of elements created (always 1)
        """
        del_elem = self.xml_handler.create_akn_element("del", text)
        self.xml_handler.set_change_attributes(del_elem, is_start=True, is_end=True)
        parent.insert(position, del_elem)
        return 1

    def _create_substitution_change(self, parent: etree.Element, position: int, old_text: str, new_text: str) -> int:
        """
        Create substitution change elements (del and ins pair).

        Args:
            parent: Parent element to insert into
            position: Position to insert at
            old_text: Text being replaced
            new_text: Replacement text

        Returns:
            Number of elements created (always 2)
        """
        del_elem = self.xml_handler.create_akn_element("del", old_text)
        self.xml_handler.set_change_attributes(del_elem, is_start=True, is_end=False)

        ins_elem = self.xml_handler.create_akn_element("ins", new_text)
        self.xml_handler.set_change_attributes(ins_elem, is_start=False, is_end=True)

        parent.insert(position, del_elem)
        parent.insert(position + 1, ins_elem)
        return 2

    def _create_insertion_change(self, parent: etree.Element, position: int, find_text: str, replace_text: str) -> int:
        """
        Create insertion change elements.

        Args:
            parent: Parent element to insert into
            position: Position to insert at
            find_text: Reference text for insertion
            replace_text: Full text including insertion

        Returns:
            Number of elements created (1 or 2)
        """
        if replace_text.startswith(find_text):
            # Inserting after
            return self._create_insert_after_elements(parent, position, find_text, replace_text)
        elif replace_text.endswith(find_text):
            # Inserting before
            return self._create_insert_before_elements(parent, position, find_text, replace_text)
        else:
            # Complex insertion - just insert the new text
            ins_elem = self.xml_handler.create_akn_element("ins", replace_text)
            self.xml_handler.set_change_attributes(ins_elem, is_start=True, is_end=True)
            parent.insert(position, ins_elem)
            return 1

    def _create_insert_after_elements(
        self, parent: etree.Element, position: int, find_text: str, replace_text: str
    ) -> int:
        """
        Create elements for inserting text after a reference.

        Args:
            parent: Parent element
            position: Insert position
            find_text: Reference text
            replace_text: Full text with insertion

        Returns:
            Number of elements created (always 2)
        """
        # Keep original text
        text_span = self.xml_handler.create_akn_element("span", find_text)
        parent.insert(position, text_span)

        # Add insertion
        ins_elem = self.xml_handler.create_akn_element("ins", replace_text[len(find_text) :])
        self.xml_handler.set_change_attributes(ins_elem, is_start=True, is_end=True)
        parent.insert(position + 1, ins_elem)
        return 2

    def _create_insert_before_elements(
        self, parent: etree.Element, position: int, find_text: str, replace_text: str
    ) -> int:
        """
        Create elements for inserting text before a reference.

        Args:
            parent: Parent element
            position: Insert position
            find_text: Reference text
            replace_text: Full text with insertion

        Returns:
            Number of elements created (always 2)
        """
        # Add insertion
        ins_elem = self.xml_handler.create_akn_element("ins", replace_text[: -len(find_text)])
        self.xml_handler.set_change_attributes(ins_elem, is_start=True, is_end=True)
        parent.insert(position, ins_elem)

        # Keep original text
        text_span = self.xml_handler.create_akn_element("span", find_text)
        parent.insert(position + 1, text_span)
        return 2
