# app/services/keeling_service.py
"""
Service for processing legislative amendments to create Keeling schedules.
Identifies amendments in amending bills and applies them to target acts
using parallel processing for LLM calls and sequential processing for XML modifications.
"""
import time
import uuid
import copy
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any
from lxml import etree
import csv
from io import StringIO

from ..kernel.llm_kernel import LLMKernel
from ..benchmarking.metrics_logger import MetricsLogger
from ..models.amendments import Amendment, AmendmentLocation, AmendmentType
from .xml_handler import XMLHandler
from .amendment_tracker import AmendmentTracker, AmendmentStatus
from .amendment_processor import AmendmentProcessor
from .utils import (
    csv_to_amendment_dict,
    sort_amendments_by_affected_provision,
    group_amendments_by_target,
    get_amendment_id,
)
from ..logging.debug_logger import get_logger, event, bind, EventType as EVT

logger = get_logger(__name__)


class KeelingService:
    """
    Main service for Keeling schedule generation.

    Processes amending bills to identify amendments and applies them to
    target legislative acts. Uses parallel processing for LLM operations
    and sequential processing for XML modifications to ensure thread safety.
    """

    # Maximum number of worker threads for parallel processing
    MAX_WORKERS = 256

    # Keywords that indicate an amendment in legislative text
    AMENDMENT_KEYWORDS = ["insert", "omit", "repeal", "substitute", "replace", "remove"]

    # Patterns that indicate where amendments can be found
    CONTEXT_PATTERNS = [
        "amended in accordance with",
        "are amended as follows",
        "is amended as follows",
        "amended by",
        "modifications made by",
        "has effect",
        "are amended",
    ]

    def __init__(self, llm_kernel: LLMKernel):
        """
        Initialise the service with required dependencies.

        Args:
            llm_kernel: LLM kernel for amendment identification and processing
        """
        self.llm_kernel = llm_kernel

        # Initialise components
        self.xml_handler = XMLHandler()
        self.amendment_tracker = AmendmentTracker()
        self.amendment_processor = AmendmentProcessor(self.xml_handler, llm_kernel)

        # Initialise metrics logger
        self.metrics_logger = MetricsLogger(
            enable_aws_bedrock=llm_kernel.llm_config.enable_aws_bedrock,
            bedrock_service_id=llm_kernel.llm_config.bedrock_service_id,
            bedrock_model_id=llm_kernel.llm_config.bedrock_model_id,
            enable_azure_openai=llm_kernel.llm_config.enable_azure_openai,
            azure_model_deployment_name=llm_kernel.llm_config.azure_model_deployment_name,
        )

        # Update tracker with logger
        self.amendment_tracker.metrics_logger = self.metrics_logger

        # Store amending bill for reference during application
        self._amending_bill: Optional[etree.ElementTree] = None

        # Store patterns for use during identification
        self._eid_patterns = {}

    # ==================== Public Interface Methods ====================

    def process_amending_bill(
        self, amending_bill_path: str, act_path: str, act_name: str, schedule_id: str
    ) -> List[Amendment]:
        """
        Process an amending bill to identify amendments.

        Args:
            amending_bill_path: Path to the amending bill XML
            act_path: Path to the target act XML being amended
            act_name: Name of the act being amended
            schedule_id: Unique identifier for this schedule

        Returns:
            List of identified amendments
        """
        with bind(schedule_id=schedule_id):
            event(
                logger,
                EVT.SCHEDULE_START,
                f"Processing amending bill for schedule {schedule_id}",
                act_name=act_name,
                bill_path=amending_bill_path,
            )

            # Get file size for logging
            import os

            bill_xml_size = os.path.getsize(amending_bill_path)

            # Log schedule start
            model_id = self.llm_kernel.llm_config.get_active_service_id()
            service_id = model_id  # Use same value for service_id
            self.metrics_logger.log_schedule_start(
                schedule_id=schedule_id,
                act_name=act_name,
                model_id=model_id,
                service_id=service_id,
                max_worker_threads=self.MAX_WORKERS,
                bill_xml_size=bill_xml_size,
                act_xml_size=None,  # Will be updated in apply_amendments
            )

            # Load and normalise the amending bill
            self._amending_bill = self.xml_handler.load_xml(amending_bill_path)

            # Find existing dnums to continue numbering
            existing_dnums = self.xml_handler.find_existing_dnums(self._amending_bill)
            self.xml_handler.set_dnum_counter(existing_dnums)

            # Load the target act and extract eId patterns for accurate identification
            target_act = self.xml_handler.load_xml(act_path)

            # Store target act as instance variable for use in amendment expansion
            self._target_act = target_act

            # ABLATION: eId pattern extraction disabled
            # self._eid_patterns = self.xml_handler.extract_eid_patterns(target_act)
            # logger.info(f"Extracted eId patterns from target Act for schedule {schedule_id}")

            # Create a simplified copy for identification only
            simplified_bill = copy.deepcopy(self._amending_bill)

            # Pre-process amending bill to simplify content and inject amendment context
            self._preprocess_for_identification(simplified_bill, act_name, schedule_id)

            # Get candidate provisions that might contain amendments
            candidates = self._get_candidate_amendments(simplified_bill, act_name, schedule_id)
            logger.info(f"Found {len(candidates)} candidate provisions")

            # Process candidates in parallel to identify amendments
            amendments = self._identify_amendments_parallel(candidates, act_name, schedule_id)
            logger.info(f"Identified {len(amendments)} amendments in schedule {schedule_id}")
            return amendments

    def apply_amendments(
        self, original_act_path: str, amendments: List[Amendment], output_path: str, schedule_id: str
    ) -> None:
        """
        Apply amendments to an act using parallel LLM calls and sequential XML application.

        Args:
            original_act_path: Path to the original act XML
            amendments: List of amendments to apply
            output_path: Where to save the amended act
            schedule_id: Schedule identifier for tracking
        """
        with bind(schedule_id=schedule_id):
            logger.info(f"Applying {len(amendments)} amendments to schedule {schedule_id}")

            # Update act size in logger
            import os

            act_xml_size = os.path.getsize(original_act_path)
            self.metrics_logger.update_schedule_act_size(schedule_id, act_xml_size)

            # Load the original act
            original_act = self.xml_handler.load_xml(original_act_path)

            # Set dnum counter to continue from existing dnums
            existing_dnums = self.xml_handler.find_existing_dnums(original_act)
            self.xml_handler.set_dnum_counter(existing_dnums)

            # Register all amendments with the tracker
            for amendment in amendments:
                self.amendment_tracker.register_amendment_from_object(amendment, schedule_id)

            # Create output document (deep copy of original)
            output_act = copy.deepcopy(original_act)

            # Attempt to algorithmically correct AI mistakes in amendment fields
            self.amendment_processor.correct_amendments(amendments, original_act, self._amending_bill)

            # Sort all amendments by document order
            sorted_amendments = sort_amendments_by_affected_provision(amendments)

            # ABLATION: Each place pattern extraction disabled
            # # Phase 1: Extract and validate patterns for "each place" amendments
            # validated_patterns, pattern_failures = self._extract_and_validate_patterns(sorted_amendments, schedule_id)
            validated_patterns = {}

            # Separate amendments by processing order
            # Validated "Each place" amendments must be applied last to avoid being overwritten
            each_place_validated = []
            amendments_to_apply_first = []

            # ABLATION: Each place separation disabled
            # # Separate amendments by processing order
            # # Validated "Each place" amendments must be applied last to avoid being overwritten
            # each_place_validated = []
            # amendments_to_apply_first = []
            #
            # for amendment in sorted_amendments:
            #     aid = get_amendment_id(amendment)
            #     if amendment.location == AmendmentLocation.EACH_PLACE and aid in validated_patterns:
            #         each_place_validated.append(amendment)
            #     else:
            #         amendments_to_apply_first.append(amendment)
            each_place_validated = []
            amendments_to_apply_first = sorted_amendments  # Process all normally - no each place

            logger.info(
                f"Processing order: {len(amendments_to_apply_first)} regular amendments first, "
                f"then {len(each_place_validated)} validated 'each place' amendments"
            )

            # Phase 2: Determine which amendments need LLM responses
            amendments_needing_llm = []
            for a in amendments_to_apply_first:
                if not a.whole_provision:
                    amendments_needing_llm.append(a)

            # Phase 3: Pre-fetch all LLM responses
            llm_responses = {}
            if amendments_needing_llm:
                logger.info(f"Pre-fetching LLM responses for {len(amendments_needing_llm)} amendments")
                llm_responses = self._fetch_llm_responses_parallel(amendments_needing_llm, original_act, schedule_id)

            # Phase 4: Apply regular LLM amendments first
            logger.info(f"Applying {len(amendments_to_apply_first)} regular amendments")
            for amendment in amendments_to_apply_first:
                self._apply_single_amendment(amendment, output_act, schedule_id, llm_responses, validated_patterns)

            # ABLATION: Each place final application disabled
            # # Phase 5: Apply validated "each place" amendments last
            # if each_place_validated:
            #     logger.info(f"Applying {len(each_place_validated)} validated 'each place' amendments last")
            #     for amendment in each_place_validated:
            #         self._apply_single_amendment(amendment, output_act, schedule_id, {}, validated_patterns)

            # Insert error comments for any failed amendments
            self.amendment_processor.insert_all_error_comments(output_act, self.amendment_tracker)

            # Final cleanup and save
            self.xml_handler.renumber_dnums(output_act, sorted_amendments)
            self.xml_handler.normalise_namespaces(output_act)
            self.xml_handler.normalise_eids(output_act)
            self.xml_handler.insert_editorial_notes(output_act, sorted_amendments)
            self.xml_handler.remove_amendment_ids(output_act)
            self.xml_handler.save_xml(output_act, output_path)

            # Validate all amendments were processed
            resolution = self.amendment_tracker.ensure_all_amendments_resolved()

            # Log completion
            total_applied = len(self.amendment_tracker.get_amendments_by_status(AmendmentStatus.APPLIED))
            self.metrics_logger.log_schedule_end(
                schedule_id=schedule_id, total_amendments_found=len(amendments), total_amendments_applied=total_applied
            )

            event(
                logger,
                EVT.SCHEDULE_END,
                "Amendment processing complete",
                total_found=len(amendments),
                total_applied=total_applied,
                unresolved=resolution["stats"]["unresolved"],
            )

            # Final safety check - ensure no silent failures
            if not resolution["all_resolved"]:
                logger.error(
                    f"WARNING: {resolution['stats']['unresolved']} amendments were not resolved! "
                    f"These amendments have no status and no error comments: {resolution['unresolved_amendments']}"
                )

    # ==================== Amendment Identification Methods ====================

    def _get_candidate_amendments(
        self, tree: etree.ElementTree, act_name: str, schedule_id: str
    ) -> List[Tuple[str, str]]:
        """
        Find provisions that potentially contain amendments.

        Uses keyword search with ancestor awareness to find candidates,
        then filters out descendant provisions to avoid duplication.

        Args:
            tree: XML tree of the amending bill to search
            act_name: Name of the act being amended
            schedule_id: Unique identifier for this schedule

        Returns:
            List of tuples containing (xml_string, eid) for candidate provisions
            that likely contain amendments
        """
        # Use keyword search with ancestor checking
        candidates = self._find_keyword_based_candidates(tree, act_name, schedule_id)

        # Filter out any descendants to avoid duplication
        return self._filter_descendant_candidates(candidates)

    def _find_keyword_based_candidates(
        self, tree: etree.ElementTree, act_name: str, schedule_id: str
    ) -> List[Tuple[str, str]]:
        """
        Find candidates using keyword search with ancestor and sibling awareness.

        Searches for provisions containing amendment keywords (insert, omit, repeal, etc.)
        and checks if the target act is mentioned either in the provision itself, in
        any of its ancestor provisions, or in preceding sibling provisions with context patterns.

        Args:
            tree: XML tree of the amending bill to search
            act_name: Name of the act being amended
            schedule_id: Unique identifier for this schedule

        Returns:
            List of tuples containing (xml_string, eid) for provisions that:
            - Contain amendment keywords
            - Have the target act mentioned in the provision, an ancestor, or a preceding sibling
        """
        potential_candidates = []

        # Find all provisions with amendment keywords
        keyword_provisions = self.xml_handler.find_provisions_containing_text(
            tree, self.AMENDMENT_KEYWORDS, exclude_quoted=True
        )

        for element, eid in keyword_provisions:
            # Check where the act is mentioned
            context_source = self._find_act_mention_context(element, act_name)

            if context_source:
                xml_str = self.xml_handler.element_to_string(element)
                potential_candidates.append((xml_str, eid))

                with bind(schedule_id=schedule_id, candidate_eid=eid):
                    event(
                        logger,
                        EVT.CANDIDATE_FOUND,
                        xml_bytes=len(xml_str),
                        act_refs=True,
                        context_source=context_source,
                    )

        return potential_candidates

    def _find_act_mention_context(self, element: etree.Element, act_name: str) -> Optional[str]:
        """
        Find where the act is mentioned relative to this element.

        Args:
            element: Element to check
            act_name: Name of the act to search for

        Returns:
            "text" if mentioned in element itself
            "ancestor" if mentioned in an ancestor
            "sibling" if mentioned in a preceding sibling
            None if not mentioned
        """
        # Check element itself
        if self._element_mentions_act(element, act_name):
            return "text"

        # ABLATION: Ancestor and sibling checking disabled
        # # Check ancestors
        # current = element.getparent()
        # while current is not None:
        #     if self._is_provision(current) and self._element_mentions_act(current, act_name):
        #         logger.debug(f"Found act mention in ancestor of {element.get('eId', '')}")
        #         return "ancestor"
        #     current = current.getparent()

        # # Check preceding siblings
        # parent = element.getparent()
        # if parent is not None:
        #     for sibling in parent:
        #         if sibling == element:
        #             break  # Stop at current element
        #
        #         if self._is_provision(sibling) and self._element_mentions_act(sibling, act_name):
        #             sibling_text = self.xml_handler.get_text_content(sibling, exclude_quoted=True)
        #             # For siblings, also check if they have context patterns
        #             if any(pattern in sibling_text.lower() for pattern in self.CONTEXT_PATTERNS):
        #                 logger.debug(f"Found act mention in preceding sibling of {element.get('eId', '')}")
        #                 return "sibling"

        return None

    def _is_provision(self, element: etree.Element) -> bool:
        """Check if an element represents a legislative provision.

        Args:
            element: XML element to check

        Returns:
            True if the element is a provision-like element, False otherwise
        """
        return (
            element.get("class") in ["prov1", "prov2", "schProv1", "schProv2", "sch"]
            or element.get("name") in ["regulation", "section", "article", "rule", "schedule"]
            or "Group" in element.get("class", "")
        )

    def _element_mentions_act(self, element: etree.Element, act_name: str) -> bool:
        """Check if an element contains a reference to the specified act.

        Args:
            element: XML element to check
            act_name: Name of the act to search for

        Returns:
            True if the act name is found in text or comments, False otherwise
        """
        text_content = self.xml_handler.get_text_content(element, exclude_quoted=True)
        comment_content = self.xml_handler.get_comment_content(element)
        return act_name.lower() in text_content.lower() or act_name.lower() in comment_content.lower()

    def _filter_descendant_candidates(self, candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Remove any candidates that are descendants of other candidates.

        This prevents duplicate processing when a parent provision and its
        child provisions both contain amendments. We only want to process
        the highest-level provision that contains amendments.

        Args:
            candidates: List of tuples containing (xml_string, eid) for all candidates

        Returns:
            Filtered list with descendant provisions removed, keeping only the
            highest-level provisions that contain amendments
        """
        final_candidates = []
        candidate_eids = [eid for _, eid in candidates]

        for xml_str, eid in candidates:
            # Check if this eid is a descendant of any other candidate
            is_descendant = False
            for other_eid in candidate_eids:
                if eid != other_eid and eid.startswith(other_eid + "__"):
                    is_descendant = True
                    logger.debug(f"Filtering out {eid} as it's a descendant of {other_eid}")
                    break

            if not is_descendant:
                final_candidates.append((xml_str, eid))

        return final_candidates

    def _identify_amendments_parallel(
        self, candidates: List[Tuple[str, str]], act_name: str, schedule_id: str
    ) -> List[Amendment]:
        """
        Identify amendments from candidates using parallel processing.

        Args:
            candidates: List of (xml_string, eid) tuples
            act_name: Name of the act being amended
            schedule_id: Schedule identifier

        Returns:
            List of identified amendments
        """
        amendments = []
        identified_candidates = 0
        start_t = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit all candidates for processing
            future_to_candidate = {
                executor.submit(self._identify_single_candidate, xml_provision, eid, act_name, schedule_id): (
                    xml_provision,
                    eid,
                )
                for xml_provision, eid in candidates
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_candidate):
                try:
                    result = future.result()
                    if result:
                        amendments.extend(result)
                        identified_candidates += 1
                except Exception as e:
                    xml_provision, eid = future_to_candidate[future]
                    logger.error(f"Error processing candidate {eid}: {e}")

        event(
            logger,
            EVT.IDENTIFICATION_SUMMARY,
            total_candidates=len(candidates),
            identified=identified_candidates,
            skipped=len(candidates) - identified_candidates,
            amendments=len(amendments),
            duration_s=time.time() - start_t,
        )

        return amendments

    def _identify_single_candidate(
        self, xml_provision: str, provision_eid: str, act_name: str, schedule_id: str
    ) -> List[Amendment]:
        """
        Identify amendments in a single candidate provision.

        Args:
            xml_provision: XML string of the provision
            provision_eid: eId of the provision
            act_name: Name of the act being amended
            schedule_id: Schedule identifier

        Returns:
            List of amendments found in this provision
        """
        try:
            start_time = time.time()

            # Call LLM to identify amendments
            response = self.llm_kernel.run_inference(
                "TableOfAmendments",
                schedule_id,
                None,
                provision_eid,
                act_name=act_name,
                xml_provision=xml_provision,
                # ABLATION: eId pattern injection disabled
                # eid_patterns=json.dumps(self._eid_patterns) if self._eid_patterns else "{}",
                eid_patterns="{}",
            )

            # Parse CSV response with target act for range expansion
            amendment_data_list = csv_to_amendment_dict(
                response, target_act=getattr(self, "_target_act", None), xml_handler=self.xml_handler
            )

            # Calculate identification time
            ident_duration = time.time() - start_time

            # Track statistics for logging
            other_act_amendments = []
            target_act_count = 0

            # First pass: analyse what was returned
            for data in amendment_data_list:
                amendment = Amendment.from_dict(data)
                if amendment.affected_document.lower() == act_name.lower():
                    target_act_count += 1
                else:
                    other_act_amendments.append(amendment.affected_document)

            # Log filtering statistics if amendments to other acts were found
            if other_act_amendments:
                unique_other_acts = list(set(other_act_amendments))
                logger.info(
                    f"Candidate {provision_eid}: identified {target_act_count} amendments to {act_name}, "
                    f"filtered out {len(other_act_amendments)} amendments to other acts: {', '.join(unique_other_acts)}"
                )

            # Second pass: create and log only target act amendments
            amendments = []
            for data in amendment_data_list:
                # Create Amendment object to check affected_document
                amendment = Amendment.from_dict(data)

                # Filter out amendments whose affected_document is not the target act
                if amendment.affected_document.lower() != act_name.lower():
                    continue

                # Generate amendment ID only for valid amendments
                amendment_id = str(uuid.uuid4())
                data["amendment_id"] = amendment_id
                amendment.amendment_id = amendment_id

                # Log the amendment (only for target act)
                self.metrics_logger.log_amendment(
                    schedule_id=schedule_id,
                    amendment_id=amendment_id,
                    source=data.get("source", ""),
                    source_eid=data.get("source_eid", ""),
                    affected_provision=data.get("affected_provision", ""),
                    location=data.get("location", ""),
                    amendment_type=data.get("type_of_amendment", ""),
                    whole_provision=data.get("whole_provision", False),
                    identification_time_seconds=(
                        ident_duration / target_act_count if target_act_count else ident_duration
                    ),
                )

                amendments.append(amendment)

            with bind(schedule_id=schedule_id, candidate_eid=provision_eid):
                event(
                    logger,
                    EVT.CANDIDATE_IDENTIFIED,
                    amendments=len(amendments),
                    total_identified=len(amendment_data_list),
                    filtered_out=len(amendment_data_list) - len(amendments),
                )

            return amendments

        except ValueError as e:
            error_msg = str(e)
            if "No valid amendments found" in error_msg:
                reason = "NO_AMENDMENTS_FOUND"
            elif "no headers" in error_msg.lower():
                reason = "EMPTY_RESPONSE"
            else:
                reason = "INVALID_CSV"

            with bind(schedule_id=schedule_id, candidate_eid=provision_eid):
                event(logger, EVT.CANDIDATE_SKIPPED, reason=reason, error=error_msg)
            return []

        except Exception as e:
            with bind(schedule_id=schedule_id, candidate_eid=provision_eid):
                event(logger, EVT.CANDIDATE_SKIPPED, reason="LLM_ERROR", error=str(e))
            return []

    def _preprocess_for_identification(self, bill_tree: etree.ElementTree, act_name: str, schedule_id: str) -> None:
        """
        Orchestrate preprocessing steps for amendment identification.

        Performs three stages of preprocessing:
        1. Simplifies the amending bill XML
        2. Injects crossheading context for schedule provisions
        3. Injects context for regulation/section references

        Args:
            bill_tree: The bill XML tree to preprocess (modified in place)
            act_name: Name of the act being amended
            schedule_id: Schedule identifier for logging

        Returns:
            None (modifies bill_tree in place)
        """
        with bind(schedule_id=schedule_id):
            # Stage 1: Simplify the XML
            logger.info("Preprocessing stage 1: Simplifying amending bill")
            self.xml_handler.simplify_amending_bill(bill_tree)

            # ABLATION: Context injection disabled
            # # Stage 2: Inject crossheading context
            # logger.info("Preprocessing stage 2: Injecting crossheading context")
            # crossheading_count = self._inject_crossheading_context(bill_tree)
            # logger.info(f"Injected crossheading context into {crossheading_count} provisions")

            # # Stage 3: Context injection for regulation/section patterns
            # logger.info("Preprocessing stage 3: Context injection for regulation patterns")
            # injected_count = self._inject_document_context(bill_tree, act_name, schedule_id)
            # logger.info(f"Context injection complete: modified {injected_count} provisions")

    def _inject_crossheading_context(self, tree: etree.ElementTree) -> int:
        """
        Inject crossheading text as context into child provisions.

        Processes all schedules in the document and injects hierarchical
        crossheading context as XML comments into provisions. This helps
        the LLM understand which act/provision is being amended when this
        information only appears in parent crossheadings.

        Args:
            tree: XML tree to process (modified in place)

        Returns:
            Number of provisions that had context injected
        """
        injected_count = 0

        # Find all schedules
        schedules = tree.xpath(".//akn:hcontainer[@name='schedule']", namespaces=self.xml_handler.namespaces)

        for schedule in schedules:
            schedule_eid = schedule.get("eId", "unknown")

            # Get schedule heading
            schedule_context = self.xml_handler.get_schedule_heading_text(schedule)
            if schedule_context:
                logger.debug(f"Schedule {schedule_eid} heading: '{schedule_context}'")

            # Find all crossheadings in this schedule
            crossheadings = self.xml_handler.get_crossheadings_in_schedule(schedule)
            logger.debug(f"Found {len(crossheadings)} crossheadings in schedule {schedule_eid}")

            for xheading in crossheadings:
                xheading_eid = xheading.get("eId", "unknown")

                # Get the crossheading text
                heading_elem = xheading.find(".//akn:heading", self.xml_handler.namespaces)
                if heading_elem is None:
                    logger.debug(f"No heading found in crossheading {xheading_eid}")
                    continue

                heading_text = self.xml_handler.get_text_content(heading_elem)
                logger.debug(f"Processing crossheading {xheading_eid}: '{heading_text}'")

                # Build cumulative context from ancestors
                ancestor_contexts = []
                if schedule_context:
                    ancestor_contexts.append(schedule_context)

                parent_xheading_contexts = self.xml_handler.get_ancestor_crossheading_contexts(xheading)
                ancestor_contexts.extend(parent_xheading_contexts)

                # Create full context string
                full_context = " > ".join(ancestor_contexts + [heading_text]) if ancestor_contexts else heading_text
                logger.debug(f"Full context for {xheading_eid}: '{full_context}'")

                # Inject into direct child provisions
                child_provisions = self.xml_handler.get_crossheading_child_provisions(xheading)
                logger.debug(f"Found {len(child_provisions)} child provisions")

                for child in child_provisions:
                    child_eid = child.get("eId", "unknown")
                    comment_text = f" Crossheading context: {full_context} "
                    self.xml_handler.inject_xml_comment(child, comment_text)
                    injected_count += 1
                    logger.info(f"Injected into {child_eid}: '{full_context}'")

        return injected_count

    def _inject_document_context(self, tree: etree.ElementTree, act_name: str, schedule_id: str) -> int:
        """
        Inject document context into provisions that need it for identification.

        Args:
            tree: XML tree to modify (in place)
            act_name: Name of the act being amended
            schedule_id: Schedule identifier for logging

        Returns:
            Number of provisions modified with injected context
        """
        injection_count = 0

        # Find context provisions
        context_provisions = self._find_context_provisions(tree, act_name)
        logger.info(f"Found {len(context_provisions)} context provisions for {act_name}")

        if not context_provisions:
            return 0

        # Phase 1: Parallelise LLM calls to identify all references
        all_references = self._identify_all_references_parallel(context_provisions, act_name, schedule_id)

        # Phase 2: Deduplicate and collect all target provisions
        # Map from target eId to set of source context eIds
        target_to_sources = {}

        for (context_elem, context_eid), references in all_references.items():
            for prov_type, start_num, end_num in references:
                if start_num == end_num:
                    # Single provision
                    target_elem = self.xml_handler.find_provision_by_type_and_number(
                        tree, prov_type, start_num, context_elem
                    )
                    if target_elem is not None:
                        target_eid = target_elem.get("eId", "")
                        if target_eid not in target_to_sources:
                            target_to_sources[target_eid] = {"element": target_elem, "source_eids": set()}
                        target_to_sources[target_eid]["source_eids"].add(context_eid)
                else:
                    # Range - find all provisions in numeric range
                    target_elems = self.xml_handler.find_provisions_in_range_by_number(
                        tree, prov_type, start_num, end_num, context_elem
                    )
                    for target_elem in target_elems:
                        target_eid = target_elem.get("eId", "")
                        if target_eid not in target_to_sources:
                            target_to_sources[target_eid] = {"element": target_elem, "source_eids": set()}
                        target_to_sources[target_eid]["source_eids"].add(context_eid)

        # Phase 3: Inject context only once per target provision
        for target_eid, target_info in target_to_sources.items():
            target_elem = target_info["element"]
            source_eids = target_info["source_eids"]

            if self._should_inject_context(target_elem, act_name, tree):
                # Combine all source references
                if len(source_eids) == 1:
                    source_ref = f"from {next(iter(source_eids))}"
                else:
                    # Sort for consistent output
                    sorted_sources = sorted(source_eids)
                    source_ref = f"from {', '.join(sorted_sources)}"

                self._inject_act_reference(target_elem, act_name, source_ref)
                injection_count += 1
                logger.debug(f"Injected context into {target_eid} ({source_ref})")

        return injection_count

    def _identify_all_references_parallel(
        self, context_provisions: List[Tuple[etree.Element, str]], act_name: str, schedule_id: str
    ) -> Dict[Tuple[etree.Element, str], List[Tuple[str, str, str]]]:
        """
        Identify references in all context provisions using parallel LLM calls.

        Args:
            context_provisions: List of tuples (element, eId) for context provisions
            act_name: Name of the act being amended
            schedule_id: Schedule identifier for logging

        Returns:
            Dictionary mapping (element, eId) tuples to lists of references.
            Each reference is a tuple of (provision_type, start_number, end_number).
            Empty list for provisions where no references were found.
        """
        all_references = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit all context provisions for processing
            future_to_context = {
                executor.submit(self._identify_single_context_references, elem, act_name, schedule_id): (elem, eid)
                for elem, eid in context_provisions
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_context):
                elem, eid = future_to_context[future]
                try:
                    references = future.result()
                    if references:
                        all_references[(elem, eid)] = references
                        logger.debug(f"Found {len(references)} references in context provision {eid}")
                    else:
                        # Store empty list to indicate we processed it but found nothing
                        all_references[(elem, eid)] = []
                        logger.debug(f"No references found in context provision {eid}")
                except Exception as e:
                    logger.error(f"Error identifying references for {eid}: {e}")
                    # Store empty list on error to continue processing
                    all_references[(elem, eid)] = []

        logger.info(
            f"Processed {len(context_provisions)} context provisions, "
            f"found references in {sum(1 for refs in all_references.values() if refs)} provisions"
        )

        return all_references

    def _identify_single_context_references(
        self, context_elem: etree.Element, act_name: str, schedule_id: str
    ) -> List[Tuple[str, str, str]]:
        """
        Use LLM to identify which provisions are referenced in a single context element.

        Args:
            context_elem: The context provision element to analyse
            act_name: Name of the act being amended
            schedule_id: Schedule identifier for logging

        Returns:
            List of tuples (provision_type, start_number, end_number) representing
            the provisions referenced as containing amendments. Returns empty list
            if no references found or on error.
        """
        try:
            context_xml = self.xml_handler.element_to_string(context_elem)
            context_eid = context_elem.get("eId", "unknown")

            with bind(schedule_id=schedule_id, context_eid=context_eid):
                logger.debug(f"Calling LLM to parse references in {context_eid}")

                # Call the LLM prompt
                response = self.llm_kernel.run_inference(
                    "IdentifyAmendmentReferences",
                    schedule_id,
                    None,  # No amendment_id
                    context_eid,  # Use as candidate_eid for logging
                    context_provision=context_xml,
                    act_name=act_name,
                )

                # Parse CSV response
                references = []
                if response:
                    lines = response.strip().split("\n")
                    # Skip header if present
                    if lines and "provision_type" in lines[0]:
                        lines = lines[1:]

                    for line in lines:
                        if line.strip():
                            parts = line.strip().split(",")
                            if len(parts) >= 3:
                                prov_type = parts[0].strip()
                                start_num = parts[1].strip()
                                end_num = parts[2].strip()
                                references.append((prov_type, start_num, end_num))

                logger.debug(f"LLM identified {len(references)} provision references in {context_eid}")
                return references

        except Exception as e:
            logger.error(f"Failed to identify references in {context_elem.get('eId', 'unknown')}: {e}")
            return []

    def _find_context_provisions(self, tree: etree.ElementTree, act_name: str) -> List[Tuple[etree.Element, str]]:
        """
        Find provisions that contain context about amendments to the act.

        Args:
            tree: XML tree to search
            act_name: Name of the act being amended

        Returns:
            List of tuples (element, eId) for context provisions
        """
        # First find all provisions mentioning the act
        act_provisions = self.xml_handler.find_provisions_containing_text(tree, [act_name], exclude_quoted=True)

        context_provisions = []

        for element, eid in act_provisions:
            text_content = self.xml_handler.get_text_content(element, exclude_quoted=True).lower()

            # Check if any context pattern is present
            if any(pattern in text_content for pattern in self.CONTEXT_PATTERNS):
                # Additional check: ensure this provision doesn't contain amendment keywords
                # (if it does, it's an amendment provision, not a context provision)
                has_amendment_keywords = any(keyword in text_content for keyword in self.AMENDMENT_KEYWORDS)

                if not has_amendment_keywords:
                    context_provisions.append((element, eid))
                    logger.debug(f"Found context provision: {eid}")

        return context_provisions

    def _should_inject_context(self, element: etree.Element, act_name: str, tree: etree.ElementTree) -> bool:
        """
        Determine if context should be injected into this element.

        Args:
            element: Element to evaluate for context injection
            act_name: Name of the act being amended
            tree: XML tree containing the element

        Returns:
            True if context should be injected, False if element already has
            sufficient context or doesn't need it
        """
        element_eid = element.get("eId", "")

        # Skip if element already mentions the act
        text_content = self.xml_handler.get_text_content(element, exclude_quoted=True)
        if act_name.lower() in text_content.lower():
            logger.debug(f"Skip {element_eid}: already contains act name")
            return False

        # Skip if any ancestor provision mentions the act
        current = element.getparent()
        while current is not None:
            # Check only provision elements, not structural containers
            is_provision = current.get("class") in ["prov1", "prov2", "schProv1", "schProv2"] or current.get(
                "name"
            ) in ["regulation", "section", "article", "rule"]

            if is_provision:
                parent_text = self.xml_handler.get_text_content(current, exclude_quoted=True)
                if act_name.lower() in parent_text.lower():
                    logger.debug(f"Skip {element_eid}: ancestor provision {current.get('eId')} contains act name")
                    return False

            current = current.getparent()

        # Skip if element doesn't contain amendment keywords
        if not any(keyword in text_content.lower() for keyword in self.AMENDMENT_KEYWORDS):
            logger.debug(f"Skip {element_eid}: no amendment keywords")
            return False

        return True

    def _inject_act_reference(self, element: etree.Element, act_name: str, source_ref: str) -> None:
        """
        Inject act reference into an element using xml_handler methods.

        Args:
            element: Element to modify
            act_name: Name of the act to inject
            source_ref: Reference string describing the source(s) of this context
        """
        # Create comment with act reference
        comment_text = f" Amendment context: {act_name} ({source_ref}) "
        self.xml_handler.inject_xml_comment(element, comment_text)

        # Also set namespaced attributes as backup
        self.xml_handler.set_namespaced_attribute(element, self.xml_handler.UKL_URI, "contextAct", act_name)
        self.xml_handler.set_namespaced_attribute(element, self.xml_handler.UKL_URI, "contextSource", source_ref)

    # ==================== Amendment Application Methods ====================

    def _fetch_llm_responses_parallel(
        self, amendments: List[Amendment], original_act: etree.ElementTree, schedule_id: str
    ) -> Dict[str, str]:
        """
        Fetch LLM responses for partial amendments using hybrid parallel/sequential processing.

        Groups amendments by target provision and processes:
        - Different provisions in parallel (maintains speed)
        - Same provision sequentially (ensures correctness)

        For provisions with multiple amendments, only the final cumulative response
        is returned since it contains all changes.

        Args:
            amendments: List of partial text amendments
            original_act: Original act tree
            schedule_id: Schedule identifier

        Returns:
            Dictionary mapping amendment_id to LLM response
            Note: For grouped amendments targeting the same provision, each amendment
            has its own response that incorporates all previous amendments' changes
        """
        # Group amendments by their target provision
        amendment_groups = group_amendments_by_target(amendments)

        logger.info(f"Processing {len(amendments)} amendments in {len(amendment_groups)} groups")

        responses = {}
        # Track amendments that failed during fetch phase with their error messages
        failed_amendments = {}  # amendment_id -> error_message

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit each group for processing
            future_to_group = {
                executor.submit(
                    self._process_amendment_group, group_amendments, original_act, schedule_id, target_eid
                ): (target_eid, group_amendments)
                for target_eid, group_amendments in amendment_groups.items()
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_group):
                target_eid, group_amendments = future_to_group[future]

                try:
                    group_result = future.result()
                    # Check if result contains responses and failures
                    if isinstance(group_result, dict) and "responses" in group_result:
                        responses.update(group_result["responses"])
                        failed_amendments.update(group_result.get("failures", {}))
                    else:
                        # Otherwise just responses
                        responses.update(group_result)

                    if responses:
                        logger.debug(f"Completed processing group for {target_eid} with {len(responses)} responses")
                    else:
                        logger.debug(f"No successful responses for group {target_eid}")
                except Exception as e:
                    # This should rarely happen - only for unexpected errors
                    # Individual amendment failures are already handled in _process_amendment_group
                    logger.error(f"Unexpected error processing group for {target_eid}: {e}")
                    logger.exception("Full traceback:")

        # Store failed amendments info for use in _apply_single_amendment
        self._fetch_phase_failures = failed_amendments

        return responses

    def _process_amendment_group(
        self, amendments: List[Amendment], original_act: etree.ElementTree, schedule_id: str, target_eid: str
    ) -> Dict[str, Any]:
        """
        Process a group of amendments that target the same provision.

        For groups with multiple amendments, processes them sequentially
        to ensure each LLM call sees cumulative changes.

        Args:
            amendments: List of amendments targeting the same provision
            original_act: Original act tree
            schedule_id: Schedule identifier
            target_eid: The target provision eId being amended

        Returns:
            Dictionary containing:
            - responses: mapping amendment_id to LLM response for successful amendments
            - failures: mapping amendment_id to error message for failed amendments
        """
        if not amendments:
            return {"responses": {}, "failures": {}}

        logger.debug(f"Processing {len(amendments)} amendments for {target_eid}")

        responses = {}
        failures = {}
        last_successful_response = None

        # Process amendments sequentially
        for i, amendment in enumerate(amendments):
            aid = get_amendment_id(amendment)
            if not aid:
                continue

            with bind(schedule_id=schedule_id, amendment_id=aid):
                # Determine which tree to use
                if i == 0 or not last_successful_response:
                    # First amendment or no successful responses yet - use original tree
                    tree_to_use = original_act
                else:
                    # Create working tree with last successful response applied
                    working_tree = copy.deepcopy(original_act)
                    target = self.xml_handler.find_element_by_eid(working_tree, target_eid)

                    if target is not None:
                        try:
                            amended_element = self.xml_handler.parse_xml_string(
                                last_successful_response, ensure_namespaces=True
                            )
                            parent = target.getparent()
                            if parent is not None:
                                parent.replace(target, amended_element)
                                tree_to_use = working_tree
                            else:
                                logger.warning(f"No parent found for {target_eid}, using last successful state")
                                tree_to_use = original_act
                        except Exception as e:
                            logger.warning(
                                f"Failed to apply previous response for amendment {aid}: {e}. "
                                f"Using last successful state."
                            )
                            tree_to_use = original_act
                    else:
                        logger.warning(f"Target {target_eid} not found in working tree, using last successful state")
                        tree_to_use = original_act

                # Fetch LLM response
                response = self._fetch_single_amendment_response(amendment, tree_to_use, schedule_id)

                if response:
                    # Try to parse and inject amendment id, but don't fail if it doesn't work
                    try:
                        amended = self.xml_handler.parse_xml_string(response, ensure_namespaces=True)
                        self.xml_handler.inject_amendment_id(amended, amendment)
                        response = self.xml_handler.element_to_string(amended)
                    except Exception as exc:
                        # Keep response unchanged
                        logger.warning(f"Amendment ID injection skipped for amendment {aid}: {exc}")

                    responses[aid] = response
                    last_successful_response = response
                    logger.debug(f"Successfully processed amendment {aid}")
                else:
                    # Amendment failed during fetch
                    # First check if it's already marked as failed in the tracker (for any error message)
                    record = self.amendment_tracker.get_amendment(aid)
                    if record and record.status == AmendmentStatus.FAILED and record.error_message:
                        # Use the error message from the tracker
                        failures[aid] = record.error_message
                    else:
                        # Fallback error message if not in tracker or no error message
                        failures[aid] = "Failed to fetch LLM response"

                    logger.debug(f"Amendment {aid} failed during fetch: {failures[aid]}")

        return {"responses": responses, "failures": failures}

    def _fetch_single_amendment_response(
        self, amendment: Amendment, act_tree: etree.ElementTree, schedule_id: str
    ) -> Optional[str]:
        """
        Fetch LLM response for a single amendment.

        Args:
            amendment: Amendment to process
            act_tree: Tree to use for finding target (original or working)
            schedule_id: Schedule identifier

        Returns:
            LLM response string or None if failed
        """
        aid = get_amendment_id(amendment)

        try:
            with bind(schedule_id=schedule_id, amendment_id=aid):
                # Find target and source elements
                target = self.xml_handler.find_element_by_eid(act_tree, amendment.affected_provision)
                source_element = self.xml_handler.find_element_by_eid(self._amending_bill, amendment.source_eid)

                if target is None:
                    if aid:
                        self.amendment_tracker.mark_failed(
                            aid, f"Target element {amendment.affected_provision} not found", error_location="llm_fetch"
                        )
                    return None

                if source_element is None:
                    if aid:
                        self.amendment_tracker.mark_failed(
                            aid, f"Source element {amendment.source_eid} not found", error_location="llm_fetch"
                        )
                    return None

                # Check token limits
                is_within_limits, token_estimate = self.amendment_processor.check_token_limits(
                    amendment, target, source_element
                )

                if not is_within_limits:
                    # Mark as failed with user-friendly error message
                    error_msg = (
                        f"Amendment in {amendment.source} was not applied because "
                        f"the affected provision is too large to process."
                    )
                    logger.debug(
                        f"Amendment {aid} exceeds token limit. "
                        f"Estimated output tokens: {token_estimate}, "
                        f"Max allowed: {self.llm_kernel.llm_config.get_max_completion_tokens()}"
                    )
                    if aid:
                        self.amendment_tracker.mark_failed(aid, error_msg, error_location="token_limit_check")
                    return None

                # Prepare LLM call data
                llm_data = self.amendment_processor.prepare_llm_amendment(
                    amendment, target, source_element, schedule_id
                )

                # Call LLM using the kernel directly
                response = self.llm_kernel.run_inference(
                    llm_data["prompt_name"], llm_data["schedule_id"], llm_data["amendment_id"], **llm_data["kwargs"]
                )

                return response

        except Exception as e:
            if aid:
                self.amendment_tracker.mark_failed(aid, f"LLM call failed: {e}", error_location="llm_fetch")
            return None

    def _apply_single_amendment(
        self,
        amendment: Amendment,
        output_act: etree.ElementTree,
        schedule_id: str,
        llm_responses: Dict[str, str],
        validated_patterns: Dict[str, Dict[str, str]],
    ) -> None:
        """
        Apply a single amendment to the output document.

        Handles both whole provision amendments (structural) and partial
        amendments (using pre-fetched LLM responses). Tracks the amendment
        status and timing throughout the process.

        Args:
            amendment: Amendment to apply
            output_act: Output document tree to modify
            schedule_id: Schedule identifier for tracking
            llm_responses: Pre-fetched LLM responses for partial amendments
            validated_patterns: Validated patterns for "each place" amendments
        """
        aid = get_amendment_id(amendment)

        try:
            with bind(schedule_id=schedule_id, amendment_id=aid):
                # Mark as applying
                if aid:
                    self.amendment_tracker.mark_applying(aid)

                start_time = time.time()
                success = False
                error_msg = None

                # ABLATION: Each place algorithmic application disabled
                # # Check if this is an "each place" amendment with validated pattern
                # if amendment.location == AmendmentLocation.EACH_PLACE and aid in validated_patterns:
                #     # Apply algorithmically with validated pattern
                #     success, error_msg = self.amendment_processor.apply_each_place_amendment(
                #         amendment, output_act, self._amending_bill, validated_patterns[aid], schedule_id
                #     )
                #
                #     if not success:
                #         logger.error(
                #             f"Validated pattern failed to apply for amendment {aid}. "
                #             f"This should not happen - pattern was pre-validated. Error: {error_msg}"
                #         )
                #
                # elif amendment.whole_provision:
                if amendment.whole_provision:  # Skip straight to this condition
                    # Apply structural amendment directly
                    success, error_msg = self.amendment_processor.apply_amendment(
                        amendment, output_act, self._amending_bill, schedule_id
                    )
                else:
                    # Apply using pre-fetched LLM response
                    llm_response = llm_responses.get(aid)
                    if llm_response:
                        success, error_msg = self.amendment_processor.apply_amendment(
                            amendment, output_act, self._amending_bill, schedule_id, llm_response
                        )
                    else:
                        success = False
                        # Check if this amendment failed during fetch phase
                        if hasattr(self, "_fetch_phase_failures") and aid in self._fetch_phase_failures:
                            # Use the original error message from fetch phase
                            error_msg = self._fetch_phase_failures[aid]
                        else:
                            # This should not happen if pipeline is working correctly
                            error_msg = "No LLM response available (pipeline error - should have been pre-fetched)"
                            logger.error(f"Pipeline error: No LLM response for amendment {aid}")

                duration = time.time() - start_time

                if success:
                    if aid:
                        self.amendment_tracker.mark_applied(aid, duration)
                else:
                    if aid:
                        self.amendment_tracker.mark_failed(aid, error_msg or "Unknown error", processing_time=duration)

        except Exception as e:
            logger.exception(f"Unexpected error applying amendment {aid}")
            if aid:
                self.amendment_tracker.mark_failed(aid, str(e))

    def _extract_and_validate_patterns(
        self, amendments: List[Amendment], schedule_id: str
    ) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
        """
        Extract patterns and validate they can be applied.

        Args:
            amendments: List of all amendments
            schedule_id: Schedule identifier for tracking

        Returns:
            Tuple of (validated_patterns, failed_amendment_ids)
        """
        # First extract patterns
        all_patterns = self._extract_amendment_patterns(amendments, schedule_id)

        # Then validate each pattern
        validated_patterns = {}
        validation_failures = []

        logger.info(f"Validating {len(all_patterns)} extracted patterns")

        for aid, pattern in all_patterns.items():
            # Find the amendment
            amendment = next((a for a in amendments if get_amendment_id(a) == aid), None)
            if not amendment:
                logger.warning(f"Could not find amendment for pattern validation: {aid}")
                validation_failures.append(aid)
                continue

            # Validate the pattern can be applied
            if self._validate_pattern_application(amendment, pattern):
                validated_patterns[aid] = pattern
                logger.debug(f"Pattern validated for amendment {aid}")
            else:
                validation_failures.append(aid)
                logger.warning(f"Pattern validation failed for amendment {aid}")

                # Mark as failed immediately
                self.amendment_tracker.mark_failed(
                    aid, "Pattern validation failed - will use LLM approach", error_location="pattern_validation"
                )

        logger.info(
            f"Pattern validation complete: {len(validated_patterns)} validated, " f"{len(validation_failures)} failed"
        )

        return validated_patterns, validation_failures

    def _extract_amendment_patterns(self, amendments: List[Amendment], schedule_id: str) -> Dict[str, Dict[str, str]]:
        """
        Extract find/replace patterns for "each place" amendments using LLM.

        Args:
            amendments: List of all amendments
            schedule_id: Schedule identifier for tracking

        Returns:
            Dictionary mapping amendment_id to pattern data (find_text, replace_text)
        """
        # Filter to only "each place" amendments
        each_place_amendments = [a for a in amendments if a.location == AmendmentLocation.EACH_PLACE]

        if not each_place_amendments:
            return {}

        logger.info(f"Extracting patterns for {len(each_place_amendments)} 'each place' amendments")
        patterns = {}
        failed_extractions = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit all pattern extraction tasks
            future_to_amendment = {
                executor.submit(self._extract_single_pattern, amendment, schedule_id): amendment
                for amendment in each_place_amendments
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_amendment):
                amendment = future_to_amendment[future]
                aid = get_amendment_id(amendment)

                try:
                    pattern_data = future.result()
                    if pattern_data:
                        patterns[aid] = pattern_data
                        event(
                            logger,
                            EVT.PATTERN_EXTRACTION_SUCCESS,
                            f"Extracted pattern for amendment {aid}",
                            amendment_id=aid,
                            find_text=pattern_data.get("find_text"),
                            replace_text=pattern_data.get("replace_text"),
                        )
                    else:
                        failed_extractions.append(aid)
                        event(
                            logger,
                            EVT.PATTERN_EXTRACTION_FAILED,
                            f"Failed to extract pattern for amendment {aid}",
                            amendment_id=aid,
                        )
                except Exception as e:
                    failed_extractions.append(aid)
                    logger.error(f"Error extracting pattern for amendment {aid}: {e}")
                    event(
                        logger,
                        EVT.PATTERN_EXTRACTION_FAILED,
                        f"Error extracting pattern for amendment {aid}",
                        amendment_id=aid,
                        error=str(e),
                    )

        logger.info(
            f"Pattern extraction complete: {len(patterns)} successful, "
            f"{len(failed_extractions)} failed (will use LLM fallback)"
        )

        return patterns

    def _validate_pattern_application(self, amendment: Amendment, pattern: Dict[str, str]) -> bool:
        """
        Validate that a pattern can be successfully applied.

        Args:
            amendment: Amendment to validate
            pattern: Pattern data with find_text and replace_text

        Returns:
            True if pattern can be applied, False otherwise
        """
        try:
            find_text = pattern.get("find_text", "")
            if not find_text:
                logger.debug("Pattern validation failed: no find_text")
                return False

            # Find the target element in the original target act
            target = self.xml_handler.find_element_by_eid(self._target_act, amendment.affected_provision)
            if target is None:
                logger.debug(f"Pattern validation failed: target element {amendment.affected_provision} not found")
                return False

            # Do a dry run to check for issues
            validation_changes = []

            # Create a deep copy of the target element for validation
            validation_element = copy.deepcopy(target)

            try:
                # Use the amendment processor to simulate the pattern application
                self.amendment_processor._replace_text_occurrences_iteratively(
                    validation_element,
                    find_text,
                    pattern.get("replace_text", ""),
                    amendment.amendment_type,
                    validation_changes,
                )

                # Check if any occurrences were found
                if len(validation_changes) == 0:
                    logger.debug(f"Pattern validation failed: no occurrences of '{find_text}' found")
                    return False

                logger.debug(f"Pattern validation successful: {len(validation_changes)} occurrences found")
                return True

            except Exception as e:
                logger.debug(f"Pattern validation failed with exception: {e}")
                return False

        except Exception as e:
            logger.error(f"Pattern validation error: {e}")
            return False

    def _extract_single_pattern(self, amendment: Amendment, schedule_id: str) -> Optional[Dict[str, str]]:
        """
        Extract pattern from a single "each place" amendment.

        Args:
            amendment: Amendment to process
            schedule_id: Schedule identifier

        Returns:
            Dictionary with find_text and replace_text, or None if extraction failed
        """
        aid = get_amendment_id(amendment)

        with bind(schedule_id=schedule_id, amendment_id=aid):
            event(
                logger,
                EVT.PATTERN_EXTRACTION_START,
                f"Extracting pattern for amendment {aid}",
            )

            # Get source element from amending bill
            source_element = self.xml_handler.find_element_by_eid(self._amending_bill, amendment.source_eid)

            if source_element is None:
                logger.error(f"Source element {amendment.source_eid} not found")
                return None

            # Convert element to string for prompt
            amendment_xml = self.xml_handler.element_to_string(source_element)

            try:
                # Call LLM to extract pattern
                response = self.llm_kernel.run_inference(
                    "ExtractEachPlacePattern",
                    schedule_id,
                    aid,
                    None,  # No candidate_eid for pattern extraction
                    amendment_xml=amendment_xml,
                )

                # Parse CSV response
                reader = csv.DictReader(StringIO(response.strip()))
                rows = list(reader)

                if not rows:
                    logger.error(f"No patterns extracted for amendment {aid}")
                    return None

                pattern = rows[0]

                # Validate required fields
                if "find_text" not in pattern or "replace_text" not in pattern:
                    logger.error(f"Missing required fields in pattern for amendment {aid}")
                    return None

                # For deletions, always use empty string regardless of what LLM returned
                if amendment.amendment_type == AmendmentType.DELETION:
                    pattern["replace_text"] = ""
                    logger.debug(f"Deletion amendment {aid}: forcing replace_text to empty string")

                return pattern

            except Exception as e:
                logger.error(f"Failed to extract pattern for amendment {aid}: {e}")
                return None