# app/services/utils.py
"""
Utility functions for the Keeling service.
Handles data transformations and common operations.
"""
import csv
import io
import logging
import re
from typing import List, Dict, Any, Tuple, Union, Optional
from lxml import etree
from ..models.amendments import Amendment

# Sort value for non-numeric eId components
NON_NUMERIC_SORT_VALUE = 999999

logger = logging.getLogger(__name__)


# ==================== Public Interface Functions ====================


def csv_to_amendment_dict(
    csv_string: str, target_act: Optional[etree.ElementTree] = None, xml_handler: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Convert CSV string response from LLM to a list of amendment dictionaries.

    Expected CSV columns:
    - source_eid: Source element ID
    - source: Human-readable source reference
    - type_of_amendment: INSERTION, DELETION, or SUBSTITUTION
    - whole_provision: Boolean indicating if entire provision affected
    - location: BEFORE, AFTER, or REPLACE
    - affected_provision: Target provision eId

    Args:
        csv_string: CSV-formatted string from LLM
        target_act: Optional target act XML tree for range expansion
        xml_handler: Optional XML handler for element lookups

    Returns:
        List of amendment data dictionaries

    Raises:
        ValueError: If CSV is invalid or missing required columns
    """
    required_cols = {"source_eid", "source", "type_of_amendment", "whole_provision", "location", "affected_provision"}

    # Find and extract header
    header_idx = _find_csv_header_index(csv_string, required_cols)
    csv_data = _extract_csv_from_header(csv_string, header_idx)

    # Parse and validate CSV
    reader = _create_normalised_csv_reader(csv_data)
    _validate_csv_columns(reader.fieldnames, required_cols)

    # Process all rows
    amendments = _parse_amendment_rows(reader, required_cols)

    # ABLATION: Compact CSV expansion disabled
    # # Expand any amendments with semicolon-separated or range-based affected provisions
    # expanded_amendments = []
    # for amendment in amendments:
    #     expanded = _expand_amendment_if_needed(amendment, target_act, xml_handler)
    #     expanded_amendments.extend(expanded)
    # 
    # if not expanded_amendments:
    #     raise ValueError("No valid amendments found in CSV")
    # 
    # return expanded_amendments

    # Return unexpanded amendments directly
    if not amendments:
        raise ValueError("No valid amendments found in CSV")
        
    return amendments


def sort_amendments_by_affected_provision(amendments: List[Amendment]) -> List[Amendment]:
    """
    Sort amendments using Akoma Ntoso-aware structure by their affected_provision eId.

    This ensures amendments are applied in document order, respecting the hierarchical
    structure of legislative documents (e.g., sections before schedules, numeric ordering
    within sections).

    Args:
        amendments: List of Amendment objects

    Returns:
        Sorted list of amendments in document order
    """
    return sorted(amendments, key=lambda a: _eid_sort_key(a.affected_provision or ""))


def sort_amendments_by_source_eid(amendments: List[Amendment]) -> List[Amendment]:
    """
    Sort amendments using Akoma Ntoso-aware structure by their source eId.

    Args:
        amendments: List of Amendment objects

    Returns:
        Sorted list of amendments in document order
    """
    return sorted(amendments, key=lambda a: _eid_sort_key(a.source_eid or ""))


def group_amendments_by_target(amendments: List[Amendment]) -> Dict[str, List[Amendment]]:
    """
    Group amendments by their target provision.

    This is used to process amendments that affect the same provision
    sequentially, ensuring each subsequent amendment sees the results
    of previous amendments.

    Args:
        amendments: List of amendments to group

    Returns:
        Dictionary mapping target provision eId to list of amendments
    """
    groups = {}
    for amendment in amendments:
        target = amendment.affected_provision
        if target not in groups:
            groups[target] = []
        groups[target].append(amendment)

    # Sort amendments within each group to maintain order
    for target in groups:
        groups[target] = sorted(groups[target], key=lambda a: _eid_sort_key(a.affected_provision or ""))

    return groups


def get_amendment_id(amendment: Amendment) -> Optional[str]:
    """
    Safely get the amendment ID from an Amendment object.

    Args:
        amendment: Amendment object

    Returns:
        Amendment ID string or None if not present
    """
    return getattr(amendment, "amendment_id", None)


# ==================== eId Formatting Functions ====================


def eid_to_source(eid: str) -> str:
    """
    Convert an Akoma Ntoso element ID (eId) to a human-readable legal citation.

    TODO:
        - test more edge cases and refine before looking to use this function for the LLM response
        - test with SIs as the conversion from eId to citation is different for secondary legislation

    Args:
        eid: Element ID string from XML (e.g., 'sec_40__subsec_2')

    Returns:
        Formatted legal citation string. Returns the original eId if formatting fails.
    """
    if not eid:
        return eid

    try:
        # Check for malformed patterns
        if eid.endswith("__") or eid.startswith("__") or "___" in eid:
            return eid  # Return malformed eIds unchanged

        # Split into component parts
        parts = eid.split("__")

        # Check for special cases first
        # 1. Definition references (e.g., sec_93__def_tenant)
        if len(parts) >= 2 and parts[-1].startswith("def_"):
            # Extract the definition term
            def_term = parts[-1].replace("def_", "").replace("_", " ")
            # Format the containing provision
            containing_provision = "__".join(parts[:-1])
            formatted_container = _format_provision_parts(containing_provision.split("__"))
            return f'definition of "{def_term}" in {formatted_container}'

        # 2. Heading references (e.g., sec_4__hdg, pt_2__hdg)
        if parts[-1] == "hdg":
            # Check for crossheading pattern (e.g., pt_1__chp_2__xhdg_28__hdg)
            if len(parts) >= 2 and parts[-2].startswith("xhdg_"):
                # Extract crossheading number and format the containing provision
                xhdg_num = parts[-2].replace("xhdg_", "")
                if len(parts) > 2:
                    containing_provision = "__".join(parts[:-2])
                    formatted_container = _format_provision_parts(containing_provision.split("__"))
                    return f"{formatted_container} crossheading {xhdg_num}"
                else:
                    return f"crossheading {xhdg_num}"

            # Regular heading
            containing_provision = "__".join(parts[:-1])
            formatted_container = _format_provision_parts(containing_provision.split("__"))
            return f"heading of {formatted_container}"

        # 3. Regular provisions
        return _format_provision_parts(parts)

    except Exception as e:
        logger.warning(f"Failed to format eId '{eid}': {e}")
        return eid  # Return original eId as fallback


def _format_provision_parts(parts: list) -> str:
    """
    Format a list of provision parts into a legal citation.

    Args:
        parts: List of eId components (e.g., ['sec', '40', 'subsec', '2'])

    Returns:
        Formatted citation string
    """
    if not parts:
        return ""

    # Process based on the main provision type (first part)
    main_part = parts[0]

    # Extract type and number/identifier
    # Match either "type_number" or just "type"
    type_match = re.match(r"([a-zA-Z]+)(?:_(.*))?$", main_part)
    if not type_match:
        return "__".join(parts)  # Fallback to original if no match

    prov_type = type_match.group(1)
    prov_num = type_match.group(2) or ""  # Handle None from optional group
    prov_type_lower = prov_type.lower()

    # Determine if this is a schedule/paragraph structure
    is_schedule_context = prov_type_lower in ["sched", "schedule"]

    # For sections, regulations, articles, rules - format with parenthetical subsections
    if prov_type_lower in ["sec", "section"]:
        return _format_section_style(parts, "s.", prov_num)
    elif prov_type_lower in ["reg", "regulation"]:
        return _format_section_style(parts, "reg.", prov_num)
    elif prov_type_lower in ["art", "article"]:
        return _format_section_style(parts, "art.", prov_num)
    elif prov_type_lower in ["rule"]:
        return _format_section_style(parts, "rule ", prov_num)

    # For schedules - special handling
    elif is_schedule_context:
        return _format_schedule_style(parts, prov_num)

    # For parts and chapters
    elif prov_type_lower in ["pt", "part"]:
        result = f"pt.{prov_num}" if prov_num else "pt."
        if len(parts) > 1:
            # Add remaining parts (e.g., Chapter, section)
            remaining = _format_provision_parts(parts[1:])
            result = f"{result} {remaining}"
        return result

    elif prov_type_lower in ["chp", "chapter"]:
        result = f"ch.{prov_num}" if prov_num else "ch."
        if len(parts) > 1:
            remaining = _format_provision_parts(parts[1:])
            result = f"{result} {remaining}"
        return result

    # For paragraphs at the top level
    elif prov_type_lower in ["para", "paragraph"]:
        return _format_paragraph_standalone(parts)

    # Unknown type - return as-is
    else:
        return "__".join(parts)


def _format_section_style(parts: list, prefix: str, main_num: str) -> str:
    """
    Format provisions that follow section-style citation (sections, regulations, articles, rules).

    These use parenthetical notation: s.40(2)(a)(i)

    Args:
        parts: List of eId components (e.g., ['sec_40', 'subsec_2', 'para_a'])
        prefix: Citation prefix (e.g., 's.', 'reg.', 'art.', 'rule')
        main_num: Main provision number (e.g., '40', '12A')

    Returns:
        Formatted citation string (e.g., 's.40(2)(a)')
    """
    # Start with the main provision
    result = f"{prefix}{main_num}" if main_num else prefix.rstrip(".")

    # Collect subsections, paragraphs, subparagraphs in parentheses
    paren_parts = []

    for i in range(1, len(parts)):
        part = parts[i]

        if part.startswith("subsec_"):
            paren_parts.append(part.replace("subsec_", ""))
        elif part.startswith("para_"):
            paren_parts.append(part.replace("para_", ""))
        elif part.startswith("subpara_"):
            paren_parts.append(part.replace("subpara_", ""))
        elif part.startswith("subsubpara_"):
            paren_parts.append(part.replace("subsubpara_", ""))
        elif part.startswith("level_"):
            # Generic level element
            paren_parts.append(part.replace("level_", ""))

    # Add parenthetical parts
    if paren_parts:
        result += "(" + ")(".join(paren_parts) + ")"

    return result


def _format_schedule_style(parts: list, sched_num: str) -> str:
    """
    Format schedule provisions which have a different citation style.

    Args:
        parts: List of eId components starting with schedule (e.g., ['sched_3', 'para_5', 'subpara_3'])
        sched_num: Schedule number (e.g., '3', '1A')

    Returns:
        Formatted schedule citation (e.g., 'Sch.1 para.1(3)')
    """
    # Start with the schedule reference
    schedule_ref = f"Sch.{sched_num}" if sched_num else "Sch."

    # Look for paragraph components
    para_parts = []
    para_num = None
    has_part = False
    part_ref = ""

    for i in range(1, len(parts)):
        part = parts[i]

        if part.startswith("para_") and para_num is None:  # Only first para_ is the main paragraph
            para_num = part.replace("para_", "")
        elif part.startswith("subpara_"):
            para_parts.append(part.replace("subpara_", ""))
        elif part.startswith("para_") and para_num is not None:  # Subsequent para_ are sub-elements
            para_parts.append(part.replace("para_", ""))
        elif part.startswith("pt_"):
            # Part within schedule
            pt_num = part.replace("pt_", "")
            has_part = True
            part_ref = f"pt.{pt_num}"

    # Build the final reference
    if has_part:
        schedule_ref = f"{schedule_ref} {part_ref}"

    # If we have paragraph information, format it
    if para_num:
        para_ref = f"para.{para_num}"
        if para_parts:
            para_ref += "(" + ")(".join(para_parts) + ")"
        return f"{schedule_ref} {para_ref}"

    return schedule_ref


def _format_paragraph_standalone(parts: list) -> str:
    """
    Format standalone paragraph references (when para is the main element).

    Args:
        parts: List of eId components starting with para (e.g., ['para_5', 'subpara_a'])

    Returns:
        Formatted paragraph citation (e.g., 'para.5(a)')
    """
    # Extract paragraph number
    para_match = re.match(r"para_(.+)", parts[0])
    if not para_match:
        return "__".join(parts)

    para_num = para_match.group(1)
    result = f"para.{para_num}"

    # Add any subparagraphs
    sub_parts = []
    for i in range(1, len(parts)):
        part = parts[i]
        if part.startswith("subpara_"):
            sub_parts.append(part.replace("subpara_", ""))

    if sub_parts:
        result += "(" + ")(".join(sub_parts) + ")"

    return result


# ==================== CSV Processing Helper Functions ====================


def _find_csv_header_index(csv_string: str, required_cols: set) -> int:
    """
    Find the index of the header line in CSV string.

    Args:
        csv_string: Raw CSV string
        required_cols: Set of required column names

    Returns:
        Index of header line
    """
    lines = csv_string.strip().split("\n")

    for i, line in enumerate(lines):
        # Check if line contains required column names (fuzzy match)
        line_lower = line.lower().replace("_", "").replace(" ", "")
        if any(col.replace("_", "") in line_lower for col in required_cols):
            return i

    return 0  # Default to first line if no match found


def _extract_csv_from_header(csv_string: str, header_idx: int) -> str:
    """
    Extract CSV data starting from the header line.

    Args:
        csv_string: Raw CSV string
        header_idx: Index of header line

    Returns:
        CSV string starting from header
    """
    lines = csv_string.strip().split("\n")
    return "\n".join(lines[header_idx:])


def _create_normalised_csv_reader(csv_data: str) -> csv.DictReader:
    """
    Create a CSV reader with normalised field names.

    Args:
        csv_data: CSV string data

    Returns:
        DictReader with normalised field names

    Raises:
        ValueError: If CSV has no headers
    """
    reader = csv.DictReader(io.StringIO(csv_data))

    if not reader.fieldnames:
        raise ValueError("CSV data has no headers")

    # Normalise field names
    normalised_fields = []
    for field in reader.fieldnames:
        normalised = field.strip().lower().replace(" ", "_").replace("-", "_")
        normalised_fields.append(normalised)
    reader.fieldnames = normalised_fields

    return reader


def _validate_csv_columns(fieldnames: List[str], required_cols: set) -> None:
    """
    Validate that all required columns are present.

    Args:
        fieldnames: List of field names from CSV
        required_cols: Set of required column names

    Raises:
        ValueError: If required columns are missing
    """
    if not required_cols.issubset(set(fieldnames)):
        missing = required_cols - set(fieldnames)
        raise ValueError(f"CSV missing required columns: {missing}")


def _parse_amendment_rows(reader: csv.DictReader, required_cols: set) -> List[Dict[str, Any]]:
    """
    Parse all rows from CSV reader into amendment dictionaries.

    Applies post-processing to fix common LLM errors in eId generation.

    Args:
        reader: CSV DictReader
        required_cols: Set of required column names

    Returns:
        List of validated and fixed amendment dictionaries
    """
    amendments = []

    for row_num, row in enumerate(reader, start=1):
        try:
            cleaned_row = _process_amendment_row(row, required_cols, row_num)
            # Apply post-processing fixes
            cleaned_row = _post_process_amendment(cleaned_row)
            amendments.append(cleaned_row)
        except Exception as e:
            logger.error(f"Error parsing CSV row {row_num}: {e}")
            logger.debug(f"Problematic row data: {row}")
            # Continue processing other rows

    return amendments


def _process_amendment_row(row: Dict[str, str], required_cols: set, row_num: int) -> Dict[str, Any]:
    """
    Process a single CSV row into a validated amendment dictionary.

    Args:
        row: Raw row data from CSV
        required_cols: Set of required column names
        row_num: Row number for error reporting

    Returns:
        Cleaned and validated amendment dictionary

    Raises:
        ValueError: If row data is invalid
    """
    # Clean values
    cleaned_row = {}
    for key, value in row.items():
        cleaned_row[key] = value.strip() if value else ""

    # Validate required fields
    for col in required_cols:
        if not cleaned_row.get(col):
            raise ValueError(f"Row {row_num}: Missing required field '{col}'")

    # Normalise eIds to match XML normalisation
    cleaned_row["affected_provision"] = _normalise_eid_string(cleaned_row["affected_provision"])
    cleaned_row["source_eid"] = _normalise_eid_string(cleaned_row["source_eid"])

    # Convert other types
    cleaned_row["whole_provision"] = cleaned_row["whole_provision"].lower() == "true"
    cleaned_row["location"] = cleaned_row["location"].upper()
    cleaned_row["type_of_amendment"] = cleaned_row["type_of_amendment"].upper()

    # Validate enums
    _validate_amendment_type(cleaned_row["type_of_amendment"], row_num)
    _validate_location(cleaned_row["location"], row_num)

    return cleaned_row


def _normalise_eid_string(eid: str) -> str:
    """
    Normalise an eId string to match XML normalisation.

    Converts structural keywords to lowercase while preserving
    the case of alphanumeric identifiers.

    Args:
        eid: Element ID string to normalise

    Returns:
        Normalised eId string
    """
    # Keywords ordered longest-first to prevent partial matches
    keywords = "subsubpara|subpara|subsec|sched|apndx|anx|chp|para|rule|sec|pt|st|reg|art"

    # Use lookbehind/lookahead to match keywords but not parts of identifiers
    pattern = re.compile(rf"(?<![A-Za-z0-9])({keywords})(?![A-Za-z])", re.IGNORECASE)

    # Single pass: lowercase only the recognised keywords
    return pattern.sub(lambda match: match.group(1).lower(), eid)


def _validate_amendment_type(amendment_type: str, row_num: int) -> None:
    """
    Validate amendment type value.

    Args:
        amendment_type: Type value to validate
        row_num: Row number for error reporting

    Raises:
        ValueError: If amendment type is invalid
    """
    valid_types = {"INSERTION", "DELETION", "SUBSTITUTION"}
    if amendment_type not in valid_types:
        raise ValueError(f"Row {row_num}: Invalid amendment type '{amendment_type}'. " f"Must be one of: {valid_types}")


def _validate_location(location: str, row_num: int) -> None:
    """
    Validate location value.

    Args:
        location: Location value to validate
        row_num: Row number for error reporting

    Raises:
        ValueError: If location is invalid
    """
    valid_locations = {"BEFORE", "AFTER", "REPLACE", "EACH_PLACE"}
    if location not in valid_locations:
        raise ValueError(f"Row {row_num}: Invalid location '{location}'. " f"Must be one of: {valid_locations}")


def _expand_amendment_if_needed(
    amendment: Dict[str, Any], target_act: Optional[etree.ElementTree] = None, xml_handler: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Expand an amendment if it has multiple affected provisions.

    Recursively handles combinations of provision lists and ranges.

    Handles two types of multi-provision amendments:
    1. Semicolon-separated lists (e.g., "sec_1__para_a;sec_1__para_b")
       - Expanded immediately into individual amendments
       - Each item is recursively checked for further expansion
    2. Range-based provisions (e.g., "sec_5__subsec_3-sec_5__subsec_6")
       - Expanded by finding all provisions between start and end in document order

    Args:
        amendment: Single amendment dictionary from CSV parsing
        target_act: Optional target act XML tree for range expansion
        xml_handler: Optional XML handler for element lookups

    Returns:
        List of amendments - either the original (if no expansion needed)
        or multiple amendments (one per affected provision)
    """
    affected_provision = amendment.get("affected_provision", "")

    logger.debug(f"Checking if expansion needed for: {affected_provision}")

    # Handle semicolon-separated lists (e.g., from "in paragraphs (a) and (b)")
    if ";" in affected_provision:
        provisions = affected_provision.split(";")
        expanded = []

        for provision in provisions:
            # Create a copy of the amendment for each provision
            new_amendment = amendment.copy()
            new_amendment["affected_provision"] = provision.strip()

            # Recursively check if this provision also needs expansion (e.g., if it's a range)
            sub_expanded = _expand_amendment_if_needed(new_amendment, target_act, xml_handler)
            expanded.extend(sub_expanded)

        logger.debug(f"Expanded semicolon-separated amendment into {len(expanded)} amendments: {affected_provision}")
        return expanded

    # Handle range-based provisions (e.g., "sec_5__subsec_3-sec_5__subsec_6")
    if "-" in affected_provision and target_act is not None and xml_handler is not None:
        try:
            logger.debug(f"Attempting range expansion for: {affected_provision}")

            expanded = _expand_range_amendment(amendment, affected_provision, target_act, xml_handler)
            if expanded:
                logger.debug(f"Expanded range-based amendment into {len(expanded)} amendments: {affected_provision}")
                return expanded
        except Exception as e:
            logger.warning(f"Failed to expand range {affected_provision}: {e}. Keeping as-is.")
            # Fall through to return unexpanded

    # No expansion needed - return single amendment
    logger.debug(f"No expansion needed for: {affected_provision}")
    return [amendment]


def _expand_range_amendment(
    amendment: Dict[str, Any], range_str: str, target_act: etree.ElementTree, xml_handler: Any
) -> List[Dict[str, Any]]:
    """
    Expand a range-based provision (e.g., "sec_5-sec_8") into individual amendments.
    """
    # Parse the range
    parts = range_str.split("-", 1)
    if len(parts) != 2:
        logger.warning(f"Invalid range format: {range_str}")
        return []

    start_eid, end_eid = parts[0].strip(), parts[1].strip()

    logger.debug(f"Expanding range: {range_str}")
    logger.debug(f"Start eId: {start_eid}, End eId: {end_eid}")

    # Use xml_handler method to find provisions in range
    provisions_in_range = xml_handler.find_provisions_in_range(start_eid, end_eid, target_act)

    logger.debug(f"Found provisions: {provisions_in_range}")

    if not provisions_in_range:
        logger.warning(f"No provisions found in range {range_str}")
        return []

    # Create individual amendments
    expanded = []
    for provision_eid in provisions_in_range:
        new_amendment = amendment.copy()
        new_amendment["affected_provision"] = provision_eid
        expanded.append(new_amendment)

    logger.info(f"Expanded range {range_str} into {len(expanded)} provisions: {provisions_in_range}")

    return expanded


def _fix_provision_nesting_in_eid(eid: str) -> str:
    """
    Fix common nesting errors in eIds based on provision type.

    For all provisions (schedules, sections, regulations):
    - Letters should be 'para': reg_2__para_1__para_a (not subpara_a)
    - Roman numerals should be 'subpara': reg_2__para_1__para_a__subpara_i

    Args:
        eid: Element ID to fix

    Returns:
        Corrected eId
    """
    original = eid

    # Fix schedule nesting (subpara -> para for letters)
    if eid.startswith("sched"):
        # Pattern: sched_X__para_Y__subpara_[letter]
        pattern = r"(sched(?:_\d+)?__para_\d+[A-Za-z]?)__subpara_([a-z])(?=__|$)"
        eid = re.sub(pattern, r"\1__para_\2", eid)
        if eid != original:
            logger.debug(f"Fixed schedule nesting: {original} -> {eid}")

    # Fix other provision nesting (subpara -> para for letters)
    elif any(eid.startswith(prefix) for prefix in ["sec_", "reg_", "art_", "rule_"]):
        # Pattern: [type]_X__para_Y__subpara_[letter] (should be para)
        pattern = r"((?:sec|reg|art|rule)_\d+[A-Za-z]?__para_\d+[A-Za-z]?)__subpara_([a-z])(?=__|$)"
        eid = re.sub(pattern, r"\1__para_\2", eid)
        if eid != original:
            logger.debug(f"Fixed provision nesting: {original} -> {eid}")

    return eid


def _post_process_amendment(amendment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply post-processing fixes to amendment data.

    Fixes common LLM errors in eId generation for affected provisions.

    Args:
        amendment: Amendment dictionary to fix

    Returns:
        Fixed amendment dictionary
    """
    # Fix nesting errors in affected_provision
    if "affected_provision" in amendment:
        amendment["affected_provision"] = _fix_provision_nesting_in_eid(amendment["affected_provision"])

    return amendment


# ==================== Sorting Helper Functions ====================


def _eid_sort_key(eid: str) -> List[Tuple[int, str, Union[int, str]]]:
    """
    Convert an eId into a structured sort key respecting Akoma Ntoso document order.

    For example, ensures that 'sec_59a' comes before 'sec_59b',
    and all sections come before schedules.

    Args:
        eid: Element ID (e.g. 'sec_59a', 'sched_2__para_5')

    Returns:
        Structured list of tuples to use as sort key
    """
    TYPE_PRIORITY = {
        "preamble": 0,
        "part": 1,
        "chapter": 2,
        "sec": 3,
        "section": 3,
        "article": 3,
        "sched": 4,
        "schedule": 4,
    }

    parts = eid.lower().split("__")
    key = []

    for part in parts:
        sort_tuple = _parse_eid_part(part, TYPE_PRIORITY)
        key.append(sort_tuple)

    return key


def _parse_eid_part(part: str, type_priority: Dict[str, int]) -> Tuple:
    """
    Parse a single eId part into a sort tuple.

    Always returns a consistent 4-element tuple:
    (priority, label, numeric_value, string_suffix)

    Args:
        part: Single part of an eId (e.g., 'sec_59a')
        type_priority: Priority mapping for element types

    Returns:
        Sort tuple for this part
    """
    # Try to match pattern like 'sec_59a' or 'para_a'
    match = re.match(r"([a-zA-Z]+)_(\d+[a-z]*|[a-z]+)", part)
    if match:
        label, value = match.groups()
        priority = type_priority.get(label, 99)

        # Try to extract numeric value with optional suffix
        value_match = re.match(r"(\d+)([a-z]*)", value)
        if value_match:
            num, suffix = value_match.groups()
            return (priority, label, int(num), suffix or "")
        else:
            # Non-numeric value (e.g., 'para_a')
            # Use constant for numeric comparison and the value as suffix
            return (priority, label, NON_NUMERIC_SORT_VALUE, value)
    else:
        # Check if this part is a known type without a value (like "preamble")
        if part in type_priority:
            return (type_priority[part], part, 0, "")
        else:
            # Unknown pattern - use high priority and the part as suffix
            return (99, part, NON_NUMERIC_SORT_VALUE, part)