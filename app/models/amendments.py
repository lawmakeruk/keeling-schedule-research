# app/models/amendments.py
"""
Models for representing legislative amendments and their types.

Provides enums for amendment locations and types, and a dataclass for
representing complete amendments with their metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

# ==================== Enumerations ====================


class AmendmentLocation(Enum):
    """
    Specifies where an amendment should be placed relative to existing content.

    These values correspond to standard legislative drafting positions:
    - BEFORE: Insert new content before the target provision
    - AFTER: Insert new content after the target provision
    - REPLACE: Replace the target provision entirely
    - EACH_PLACE: Apply the amendment to each occurrence within the target provision
    """

    BEFORE = "Before"
    AFTER = "After"
    REPLACE = "Replace"
    EACH_PLACE = "Each_Place"


class AmendmentType(Enum):
    """
    Defines the type of modification an amendment makes to legislation.

    These are the fundamental operations in legislative drafting:
    - INSERTION: Adding new content
    - DELETION: Removing existing content
    - SUBSTITUTION: Replacing existing content (deletion + insertion)
    """

    INSERTION = "insertion"
    DELETION = "deletion"
    SUBSTITUTION = "substitution"


# ==================== Data Models ====================


@dataclass
class Amendment:
    """
    Represents a single legislative amendment with all its metadata.

    This is the core data structure that flows through the entire Keeling
    Schedule generation pipeline, from identification through application.

    Attributes:
        source: Human-readable reference to the provision containing the amendment
        source_eid: XML element ID of the source provision in the amending bill
        affected_document: The title of the target act
        affected_provision: The eId of the provision being modified in the target act
        location: Where to apply the amendment relative to the target
        amendment_type: Type of modification (insertion, deletion, substitution)
        whole_provision: Whether the amendment affects an entire structural unit
                        (e.g., whole section) or just partial text within it
        amendment_id: Unique identifier for tracking through the system (optional)
    """

    source: str
    source_eid: str
    affected_document: str
    affected_provision: str
    location: AmendmentLocation
    amendment_type: AmendmentType
    whole_provision: bool
    dnum_list: List[str] = field(default_factory=list)
    amendment_id: Optional[str] = None

    # ==================== Factory Methods ====================

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Amendment":
        """
        Creates an Amendment instance from a dictionary representation.

        Used for parsing LLM outputs and API responses. Handles the conversion
        of string values to appropriate enum types.

        Args:
            data: Dictionary containing amendment data with keys:
                - source: Source reference string
                - source_eid: Source element ID
                - affected_document: Target document title
                - affected_provision: Target provision eId
                - location: Location string (must match AmendmentLocation values)
                - type_of_amendment: Type string (must match AmendmentType values)
                - whole_provision: Boolean or string representation
                - amendment_id: Optional unique identifier

        Returns:
            Amendment: New Amendment instance

        Raises:
            KeyError: If required fields are missing
            AttributeError: If location or type values don't match enum values
        """
        # Convert string values to enums
        location = getattr(AmendmentLocation, data["location"].upper().replace(" ", "_"))
        amendment_type = getattr(AmendmentType, data["type_of_amendment"].upper())

        # Extract amendment_id if present
        amendment_id = data.get("amendment_id")

        return cls(
            source=data["source"],
            source_eid=data["source_eid"],
            affected_document=data["affected_document"],
            affected_provision=data["affected_provision"],
            location=location,
            amendment_type=amendment_type,
            whole_provision=data["whole_provision"],
            amendment_id=amendment_id,
        )

    # ==================== Instance Methods ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Amendment to a dictionary representation.

        Useful for serialisation, logging, and API responses.

        Returns:
            Dictionary with all amendment attributes, with enums converted to strings
        """
        return {
            "source": self.source,
            "source_eid": self.source_eid,
            "affected_document": self.affected_document,
            "affected_provision": self.affected_provision,
            "location": self.location.value,
            "type_of_amendment": self.amendment_type.value,
            "whole_provision": self.whole_provision,
            "amendment_id": self.amendment_id,
        }

    def is_insertion(self) -> bool:
        """
        Determines if the amendment is an insertion amendment.

        Returns:
            bool: True if the amendment is both an insertion amendment, False otherwise.
        """
        return self.amendment_type == AmendmentType.INSERTION

    def is_deletion(self) -> bool:
        """
        Determines if the amendment is a deletion amendment.

        Returns:
            bool: True if the amendment is both a deletion amendment, False otherwise.
        """
        return self.amendment_type == AmendmentType.DELETION

    def is_substitution(self) -> bool:
        """
        Determines if the amendment is a substitution amendment.

        Returns:
            bool: True if the amendment is both a substitution amendment, False otherwise.
        """
        return self.amendment_type == AmendmentType.SUBSTITUTION

    def __str__(self) -> str:
        """
        Human-readable string representation of the amendment.

        Returns:
            String describing the amendment in natural language
        """
        provision_type = "whole provision" if self.whole_provision else "partial"
        return (
            f"{self.amendment_type.value.capitalize()} amendment: "
            f"{provision_type} {self.location.value.lower()} {self.affected_document} {self.affected_provision} "
            f"(from {self.source})"
        )
