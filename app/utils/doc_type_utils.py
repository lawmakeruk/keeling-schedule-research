# app/utils/doc_type_utils.py
"""
Service that provides utility methods for document types.

Handles the classification of legislative documents by jurisdiction and type
based on their document type codes used in UK legislation XML.
"""
from enum import Enum
from typing import Dict, Optional, Any
from lxml import etree


# ==================== Enumerations ====================


class Jurisdiction(Enum):
    """
    Represents the jurisdiction of a legislative document.

    Values:
        UK: United Kingdom
        SP: Scottish Parliament
        NI: Northern Ireland
        WP: Welsh Parliament (Senedd Cymru)
        EU: European Union
    """

    UK = "UK"
    SP = "SP"
    NI = "NI"
    WP = "WP"
    EU = "EU"


class Legislation(Enum):
    """
    Represents the type/hierarchy of legislation.

    Values:
        PRIMARY: Primary legislation (Acts)
        SECONDARY: Secondary/subordinate legislation (Statutory Instruments)
        EU: European Union legislation (special category)
    """

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    EU = "EU"


# ==================== Main Service Class ====================


class DocTypeUtils:
    """
    Utility service for determining jurisdiction and legislation type from document type codes.

    This class maintains a mapping of document type codes (e.g., 'ukpga', 'ssi') to their
    corresponding jurisdiction and legislation type, enabling classification of legislative
    documents based on their type attributes in XML.
    """

    def __init__(self):
        """Initialise the service with document type mappings and namespace definitions."""
        self.namespaces = {
            "akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0",
            "ukl": "https://www.legislation.gov.uk/namespaces/UK-AKN",
        }

        # Initialise the document type mapping data
        self.data = self._initialise_doc_type_mappings()

    # ==================== Public Interface Methods ====================

    def get_jurisdiction_type_from_root(self, root: etree.Element) -> Optional[Jurisdiction]:
        """
        Extract jurisdiction type from an XML document root element.

        Looks for bill or act elements and extracts the 'name' attribute
        to determine jurisdiction.

        Args:
            root: Root element of the XML document

        Returns:
            Jurisdiction enum value if found, None otherwise
        """
        doc_type = self._extract_doc_type_from_root(root)
        if doc_type:
            return self.get_jurisdiction_type(doc_type)
        return None

    def get_legislation_type_from_root(self, root: etree.Element) -> Optional[Legislation]:
        """
        Extract legislation type from an XML document root element.

        Looks for bill or act elements and extracts the 'name' attribute
        to determine legislation type.

        Args:
            root: Root element of the XML document

        Returns:
            Legislation enum value if found, None otherwise
        """
        doc_type = self._extract_doc_type_from_root(root)
        if doc_type:
            return self.get_legislation_type(doc_type)
        return None

    def get_jurisdiction_type(self, doc_type: str) -> Optional[Jurisdiction]:
        """
        Get jurisdiction type for a given document type code.

        Args:
            doc_type: Document type code (e.g., 'ukpga', 'ssi')

        Returns:
            Jurisdiction enum value if found, None otherwise
        """
        entry = self.data.get(doc_type)
        if entry is not None:
            return entry["jurisdiction"]
        return None

    def get_legislation_type(self, doc_type: str) -> Optional[Legislation]:
        """
        Get legislation type for a given document type code.

        Args:
            doc_type: Document type code (e.g., 'ukpga', 'ssi')

        Returns:
            Legislation enum value if found, None otherwise
        """
        entry = self.data.get(doc_type)
        if entry is not None:
            return entry["legislation"]
        return None

    # ==================== Private Helper Methods ====================

    def _extract_doc_type_from_root(self, root: etree.Element) -> Optional[str]:
        """
        Extract document type from root element.

        Checks for bill element first, then act element, and extracts
        the 'name' attribute which contains the document type code.

        Args:
            root: Root element of the XML document

        Returns:
            Document type string if found, None otherwise
        """
        # Try bill element first
        doc_type_element = root.find("./akn:bill", self.namespaces)
        if doc_type_element is None:
            # Fall back to act element
            doc_type_element = root.find("./akn:act", self.namespaces)

        if doc_type_element is not None:
            return doc_type_element.get("name")
        return None

    def _initialise_doc_type_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialise the complete mapping of document types to jurisdictions and legislation types.

        Returns:
            Dictionary mapping document type codes to their metadata
        """
        return {
            # ========== UK Document Types ==========
            # UK Bills
            "ukpubb": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "ukprib": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "ukhybb": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "ukfinres": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            # UK Primary Legislation
            "ukpga": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "ukla": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "ukppa": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "ukcm": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "ukdcm": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            # Historical UK Acts
            "apgb": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "gbla": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "gbppa": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            "aep": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.PRIMARY},
            # UK Secondary Legislation
            "ukci": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.SECONDARY},
            "uksi": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.SECONDARY},
            "ukdsi": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.SECONDARY},
            "ukmo": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.SECONDARY},
            "uksro": {"jurisdiction": Jurisdiction.UK, "legislation": Legislation.SECONDARY},
            # ========== Scottish Document Types ==========
            # Scottish Bills
            "sppubb": {"jurisdiction": Jurisdiction.SP, "legislation": Legislation.PRIMARY},
            "spprib": {"jurisdiction": Jurisdiction.SP, "legislation": Legislation.PRIMARY},
            "sphybb": {"jurisdiction": Jurisdiction.SP, "legislation": Legislation.PRIMARY},
            # Scottish Primary Legislation
            "asp": {"jurisdiction": Jurisdiction.SP, "legislation": Legislation.PRIMARY},
            "aosp": {"jurisdiction": Jurisdiction.SP, "legislation": Legislation.PRIMARY},
            # Scottish Secondary Legislation
            "ssi": {"jurisdiction": Jurisdiction.SP, "legislation": Legislation.SECONDARY},
            "sdsi": {"jurisdiction": Jurisdiction.SP, "legislation": Legislation.SECONDARY},
            # ========== Welsh Document Types ==========
            # Welsh Primary Legislation
            "asc": {"jurisdiction": Jurisdiction.WP, "legislation": Legislation.PRIMARY},
            "anaw": {"jurisdiction": Jurisdiction.WP, "legislation": Legislation.PRIMARY},
            "mwa": {"jurisdiction": Jurisdiction.WP, "legislation": Legislation.PRIMARY},
            # Welsh Secondary Legislation
            "wsi": {"jurisdiction": Jurisdiction.WP, "legislation": Legislation.SECONDARY},
            "wdsi": {"jurisdiction": Jurisdiction.WP, "legislation": Legislation.SECONDARY},
            # ========== Northern Irish Document Types ==========
            # Northern Irish Bills
            "nipubb": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.PRIMARY},
            "niprib": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.PRIMARY},
            "nihybb": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.PRIMARY},
            # Northern Irish Primary Legislation
            "nia": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.PRIMARY},
            "aip": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.PRIMARY},
            "apni": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.PRIMARY},
            "mnia": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.PRIMARY},
            # Northern Irish Secondary Legislation
            "nisi": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.SECONDARY},
            "nidsi": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.SECONDARY},
            "nisr": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.SECONDARY},
            "nidsr": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.SECONDARY},
            "nisro": {"jurisdiction": Jurisdiction.NI, "legislation": Legislation.SECONDARY},
            # ========== EU Document Types ==========
            "eudr": {"jurisdiction": Jurisdiction.EU, "legislation": Legislation.EU},
            "eut": {"jurisdiction": Jurisdiction.EU, "legislation": Legislation.EU},
            "eur": {"jurisdiction": Jurisdiction.EU, "legislation": Legislation.EU},
            "eudn": {"jurisdiction": Jurisdiction.EU, "legislation": Legislation.EU},
        }
