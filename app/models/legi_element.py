from lxml import etree
from typing import List, Optional
import re


class LegiElement:
    """
    A wrapper class for lxml.etree.Element to provide
    convenient functionality to manipulate XML elements.
    """

    NAMESPACES = {
        "akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0",
        "ukl": "https://www.legislation.gov.uk/namespaces/UK-AKN",
    }

    def __init__(self, element: etree.Element):
        """
        Initialize a LegiNode instance from an lxml Element.

        Args:
            element: The lxml Element representing the legislative node.
        """
        if element is None:
            raise ValueError("Element cannot be None")
        self.element = element

    def get_tag_name(self) -> str:
        """
        Get the tag name of the element.

        Returns:
            The tag name of the element as a string.
        """
        return self.element.tag

    def delete_element(self) -> None:
        """
        Deletes the element. The element's tail text will be retained unless is it newline whitespace.
        """
        parent = self.element.getparent()
        if parent is not None:
            should_preserve_tail = self.element.tail and re.fullmatch(r" *\n *", self.element.tail) is None
            if should_preserve_tail:
                prev = self.element.getprevious()
                if prev is not None:
                    # Append tail to the previous sibling's tail
                    prev.tail = (prev.tail or "") + self.element.tail
                else:
                    # If no previous sibling, append tail to parent's text
                    parent.text = (parent.text or "") + self.element.tail
            # Delete the element
            parent.remove(self.element)

    def unwrap_element(self) -> None:
        """
        Unwraps the element, by replacing the element with its child elements and text content.
        """
        parent = self.element.getparent()
        index = parent.index(self.element)

        # Build a replacement list to maintain the order of the children
        replacement = []

        # Add the starting text of the element
        if self.element.text:
            replacement.append(self.element.text)

        # Add all children of the element and their tail text
        for child in self.element:
            replacement.append(child)
            if child.tail:
                replacement.append(child.tail)

        # Insert the replacement content into the parent in reverse order
        for item in reversed(replacement):
            if isinstance(item, str):  # Text
                if index == 0:
                    parent.text = (parent.text or "") + item
                else:
                    # Add the text as tail of the previous element or set it as parent.text
                    parent[index - 1].tail = (parent[index - 1].tail or "") + item
            else:  # Child element
                parent.insert(index, item)

        # Delete the element
        self.delete_element()

    # def replace_element(self, replacement: str) -> LegiElement:

    def get_parent(self) -> Optional["LegiElement"]:
        """
        Get the parent element of this element.

        Returns:
            A LegiNode instance representing the parent element, or None if this element is the root.
        """
        parent = self.element.getparent()
        if parent is not None:
            return LegiElement(parent)
        else:
            return None

    def get_ancestors(self) -> List["LegiElement"]:
        """
        Get all ancestors of this element in order, starting from the parent and going upwards.

        Returns:
            List of ancestor elements.
        """
        ancestors = []
        current = self.get_parent()
        while current is not None:
            ancestors.append(current)
            current = current.get_parent()
        return ancestors

    def get_children(self) -> List["LegiElement"]:
        """
        Get the child elements of this element.

        Returns:
            A list of LegiNode instances representing the child elements.
        """
        return [LegiElement(child) for child in self.element.getchildren()]

    # def get_siblings(self) -> List['LegiElement']:

    # def get_next_sibling(self) -> 'LegiElement':

    # def get_previous_sibling(self) -> 'LegiElement':

    def get_descendant(self, xpath: str) -> Optional["LegiElement"]:
        """
        Find a descendant element using an XPath.

        Args:
            xpath: XPath string.

        Returns:
            The first matching descendant element or None, if not found.
        """
        descendant = self.element.find(xpath, namespaces=self.NAMESPACES)
        if descendant is not None:
            return LegiElement(descendant)
        else:
            return None

    def get_descendants(self, xpath: str) -> List["LegiElement"]:
        """
        Find all matching descendants using an XPath.

        Args:
            xpath: XPath string.

        Returns:
            List of matching descendant elements.
        """
        return [
            LegiElement(element)
            for element in self.element.xpath(xpath, namespaces=self.NAMESPACES)
            if element is not None
        ]

    def get_attribute(self, attribute: str) -> str:
        """
        Get the value of the specified attribute of the element.

        Args:
            attribute: The name of the attribute to retrieve.

        Returns:
            The value of the specified attribute as a string.
        """
        return self.element.get(attribute, "")

    def set_attribute(self, attribute: str, value: str) -> None:
        """
        Set the value of the specified attribute of the element.

        Args:
            attribute: The name of the attribute to set.
            value: The value to set for the attribute.

        Returns: None
        """
        self.element.set(attribute, value)

    def is_processing_instruction(self) -> bool:
        """
        Check if the current node is a processing instruction.

        Returns:
            True if the node is a processing instruction, False otherwise.
        """
        return self.element.tag is etree.PI

    def to_string(self) -> str:
        """
        Convert an element to its XML string representation.

        Returns:
            UTF-8 encoded XML string
        """
        return etree.tostring(self.element, encoding="utf-8").decode("utf-8")

    def get_tail(self) -> str:
        """
        Get the tail text of the current element.

        Returns:
            The tail text of the current element.
        """
        return self.element.tail or ""

    def set_tail(self, text: str) -> None:
        """
        Set the tail text of the current element.

        Args:
            text: The new tail text.
        """
        self.element.tail = text

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()
