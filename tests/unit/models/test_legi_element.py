# test_models_legi_element.py
import unittest
from lxml import etree
from app.models.legi_element import LegiElement


class TestLegiElement(unittest.TestCase):
    def setUp(self):
        # Create a sample XML structure for testing
        self.root = etree.Element("root")
        self.child1 = etree.SubElement(self.root, "child1")
        self.child2 = etree.SubElement(self.root, "child2")
        self.child1.text = "Child 1 Text"
        self.child2.text = "Child 2 Text"
        self.child2.tail = "Child 2 Tail"
        self.grandchild1 = etree.SubElement(self.child1, "grandchild1")
        self.grandchild1.tail = "Grandchild 1 Tail"
        self.legi_root = LegiElement(self.root)
        self.legi_child1 = LegiElement(self.child1)
        self.legi_child2 = LegiElement(self.child2)
        self.legi_grandchild1 = LegiElement(self.grandchild1)
        self.child1.set("initial_attr", "initial_value")

    def test_get_children(self):
        children = self.legi_root.get_children()
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0].get_tag_name(), "child1")
        self.assertEqual(children[1].get_tag_name(), "child2")

    def test_get_parent(self):
        parent = self.legi_child1.get_parent()
        self.assertIsNotNone(parent)
        self.assertEqual(parent.get_tag_name(), "root")
        self.assertIsNone(self.legi_root.get_parent())  # Root should have no parent

    def test_get_ancestors(self):
        ancestors = self.legi_child1.get_ancestors()
        self.assertEqual(len(ancestors), 1)
        self.assertEqual(ancestors[0].get_tag_name(), "root")

    def test_get_descendant(self):
        descendant = self.legi_root.get_descendant(".//child1")
        self.assertIsNotNone(descendant)
        self.assertEqual(descendant.get_tag_name(), "child1")
        invalid_descendant = self.legi_root.get_descendant("nonexistent")
        self.assertIsNone(invalid_descendant)

    def test_get_descendants(self):
        descendants = self.legi_root.get_descendants(".//*")
        self.assertEqual(len(descendants), 3)
        self.assertEqual(descendants[0].get_tag_name(), "child1")
        self.assertEqual(descendants[1].get_tag_name(), "grandchild1")
        self.assertEqual(descendants[2].get_tag_name(), "child2")
        descendants = self.legi_child1.get_descendants(".//*")
        self.assertEqual(len(descendants), 1)

    def test_to_string(self):
        xml_string = self.legi_root.to_string()
        self.assertIn("<root>", xml_string)
        self.assertIn("<child1", xml_string)

    def test_get_tail(self):
        tail_text = self.legi_child2.get_tail()
        self.assertEqual(tail_text, "Child 2 Tail")

    def test_set_tail(self):
        self.legi_child2.set_tail("New Tail")
        self.assertEqual(self.child2.tail, "New Tail")

    def test_is_processing_instruction(self):
        mock_pi = etree.ProcessingInstruction("test", "instruction")
        pi_element = LegiElement(mock_pi)
        self.assertTrue(pi_element.is_processing_instruction())
        self.assertFalse(self.legi_child1.is_processing_instruction())

    def test_unwrap_element(self):
        self.legi_child1.unwrap_element()
        self.assertEqual(len(self.root), 2)  # child2 should remain and grandchild1 should be moved up
        self.assertEqual("Grandchild 1 TailChild 1 Text", self.root.text)  # Child1's text should be added to parent
        self.legi_child2.unwrap_element()
        self.assertEqual(len(self.root), 1)  # Only grandchild1 should remain
        self.assertEqual("Grandchild 1 TailChild 1 Text", self.root.text)  # Child2's tail should be added to parent

    def test_delete_element(self):
        self.legi_child2.delete_element()
        self.assertEqual(len(self.root), 1)  # Only child2 should remain
        self.assertEqual(self.root[0].tail, "Child 2 Tail")
        self.legi_child1.delete_element()
        self.assertEqual(len(self.root), 0)
        self.assertEqual(self.root.text, "Child 2 Tail")

    def test_get_attribute_existing_attribute(self):
        assert self.legi_child1.get_attribute("initial_attr") == "initial_value"

    def test_get_attribute_nonexistent_attribute(self):
        assert self.legi_child1.get_attribute("nonexistent_attr") == ""

    def test_set_attribute_new_attribute(self):
        self.legi_child1.set_attribute("new_attr", "new_value")
        assert self.child1.get("new_attr") == "new_value"

    def test_set_attribute_update_existing_attribute(self):
        self.legi_child1.set_attribute("initial_attr", "updated_value")
        assert self.child1.get("initial_attr") == "updated_value"

    def test_str(self):
        assert str(self.legi_child2) == "<child2>Child 2 Text</child2>Child 2 Tail"

    def test_repr(self):
        assert repr(self.legi_child2) == "<child2>Child 2 Text</child2>Child 2 Tail"

    def test_instantiate_class_without_element(self):
        with self.assertRaises(ValueError):
            LegiElement(None)
