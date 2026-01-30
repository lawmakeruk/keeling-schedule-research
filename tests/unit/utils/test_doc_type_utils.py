import os
import tempfile

import pytest
from app.utils.doc_type_utils import DocTypeUtils, Legislation, Jurisdiction
from lxml import etree


@pytest.fixture
def doc_type_utils():
    """Fixture to provide a DocTypeUtils instance."""
    return DocTypeUtils()


def test__get_legislation_type(doc_type_utils):
    legislation_type = doc_type_utils.get_legislation_type("ukpubb")
    assert legislation_type is Legislation.PRIMARY


def test__get_jurisdiction_type(doc_type_utils):
    jurisdiction_type = doc_type_utils.get_jurisdiction_type("ukpubb")
    assert jurisdiction_type is Jurisdiction.UK


def test__get_legislation_type__none(doc_type_utils):
    legislation_type = doc_type_utils.get_legislation_type("invalid")
    assert legislation_type is None


def test__get_jurisdiction_type__none(doc_type_utils):
    jurisdiction_type = doc_type_utils.get_jurisdiction_type("invalid")
    assert jurisdiction_type is None


def test__get_legislation_type_from_root(doc_type_utils):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        <bill name="nisi">
            <body>
                <hcontainer name="schedule">
                    <num>Schedule 1</num>
                    <heading>Amendments to Test Act</heading>
                    <content>
                        <p>The Test Act is amended by inserting...</p>
                    </content>
                </hcontainer>
            </body>
        </bill>
    </akomaNtoso>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        tmp.write(xml_content.encode("utf-8"))
        tmp_path = tmp.name

    try:
        tree = etree.parse(tmp_path)
        legislation_type = doc_type_utils.get_legislation_type_from_root(tree.getroot())
        assert legislation_type is Legislation.SECONDARY
    finally:
        os.remove(tmp_path)


def test__get_jurisdiction_type_from_root(doc_type_utils):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        <bill name="nisi">
            <body>
                <hcontainer name="schedule">
                    <num>Schedule 1</num>
                    <heading>Amendments to Test Act</heading>
                    <content>
                        <p>The Test Act is amended by inserting...</p>
                    </content>
                </hcontainer>
            </body>
        </bill>
    </akomaNtoso>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        tmp.write(xml_content.encode("utf-8"))
        tmp_path = tmp.name

    try:
        tree = etree.parse(tmp_path)
        legislation_type = doc_type_utils.get_jurisdiction_type_from_root(tree.getroot())
        assert legislation_type is Jurisdiction.NI
    finally:
        os.remove(tmp_path)


def test__get_legislation_type_from_root_with_act(doc_type_utils):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        <act name="ukpga">
            <body>
                <section>
                    <num>1</num>
                    <heading>Sample Section</heading>
                    <content>
                        <p>Sample content</p>
                    </content>
                </section>
            </body>
        </act>
    </akomaNtoso>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        tmp.write(xml_content.encode("utf-8"))
        tmp_path = tmp.name

    try:
        tree = etree.parse(tmp_path)
        legislation_type = doc_type_utils.get_legislation_type_from_root(tree.getroot())
        assert legislation_type is Legislation.PRIMARY
    finally:
        os.remove(tmp_path)


def test__get_jurisdiction_type_from_root_with_act(doc_type_utils):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        <act name="ukpga">
            <body>
                <section>
                    <num>1</num>
                    <heading>Sample Section</heading>
                    <content>
                        <p>Sample content</p>
                    </content>
                </section>
            </body>
        </act>
    </akomaNtoso>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        tmp.write(xml_content.encode("utf-8"))
        tmp_path = tmp.name

    try:
        tree = etree.parse(tmp_path)
        jurisdiction_type = doc_type_utils.get_jurisdiction_type_from_root(tree.getroot())
        assert jurisdiction_type is Jurisdiction.UK
    finally:
        os.remove(tmp_path)


def test__get_legislation_type_from_root_no_elements(doc_type_utils):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        <debate>
            <body>
                <debateSection>
                    <heading>Sample Debate</heading>
                    <speech>
                        <p>Sample speech</p>
                    </speech>
                </debateSection>
            </body>
        </debate>
    </akomaNtoso>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        tmp.write(xml_content.encode("utf-8"))
        tmp_path = tmp.name

    try:
        tree = etree.parse(tmp_path)
        legislation_type = doc_type_utils.get_legislation_type_from_root(tree.getroot())
        assert legislation_type is None
    finally:
        os.remove(tmp_path)


def test__get_jurisdiction_type_from_root_no_elements(doc_type_utils):
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        <debate>
            <body>
                <debateSection>
                    <heading>Sample Debate</heading>
                    <speech>
                        <p>Sample speech</p>
                    </speech>
                </debateSection>
            </body>
        </debate>
    </akomaNtoso>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        tmp.write(xml_content.encode("utf-8"))
        tmp_path = tmp.name

    try:
        tree = etree.parse(tmp_path)
        jurisdiction_type = doc_type_utils.get_jurisdiction_type_from_root(tree.getroot())
        assert jurisdiction_type is None
    finally:
        os.remove(tmp_path)
