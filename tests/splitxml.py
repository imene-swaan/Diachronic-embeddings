import unittest
from src.data.data_loader import split_xml
import xml.etree.ElementTree as ET 
import os
import shutil




class TestSplitXML(unittest.TestCase):

    def setUp(self):
        # Set up a temporary directory and create a test XML file
        self.test_dir = "temp_test_dir"
        os.mkdir(self.test_dir)

        self.test_xml_path = f"{self.test_dir}/test.xml"
        self.test_output_dir = f"{self.test_dir}/output"
        os.mkdir(self.test_output_dir)

        # Create a test XML file with a known number of children
        root = ET.Element("root")
        for i in range(2003):  # Adjust the range to create as many children as needed for the test
            child = ET.Element("child")
            child.text = str(i)
            root.append(child)
        tree = ET.ElementTree(root)
        tree.write(self.test_xml_path)

    def tearDown(self):
        # Clean up the directory after tests
        shutil.rmtree(self.test_dir)

    def test_split_xml(self):
        # Call the function with the test XML
        generated_files = split_xml(self.test_xml_path, self.test_output_dir, 500)

        # Assert the expected number of files were created
        self.assertEqual(len(generated_files), 5)  # We expect 2003/500 = 5 files

        # Assert that each file has the correct number of children, except the last one
        for file_path in generated_files[:-1]:
            tree = ET.parse(file_path)
            root = tree.getroot()
            self.assertEqual(len(root), 500)

        # Check the last file for the correct number of children (2003 % 500 = 3 children)
        tree = ET.parse(generated_files[-1])
        root = tree.getroot()
        self.assertEqual(len(root), 3)


if __name__ == '__main__':
    unittest.main()
