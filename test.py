import xml.etree.ElementTree as ET

def load_data(xml_file_path, tag):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # List to hold the text content of the specified tag
    data_list = []
    
    # Use an XPath expression to find all occurrences of the specified tag
    for elem in root.findall('.//' + tag):
        if isinstance(elem.text, str):
            data_list.append(elem.text)
    return data_list

# Get user input for the XML file path and tag
xml_file_path = "input/xml/TheNewYorkTimes1980.xml"
tag = "fulltext"

# Load the data
data_list = load_data(xml_file_path, tag)[:10]

# Print the data
for data in data_list:
    print("----")
    print(data)
