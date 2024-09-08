import xml.etree.ElementTree as ET

def show_first_3_annotations(xml_file):
    """
    Displays the first 3 annotations from the specified XML file.
    
    Parameters:
    - xml_file (str): Path to the XML file.
    """
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    print("Annotations:")
    for i, image in enumerate(root.findall('image')):
        if i >= 3:
            break
        
        print(f"Image {i+1}:")
        print(f"  ID: {image.get('id')}")
        print(f"  Name: {image.get('name')}")
        
        for box in image.findall('box'):
            print(f"  Box Label: {box.get('label')}")
            print(f"  Coordinates: ({box.get('xtl')}, {box.get('ytl')}) to ({box.get('xbr')}, {box.get('ybr')})")
            print(f"  Color: {box.find('attribute').text if box.find('attribute') is not None else 'N/A'}")
            break  # Show only the first box for each image
        
        print()  # Newline for readability