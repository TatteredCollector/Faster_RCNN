from lxml import etree


def parse_xml_to_dict(xml):
    """
    :param xml:
    Element对象，
    :return:xml转化为字典形式
    """
    if len(xml) == 0:  # 底层长度为0
        print({xml.tag: xml.text})
        return {xml.tag: xml.text}
    result = {}

    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != "object":
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                #print(child.tag)
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

if __name__ == '__main__':
    with open('./VOCdevkit/VOC2012/Annotations/2007_000032.xml') as f:
        some_xml_data=f.read()
    xml = etree.fromstring(some_xml_data)
    data = parse_xml_to_dict(xml)["annotation"]
    print(len(data['object']))