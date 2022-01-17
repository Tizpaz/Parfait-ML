import xml.etree.ElementTree as ET

def xml_parser_domains(inp_file, num_args):
    arr_min = []
    arr_max = []
    arr_type = []
    arr_default = []
    tree = ET.parse("./subjects/" + inp_file)
    root = tree.getroot()
    cur = 0
    for str_arg in range(num_args):
        child = root.find("param"+str(cur))
        type = child.find('type').text
        min_param = child.find('min').text
        max_param = child.find('max').text
        default_param = child.find('def').text
        arr_min.append(min_param)
        arr_max.append(max_param)
        arr_type.append(type)
        arr_default.append(default_param)
        cur += 1
    return arr_min, arr_max, arr_type, arr_default
