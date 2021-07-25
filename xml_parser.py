import xml.etree.ElementTree as ET

def xml_parser(inp_file,inp):
    arr = []
    arr_feature = []
    tree = ET.parse(inp_file)
    root = tree.getroot()
    cur = 0
    for str_arg in inp:
        if not root.find("param"+str(cur)):
            arr.append(str_arg)
            arr_feature.append("param"+str(cur))
        else:
            child = root.find("param"+str(cur))
            name = child.find('name').text
            type = child.find('type').text
            min_param = child.find('min').text
            max_param = child.find('max').text
            default_param = child.find('def').text
            vals_trans = []
            for val in child.findall('valtrans'):
                vals_trans.append(val.text)
            try:
                if "int" in type:
                    def_val = int(default_param)
                    val = int(str_arg)
                    min_val = int(min_param)
                    max_val = int(max_param)
                elif "bool" in type:
                    def_val = bool(int(default_param))
                    val = bool(int(str_arg))
                    min_val = bool(int(min_param))
                    max_val = bool(int(max_param))
                elif "float" in type:
                    def_val = float(default_param)
                    val = float(str_arg)
                    min_val = float(min_param)
                    max_val = float(max_param)
                else:
                    val = str_arg
                if min_param != "":
                    if val < min_val:
                        val = min_val
                if max_param != "":
                    if val > max_val:
                        val = max_val
            except ValueError:
                val = def_val
            poss_vals = len(vals_trans)
            if poss_vals > 0:
                mod = val % poss_vals
                val = vals_trans[mod]
        arr.append(val)
        arr_feature.append(name)
        cur += 1
    return arr, arr_feature
