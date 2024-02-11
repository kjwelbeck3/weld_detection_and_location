import os
import numpy as np
import point_cloud

def get_dataset_list(folder_path):
    try:
        ## Verifying the selected folder structure
        assert os.path.isdir(folder_path), f"The selected {folder_path} is not a directory"
        scans_path = folder_path+"/scans/"
        assert os.path.isdir(scans_path), "The selcted directory must contain a 'scans/' folder"
        labels_path = folder_path+"/labels/"
        assert os.path.isdir(labels_path), "The selcted directory must contain a 'labels/' folder"
        adjusted_labels_path = folder_path+"/adjusted-labels/"
        if not os.path.isdir(adjusted_labels_path):
            os.makedirs(adjusted_labels_path)

        ## List all the files in common between scans and labels subfolders
        scans = os.listdir(scans_path)
        # scans = [scan.replace(scans_path) for scan in scans]
        scans = ["".join(scan.split(".")[:-1]) for scan in scans if scan.endswith(('.csv', '.bin', '.ply'))]

        txts= os.listdir(labels_path)
        # txts= [txt.replace(labels_path) for txt in txts]
        txts= ["".join(txt.split(".")[:-1]) for txt in txts]

        pairs = [scan for scan in scans if scan in txts]
        missed_scans = [txt for txt in txts if txt not in scans]
        missed_labels = [scan for scan in scans if scan not in txts]
        meta = {
            "root_path": folder_path,
            "scans_path": scans_path,
            "scans_count": len(scans),
            "labels_path": labels_path,
            "labels_count": len(txts),
            "pair_count": len(pairs),
            "missed_scans": missed_scans,
            "missed_labels": missed_labels,
            "adjusted_labels_path": adjusted_labels_path
        }
        return len(scans), pairs, meta
    
    except AssertionError as e:
        return -1, e, None


def parse_label_string(string):
    ''' Expecting a comma separated string of "xloc, yloc, [ or ]"  '''

    contents = [string.strip() for string in string.split(",")]
    assert len(contents) == 4, f"Expected 4 comma-separated items but got {len(contents)} in \"{', '.join(contents)}\""

    [xloc,yloc, yaw, cls] = contents
    xloc = float(xloc)
    yloc = float(yloc)
    yaw = float(yaw)
    assert cls in ["[", "]"], f"Expected [ or ] but got '{cls}'"
    cls = 1 if cls=="[" else 2

    return 0, {'xloc': xloc, 'yloc':yloc, 'yaw': yaw, 'cls':cls }

def parse_textbox_labels(text):
    ''' Expecting a semicolon separated string of "xloc, yloc, [ or ]" as substrings '''
    
    try:
        
        substrings = [substring.strip() for substring in text.split(";")]
        substrings = [substring for substring in substrings if substring!=""]
        
        labels = [parse_label_string(substring)[1] for substring in substrings]
        return len(labels), labels
    
    except AssertionError as e:
        return -1, e
    
def label_2_string(label):
    string_els = []
    string_els.append(label['xloc'])
    string_els.append(label['yloc'])
    string_els.append(label['yaw'])
    
    # print("label['cls']", label['cls'])
    string_els.append("[" if label['cls']==1 else "]")
    string_els = [str(el) for el in string_els]
    string = ", ".join(string_els)
    return string
    
def labels_2_string(labels):
    string = ";\n".join([label_2_string(label) for label in labels])
    return string

def parse_labelfile(labelfile_path):
    welds = []
    label_dict = {}
    with open(labelfile_path, 'r') as labelfile:
        lines = labelfile.readlines()
        for line in lines:
            line = line.strip().strip(";")
            xloc, yloc, yaw, cls = "0, 0, 0, [".split(",")
            if line.startswith(("image_w", "image_h", "weld_w", "weld_h")):
                key, value = line.split()
                label_dict[key.strip()] = float(value.strip())
            else:
                line_els = line.split(",")
                if len(line_els) == 4:
                    xloc, yloc, yaw, cls = line_els
                elif len(line_els) ==3:
                    xloc, yloc, cls = line_els
                # print(cls)
                welds.append({"xloc": float(xloc.strip()),
                                "yloc": float(yloc.strip()),
                                "yaw" : float(yaw.strip()),
                                "cls": 1 if cls.strip()=="[" else 2})
    label_dict["welds"] = welds
    return label_dict
            
def parse_dataset_file(data_set_filepath):
    """
    Looks at the given text file that list scan and label path per row eg test.txt
    Returns:
     - a list of dicts, where each contains the "scan_path", the "label_path" and the sample "name"
    """
    
    samples = []
    with open(data_set_filepath, 'r') as setfile:
        lines = setfile.readlines()
        for line in lines:
            line = line.strip().strip("[]") # removing the [] bookending each line
            scan, label = line.split(",")
            scan_path = scan.strip().strip("'")
            label_path = label.strip().strip("'")
            name = label_path.split("/")[-1].strip(".txt")
            sample = {"scan_path": scan_path, "label_path": label_path, "name": name}
            samples.append(sample)
    return samples