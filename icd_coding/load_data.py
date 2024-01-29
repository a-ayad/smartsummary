from collections import defaultdict
import csv
import numpy as np
import pandas as pd


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def load_code_descriptions():
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    with open("mimicdata/D_ICD_DIAGNOSES.csv", 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            desc_dict[reformat(code, True)] = desc
    with open("mimicdata/D_ICD_PROCEDURES.csv", 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            if code not in desc_dict.keys():
                desc_dict[reformat(code, False)] = desc
    with open('mimicdata/ICD9_descriptions', 'r') as labelfile:
        for i, row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])
    return desc_dict


if __name__ == '__main__':
    desc_dict = load_code_descriptions()
    code_df = pd.read_csv("mimicdata/TOP_50_CODES.csv", header=None)
    all_codes = sorted(code_df.iloc[:, 0].tolist())
    ind2c = {index: code for index, code in enumerate(all_codes)}

    preds = vector2code(prediction, all_codes, prob=True, return_index=True)
    input()