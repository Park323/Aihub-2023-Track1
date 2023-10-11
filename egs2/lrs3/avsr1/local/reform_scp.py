# Jeongkyun Park, Sogang Univ.
# 2023

import os
import sys

import re
import glob

def replace_path(path:str, search_pattern:str, sub_pattern:str, condition:str=None):
    if condition is None or re.search(condition, path) is not None:
        return re.sub(search_pattern, sub_pattern, path)
    else:
        return path
    
def main(args):
    assert len(args) >= 2, args
    scp_path = args[0]
    with open(scp_path, 'r') as f:
        file_paths = f.readlines()
    with open(scp_path + '.backup', 'w') as f:
        [f.write(path) for path in file_paths]
    with open(scp_path, 'w') as f:
        for path in file_paths:
            path = replace_path(path, *args[1:])
            f.write(path)

if __name__=='__main__':
    main(sys.argv[1:])