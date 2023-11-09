#!/usr/bin/python

import pathlib
import readline
import pandas as pd
from colorama import init, Fore, Style
import os
import sys

################################################################################
#### Input tab completion for files                                         ####
################################################################################
# from from Peter Mitrano via stack overflow
def complete_path(text, state):
    incomplete_path = pathlib.Path(text)
    if incomplete_path.is_dir():
        completions = [p.as_posix() for p in incomplete_path.iterdir()]
    elif incomplete_path.exists():
        completions = [incomplete_path]
    else:
        exists_parts = pathlib.Path('.')
        for part in incomplete_path.parts:
            test_next_part = exists_parts / part
            if test_next_part.exists():
                exists_parts = test_next_part

        completions = []
        for p in exists_parts.iterdir():
            p_str = p.as_posix()
            if p_str.startswith(text):
                completions.append(p_str)
    return completions[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete_path)
################################################################################

#allow the user to exit anytime
def check_exit(in_text):
    if in_text == 'exit':
        print("You have chosen to exit.")
        sys.exit()
    return in_text


def input_loop(initial_prompt, file_type, label_prompt=True, prompt_multiple=False):
    records = []
    done = False
    yn = check_exit(input(Fore.GREEN + initial_prompt + Style.RESET_ALL))
    if yn == 'n':
        return []
    while not done:
        file = check_exit(input(Fore.GREEN + file_prompt + Style.RESET_ALL))
        if label_prompt:
            label = check_exit(input(Fore.GREEN + name_prompt + Style.RESET_ALL))
        else:
            label = file_type
        records.append({
            'filepath': os.path.abspath(file),
            'type': file_type,
            'label': label}
        )
        if prompt_multiple:
            valid_end = False
            while not valid_end:
                another = check_exit(input(Fore.GREEN + more_prompt + Style.RESET_ALL))
                if another == '1':
                    valid_end = True
                    done = True
                elif another == '2':
                    valid_end = True
        else:
            done = True
    return records

#Prompts
pa_prompt = 'Do you have a feature presence/absence table? (y/n)'
dm_prompt = 'Do you have one or more distance matrices? (y/n)'
tree_prompt = 'Do you have a treefile? (y/n)'
meta_prompt = 'Do you have any metadata files? (y/n)'
file_prompt = 'Enter filepath:'
name_prompt = "Please name the file:"
more_prompt = 'Enter 1 to continue, 2 to enter another file:'
###
print("Welcome to Indizio.")
print("Please make use of this utility to format your input data sheet.")
print('You may exit any time by typing "exit" and hitting enter.')
outfile = check_exit(input(Fore.GREEN + 'Please name your spreadsheet:' + Style.RESET_ALL))
pa = input_loop(pa_prompt, 'P', False, False)
dm = input_loop(dm_prompt, 'DM', True, True)
if not pa and not dm:
    sys.exit("You require either a presence/absence matrix or at least one distance matrix.")
tree = input_loop(tree_prompt, 'T', False, False)
meta = input_loop(meta_prompt, 'M', True, True)

records = pa + dm + tree + meta
df = pd.DataFrame.from_records(records)
df.to_csv(outfile, sep=',', index=False)
