# author: Jelly Lee
# create date: 06/22/2024
# last modification: 06/22/2024

import os


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
