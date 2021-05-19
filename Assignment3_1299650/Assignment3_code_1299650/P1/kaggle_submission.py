import os
import csv
import numpy as np


def write_csv(file_path, y_list):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id'] + ['category'])
        writer.writerow(['1'] + [np.array(y_list)])


def output_submission_csv(output_file_path, y_test):
    write_csv(output_file_path, y_test)
