"""
This script processes raw outputs from attack_dt2.py
It extracts the losses and outputs all values to a .csv
matching the input filename
usage: python src/results.py -i results/results.txt
output: results.csv
"""
import re
import csv
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file path.")
    args = parser.parse_args()
    
    fp = args.input
    base = os.path.basename(fp)
    base = os.path.splitext(base)[0]
    dir = os.path.dirname(fp)
    op = os.path.join(dir,f"{base}.csv")

    losses = []
    with open(fp, 'r') as file:
        for line in file:
            if "loss" in line:
                result = re.search(r'loss: (\d+\.\d+)', line)
                if result:
                    extracted_number = float(result.group(1))
                    losses.append(extracted_number)
                    print(extracted_number)
                else:
                    print("Number not found in string.") 
    
    with open(op, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(losses)
