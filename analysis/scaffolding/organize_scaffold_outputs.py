import argparse
import os
import csv
import re
import shutil
import glob

# Define problem sets
RF_PROBLEMS = ['00_1PRW', '01_1BCF', '02_5TPN', '03_5IUS', '04_3IXT', '05_5YUI', '06_1QJG', '07_1YCR', '08_2KL8', '09_7MRX', '10_7MRX', '11_7MRX', '12_4JHW', '13_4ZYP', '14_5WN9', '15_6VW1', '16_5TRV', '17_5TRV', '18_5TRV', '19_6E6R', '20_6E6R', '21_6E6R', '22_6EXZ', '23_6EXZ', '24_6EXZ']
MOTIFBENCH_PROBLEMS = ['00_1LDB', '01_1ITU', '02_2CGA', '03_5WN9', '04_5ZE9', '05_6E6R', '06_6E6R', '07_7AD5', '08_7CG5', '09_7WRK', '10_3TQB', '11_4JHW', '12_4JHW', '13_5IUS', '14_7A8S', '15_7BNY', '16_7DGW', '17_7MQQ', '18_7MQQ', '19_7UWL', '20_1B73', '21_1BCF', '22_1MPY', '23_1QY3', '24_2RKX', '25_3B5V', '26_4XOJ', '27_5YUI', '28_6CPA', '29_7UWL']

def determine_problem_set(case_dir):
    """
    Determine which problem set a case belongs to.
    
    Args:
        case_dir (str): The name of the case directory.
        
    Returns:
        str: Either 'rf' or 'motif_bench' depending on which set the case belongs to.
    """
    if case_dir in RF_PROBLEMS or case_dir in MOTIFBENCH_PROBLEMS:
        if case_dir in RF_PROBLEMS: 
            return 'rf'
        elif case_dir in MOTIFBENCH_PROBLEMS:
            return 'motif_bench'
    elif case_dir in RF_PROBLEMS and case_dir in MOTIFBENCH_PROBLEMS: 
        print(case_dir) # TODO 12_4JHW is in both problem sets, will assume the later run (rf) - Caution while using and re-number this problem 
        print(f"case_dir {case_dir} is in both RF and MOTIF_BENCH problem sets. Defaulting to RF. CHECK THIS.")
        return 'rf'
    else:
        print(f"Case {case_dir} not found in predefined problem sets.")
        return None

def organize_scaffold_outputs(input_dir, output_dir, already_divided=False):
    """
    Organize scaffold outputs into the required directory structure and metadata format.

    Args:
        input_dir (str): The input directory containing scaffold cases.
        rf_output_dir (str): The output directory for RF problems.
        motif_bench_output_dir (str): The output directory for MOTIF_BENCH problems.
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    save_file_names = True if 'dayhoff' in input_dir else False

    # Iterate over each case directory in the input directory
    for case_dir in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case_dir, 'pdb/esmfold') # case_dir = pdb name folder only, case_path = full path to the case dir

        # Skip if not a directory
        if not os.path.isdir(case_path):
            continue

        # Determine which problem set this case belongs to and make dir if not exist
        if already_divided: 
            problem_set = 'rf' if 'rf' in case_path else 'motif_bench' if 'motif' in case_path else None
        else:
            problem_set = determine_problem_set(case_dir)
        if problem_set is not None: 
            output_case_path = os.path.join(output_dir, problem_set, case_dir)
            os.makedirs(output_case_path, exist_ok=True)
        
            print(f"Processing case: {case_dir} (Problem set: {problem_set})")

            # Find all PDB files in the case directory
            pdb_files = os.listdir(case_path)

            # Copy and rename PDB files to the output directory
            output_pdb_files = []
            for src_pdb in pdb_files:
                sample_num = re.search(r'generation_(\d+)', src_pdb).group(1)
                dest_filename = f"{case_dir}_{sample_num}.pdb"
                dest_path = os.path.join(output_case_path, dest_filename)
                
                # Copy the file
                shutil.copy2(os.path.join(case_path, src_pdb), dest_path)
                #print(f"  Copied {os.path.basename(src_pdb)} to {dest_filename}")
                if save_file_names: 
                    # Save the original filename in the output directory
                    os.makedirs(os.path.join(output_dir, problem_set, 'og_file_names'), exist_ok=True)
                    with open(os.path.join(output_dir, problem_set, 'og_file_names', case_dir + '.txt'), 'a') as f:
                        f.write(src_pdb+ "\n")
                
                output_pdb_files.append(dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize scaffold outputs into the required directory structure and metadata format.")
    parser.add_argument("--input_dir", type=str, required=True, help="The input directory containing scaffold cases.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory for RF problems.")
    parser.add_argument("--already_divided", action="store_true", help="Folders are already divided into rf and motif_bench. Do not create new folders.")

    args = parser.parse_args()

    organize_scaffold_outputs(args.input_dir, args.output_dir, args.already_divided)