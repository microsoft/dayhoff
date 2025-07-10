import argparse
import os
import shutil

# Define problem sets
RF_PROBLEMS = ['00_1PRW', '01_1BCF', '02_5TPN', '03_5IUS', '04_3IXT', '05_5YUI', '06_1QJG', '07_1YCR', '08_2KL8', '09_7MRX', '10_7MRX', '11_7MRX', '12_4JHW', '13_4ZYP', '14_5WN9', '15_6VW1', '16_5TRV', '17_5TRV', '18_5TRV', '19_6E6R', '20_6E6R', '21_6E6R', '22_6EXZ', '23_6EXZ', '24_6EXZ']
MOTIFBENCH_PROBLEMS = ['00_1LDB', '01_1ITU', '02_2CGA', '03_5WN9', '04_5ZE9', '05_6E6R', '06_6E6R', '07_7AD5', '08_7CG5', '09_7WRK', '10_3TQB', '11_4JHW', '12_4JHW', '13_5IUS', '14_7A8S', '15_7BNY', '16_7DGW', '17_7MQQ', '18_7MQQ', '19_7UWL', '20_1B73', '21_1BCF', '22_1MPY', '23_1QY3', '24_2RKX', '25_3B5V', '26_4XOJ', '27_5YUI', '28_6CPA', '29_7UWL']


def organize_scaffold_outputs(input_dir, output_dir, pdb_dir, offset_index=False):
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

    # Iterate over each case directory in the input directory
    for case_dir in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case_dir, 'pdb/esmfold') # case_dir = pdb name folder only, case_path = full path to the case dir
        # Skip if not a directory
        if not os.path.isdir(case_path):
            print(f"Skipping {case_dir}: not a directory")
            continue
        
        # Determine which problem set this case belongs to and make dir if not exist
        problem_set = RF_PROBLEMS if 'rf' in case_path else MOTIFBENCH_PROBLEMS if 'motif' in case_path else None
        
        if problem_set is not None: 
            if case_dir in problem_set: 
                # Create the output directory for this case
                new_num, case_dir_postfix = case_dir.split('_')
                if offset_index:
                    new_num = int(new_num) + 1
                else: 
                    new_num = int(new_num)
                out_case_dir = "{:02}_{case_dir_postfix}".format(new_num, case_dir_postfix=case_dir_postfix)
                output_case_path = os.path.join(output_dir, out_case_dir)
                os.makedirs(output_case_path, exist_ok=True)
            
                print(f"Processing case: {case_dir}")

                # Find all PDB files in the case directory
                pdb_files = os.listdir(case_path)

                # Copy scaffold_motif.csv into the correct dir 
                shutil.copy2(os.path.join(pdb_dir, case_dir, 'scaffold_info.csv'), os.path.join(output_case_path, 'scaffold_info.csv'))
                for src_pdb in pdb_files:
                    # Handle standard generation format
                    if '.json_' in src_pdb:
                        dest_pdb = src_pdb.replace('.json', '')
                    else: 
                        dest_pdb = src_pdb
                    pdb_id, case, num = dest_pdb.split('_')
                    dest_filename = "{:02}_{case}_{num}".format(int(pdb_id)+1, case=case, num=num)
                    
                        # # Skip files that don't match expected patterns
                        # print(f"  Skipping {src_pdb}: unrecognized format")
                        # continue
                        
                    dest_path = os.path.join(output_case_path, dest_filename)
                    
                    # Copy the file
                    shutil.copy2(os.path.join(case_path, src_pdb), dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize scaffold outputs into the required directory structure and metadata format.")
    parser.add_argument("--input_dir", type=str, required=True, help="The input directory containing scaffold cases.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory for cleaned files.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Location where folders containing contig csvs are located")
    parser.add_argument("--offset_index", action="store_true", help="MotifBench problems are indexed at 01, I accidentally index at 00 - need to offset results")

    args = parser.parse_args()

    organize_scaffold_outputs(args.input_dir, args.output_dir, args.pdb_dir, args.offset_index)