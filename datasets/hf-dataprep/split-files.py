import argparse
import os
from logging import getLogger

logger = getLogger(__name__)

class FileSplitter:
    def __init__(self,input_path:str, output_dir:str, max_chunk_gb:float):
        self.input_path = input_path
        self.output_dir = output_dir
        self.max_chunk_gb = max_chunk_gb
        self.max_bytes = max_chunk_gb * 1024 ** 3
        self.file_name = os.path.basename(input_path)
        self.file_name_stem, self.extension = self.file_name.split(".")

        #States
        self.current_chunk = None
        self.current_chunk_size = 0
        self.current_chunk_idx = 0
        
        os.makedirs(output_dir, exist_ok=True)

    def init_chunk(self):
        if not self.current_chunk:
            chunk_path = os.path.join(self.output_dir, f"{self.file_name_stem}_{self.current_chunk_idx:05d}.{self.extension}")
            self.current_chunk = open(chunk_path, "w")
            self.current_chunk_size = 0
        else:
            raise ValueError("Chunk already initialized. Close the current chunk before initializing a new one.")

    def close_chunk(self):
        if self.current_chunk:
            self.current_chunk.close()
            self.current_chunk = None
            logger.info(f"Wrote file {self.file_name_stem}_{self.current_chunk_idx:05d}.{self.extension}")
        else:
            raise ValueError("No chunk is currently open.")
            

    def init_record(self):
        record = []
        record_size = 0
        return record, record_size
    
    def write_record(self, record,record_size):
        if self.current_chunk_size + record_size > self.max_bytes:
            self.close_chunk()
            self.current_chunk_idx += 1
            self.init_chunk()
        self.current_chunk.writelines(record)
        self.current_chunk_size += record_size


    def _split_fasta(self):
        # Initialize chunk and record
        self.init_chunk()
        record, record_size = self.init_record()

        with open(self.input_path, "r") as infile:
            for line in infile:
                if line.startswith(">") and record:
                    self.write_record(record,record_size) #Will write the record in current or new chunk depending on the size
                    # After writting the record, we need to reset the record and its size
                    record, record_size = self.init_record()
                record_size += len(line.encode('utf-8'))
                record.append(line)

            # Write the last record
            if record:
                self.write_record(record,record_size)
                self.close_chunk()

    def _split_jsonl(self):
        # Initialize chunk and record
        #TODO: FINISH THIS CODE
        self.init_chunk()
        record, record_size = self.init_record()

        with open(self.input_path, "r") as infile:
            for line in infile:
                record_size += len(line.encode('utf-8'))
                record.append(line)
                self.write_record(record,record_size)
                record, record_size = self.init_record()
            self.close_chunk()   

    def split(self):
        if self.extension == "fasta":
            self._split_fasta()
        elif self.extension == "jsonl":
            self._split_jsonl()
        else:
            raise ValueError(f"Unsupported file extension: {self.extension}")

            

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Split a file into smaller files.")
    parser.add_argument("--input-files", help="List of input files to split.", nargs="+")
    parser.add_argument("--output-dir", type=str, help="The directory to save the split files.")
    parser.add_argument("--max-chunk-gb", type=float, default=40, help="The number of lines per chunk.")
    args = parser.parse_args()

    is_amlt = os.environ.get("AMLT_OUTPUT_DIR", None) is not None

    if is_amlt:
        # If running in AMLT, set the output directory to the AMLT output directory
        args.output_dir = os.path.join(os.environ["AMLT_DIRSYNC_DIR"], args.output_dir)

        # prepend input files with AMLT input directory
        args.input_files = [os.path.join(os.environ["AMLT_DATA_DIR"], f) for f in args.input_files]

        logger.info("AMLT_OUTPUT_DIR:", os.environ.get("AMLT_OUTPUT_DIR", None))
        logger.info("AMLT_DATA_DIR:", os.environ.get("AMLT_DATA_DIR", None))
        logger.info("AMLT_DIRSYNC_DIR:", os.environ.get("AMLT_DIRSYNC_DIR", None))
        
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    
    logger.info("Writing split files to:", args.output_dir)

    for input_file in args.input_files:
        print(f"Splitting {input_file} into smaller files...")
        splitter = FileSplitter(
            input_path = input_file,
            output_dir = args.output_dir,
            max_chunk_gb = args.max_chunk_gb
            )
        splitter.split()
