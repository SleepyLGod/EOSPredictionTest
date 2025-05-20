import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DIR_FOR_TRAINING_DATA = Path("../training_data/ebd/features/llama3_70b/")
OUTPUT_USED_IDS_FILE = Path("./used_prompt_ids.txt")

def extract_and_save_used_ids(feature_dir: Path, output_file: Path):
    if not feature_dir.exists() or not feature_dir.is_dir():
        logger.error(f"Feature directory for training data not found: {feature_dir}")
        return
    
    used_qa_ids = set()
    id_pattern = re.compile(r"_qa(\d+)\.npz$") 
    
    logger.info(f"Scanning directory: {feature_dir} for .npz files...")
    file_count = 0
    for filename in os.listdir(feature_dir):
        if filename.endswith(".npz"):
            file_count += 1
            match = id_pattern.search(filename)
            if match:
                qa_id_str = match.group(1)
                try:
                    used_qa_ids.add(int(qa_id_str)) 
                except ValueError:
                    logger.warning(f"Could not parse ID from filename: {filename}")
            # else:
                # logger.debug(f"Filename {filename} did not match ID pattern.")
    
    logger.info(f"Scanned {file_count} .npz files. Found {len(used_qa_ids)} unique QA IDs.")
    if used_qa_ids:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for qa_id in sorted(list(used_qa_ids)): #排序后写入，方便查看
                f.write(str(qa_id) + '\n')
        logger.info(f"Successfully saved {len(used_qa_ids)} used QA IDs to {output_file}")
    else:
        logger.warning("No QA IDs were extracted. The output file will be empty or not created.")

if __name__ == "__main__":
    if not FEATURE_DIR_FOR_TRAINING_DATA.exists():
        logger.error(f"CRITICAL: The specified feature directory for training data generation does not exist: {FEATURE_DIR_FOR_TRAINING_DATA}")
        logger.error("Please ensure the path is correct. This script needs to scan .npz filenames from that directory.")
    else:
        extract_and_save_used_ids(FEATURE_DIR_FOR_TRAINING_DATA, OUTPUT_USED_IDS_FILE)