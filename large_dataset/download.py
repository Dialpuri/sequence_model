import os.path
import urllib.request
from multiprocessing import Pool
from tqdm import tqdm


def worker(pdb):
    pdb_url = f"https://files.rcsb.org/download/{pdb}.pdb"
    mtz_url = f"https://edmaps.rcsb.org/coefficients/{pdb.lower()}.mtz"

    pdb_path = f"large_dataset/pdb_files/{pdb.lower()}.pdb"
    mtz_path = f"large_dataset/mtz_files/{pdb.lower()}.mtz"

    if os.path.exists(pdb_path) and os.path.exists(mtz_path):
        return

    try:
        urllib.request.urlretrieve(pdb_url, pdb_path)
        urllib.request.urlretrieve(mtz_url, mtz_path)
    except:
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
        return

def main():
    file_name = "large_dataset/dataset.txt"
    pdb_list = []
    with open(file_name, "r") as file_:
        for line in file_:
            pdb_list = line.split(",")

    with Pool() as pool:
        x = list(tqdm(pool.imap_unordered(worker, pdb_list), total=len(pdb_list)))


if __name__ == "__main__":
    main()
