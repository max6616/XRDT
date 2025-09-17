#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from tkinter import N
from tqdm import tqdm
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
import multiprocessing
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import Optional, Tuple, List


mpr: Optional[MPRester] = None
WORKER_API_KEY: Optional[str] = None

def parse_args():
    parser = argparse.ArgumentParser(description="Download symmetrized CIFs from Materials Project in parallel.")
    parser.add_argument('--api-key', default=None, help='Materials Project API key (default: env MP_API_KEY or MAPI_KEY)')
    parser.add_argument('--output-dir', default='/root/autodl-tmp/cif/mp_all', help='Output directory to save CIF files')
    parser.add_argument('--ids-file', default='all_material_ids.txt', help='Path to cached material IDs file')
    parser.add_argument('--processes', type=int, default=(os.cpu_count() or 1), help='Number of worker processes; use 1 for sequential')
    parser.add_argument('--chunksize', type=int, default=10, help='Chunksize for imap_unordered')
    parser.add_argument('--symprec', type=float, default=0.1, help='Symmetry precision for SpacegroupAnalyzer/CifWriter')
    parser.add_argument('--report-width', type=int, default=50, help='Width for report separators')
    return parser.parse_args()

def init_worker(api_key: str):
    global mpr, WORKER_API_KEY
    WORKER_API_KEY = api_key
    mpr = MPRester(api_key)

def worker(material_id: str, output_dir: str, symprec: float) -> Tuple[str, str]:
    global mpr
    if mpr is None:
        raise RuntimeError("Worker not initialized with API key. Initializer was not called.")
    try:
        primitive_structure = mpr.get_structure_by_material_id(material_id, final=True, conventional_unit_cell=True)
        analyzer = SpacegroupAnalyzer(primitive_structure, symprec=symprec)
        symmetrized_structure = analyzer.get_symmetrized_structure()
        cif_writer = CifWriter(symmetrized_structure, symprec=symprec)
        filename = f"{material_id.split('-')[1]}.cif"
        cif_writer.write_file(os.path.join(output_dir, filename))
        return (material_id, "Success")
    except Exception as e:
        return (material_id, f"Error: {e}")

def download_all_structures_parallel(api_key: str, output_dir: str, ids_file: str, processes: int, chunksize: int, symprec: float, report_width: int) -> Tuple[int, int, List[Tuple[str, str]]]:
    if os.path.exists(ids_file):
        print(f"Reading material IDs from local file '{ids_file}'...")
        with open(ids_file, "r") as f:
            all_material_ids = [line.strip() for line in f]
        print(f"Successfully fetched {len(all_material_ids)} material IDs.")
    else:
        print("Fetching all material IDs from Materials Project...")
        try:
            with MPRester(api_key) as mpr_main:
                docs = mpr_main.materials.search(fields=["material_id"])
                all_material_ids = [doc.material_id for doc in docs]
            with open(ids_file, "w") as f:
                for mid in all_material_ids:
                    f.write(f"{mid}\n")
            print(f"Successfully fetched and saved {len(all_material_ids)} material IDs.")
        except Exception as e:
            print(f"Error fetching material ID list from Materials Project: {e}")
            return (0, 0, [])

    total_files = len(all_material_ids)
    if total_files == 0:
        print("No material IDs to download.")
        return (0, 0, [])

    print("Start downloading structure files in parallel...")
    print(f"Using {processes} processes for parallel download.")
    os.makedirs(output_dir, exist_ok=True)

    if processes <= 1:
        init_worker(api_key)
        results = []
        for mid in tqdm(all_material_ids, total=total_files, desc="Downloading structures"):
            results.append(worker(mid, output_dir=output_dir, symprec=symprec))
    else:
        from functools import partial
        worker_fn = partial(worker, output_dir=output_dir, symprec=symprec)
        with multiprocessing.Pool(processes=processes, initializer=init_worker, initargs=(api_key,)) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_fn, all_material_ids, chunksize=chunksize),
                total=total_files, desc="Downloading structures"
            ))
    success_count = sum(1 for _, status in results if status == "Success")
    failed_items = [item for item in results if item[1] != "Success"]

    print("\n" + "="*report_width)
    print("Download finished!")
    print(f"Total: {total_files} files")
    print(f"Success: {success_count} files")
    print(f"Failed: {len(failed_items)} files")
    print(f"All successful CIF files are saved in '{output_dir}' directory.")

    if failed_items:
        print("\nThe following material_id failed:")
        for material_id, error_message in failed_items:
            print(f"  - {material_id}: {error_message}")

    print("="*report_width)

    return (total_files, success_count, failed_items)


if __name__ == '__main__':
    multiprocessing.set_start_method("fork", force=True)
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key is required. Provide via --api-key or set MP_API_KEY/MAPI_KEY.")
    download_all_structures_parallel(
        api_key=args.api_key,
        output_dir=args.output_dir,
        ids_file=args.ids_file,
        processes=args.processes,
        chunksize=args.chunksize,
        symprec=args.symprec,
        report_width=args.report_width,
    )