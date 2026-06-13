import argparse

from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
import os
from Bio.SeqUtils import seq1 as three_to_one
import pandas as pd
import glob
import tqdm
import gzip
from multiprocessing import Pool
import warnings
from Bio import BiopythonWarning, SeqIO
from parallelbar import progress_map
import subprocess
from collections import defaultdict
import json
import pickle
import logging
import shlex
import re
from datetime import datetime
from func_timeout import func_set_timeout, FunctionTimedOut

# Suppress Biopython warnings
warnings.simplefilter('ignore', BiopythonWarning)

# Constants
ATOMS = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG']

common_aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
common_aas_1 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA3_SET = set(common_aas)
AA1_SET = set(common_aas_1)
VALID_ALT_LOCS = {'A', ''}

rename_dict = {'auth_asym_id': 'chain_id', 'auth_seq_id': 'residue_number', 'auth_comp_id': 'residue_name', 'auth_atom_id': 'atom_name', 'label_alt_id': 'alt_loc', 'pdbx_PDB_ins_code': 'insertion'}

EXCLUDE_PDBS = ['5z8w', '7why', '7a7y', '7a7w', '7ttb']
LARGE_PDBS = ['9cpc', '9e2g', '8p9a', '8i7r', '11eq', '9mjn', '9ijj', '4u4u', '8g2z', '6wln', '8otz', '8qo1', '6kgx', '6hhq', '5dfe', '3j3q', '6qz0', '4u3n', '1htq', '5y6p', '8bqs', '6u42', '7y5e', '3j3y', '4tyv', '8glv', '5vyc', '4u4y', '8iyj', '11jb', '5mei', '9y6s', '7vs5', '7ezx', '5ndv', '4u6f', '5ivh', '7mho', '4p3r', '7sqc', '6wlo', '9y9z', '5ivk', '5ijn', '4u4o', '4u4r', '4u56', '4u51', '5ndw', '8qo0', '4u52', '8sf7', '8j07', '4u4z', '6tml', '9fqr', '6qvk', '9v10', '4p3q', '8p4v', '8ckb', '2ku2', '7ung', '4u4n', '4tz5', '6qyd', '4v4g', '2hyn', '9cpb', '4u3u', '9qw9', '4u55', '9mkb', '4cbo', '7y4l', '7n6g', '8to0', '2kox', '6osi', '6x63', '8g4l', '4u3m', '4pth', '6q3g', '5on6', '8snb', '9ly9', '9d5n', '8g3d', '5ndg', '8rqe', '9e5c', '7y7a', '4u50', '9e78', '4u4q', '5tbw', '5obm', '4u53', '7rro']

parser = argparse.ArgumentParser(description='Process PDB files to extract mutation information.')
parser.add_argument('--pdb_root', type=str, default='/rds/projects/l/liuje-multiai/shuo/datasets', help='Directory containing PDB files')
parser.add_argument('--output_dir', type=str, default='/rds/homes/s/sxz325/shuo/mutation/tmp/muts_data', help='Directory to save extracted mutation information')
parser.add_argument('--multi_site', action='store_true', help='Flag to indicate if multi-site mutations should be processed')
parser.add_argument('--pdb_version', type=str, default='pdb_241028', help='pdb_241028 pdb_260603')
parser.add_argument('--pdb_format', type=str, default='pdb', help='pdb mmcif')
parser.add_argument('--num_workers', type=int, default=-1)
parser.add_argument('--re_symlink_mut', action='store_true', help='Whether to restart from scratch instead of using cached intermediate files')
parser.add_argument('--re_group_seqadv', action='store_true', help='Whether to restart from scratch instead of using cached intermediate files')
parser.add_argument('--re_mutations', action='store_true', help='Whether to restart from scratch instead of using cached intermediate files')
parser.add_argument('--re_seqfasta', action='store_true', help='Whether to restart from scratch instead of using cached intermediate files')
parser.add_argument('--re_wholefasta', action='store_true')
parser.add_argument('--re_genmatchesm8', action='store_true')
parser.add_argument('--re_gen_matching_dict', action='store_true')
parser.add_argument('--re_internalcsv', action='store_true')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

logging.basicConfig(
	filename=os.path.join(args.output_dir, "process_data.log"),
	filemode="w",
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
)
logger = logging.getLogger("process_data")

dataset_prefix = "MultiMutPairs2024" if args.multi_site else "SingleMutPairs2024"
pair_internal_csv = os.path.join(args.output_dir, f'{dataset_prefix}_internal_pairs.csv')  # internal file used before cluster_id_30 is known
pair_csv = os.path.join(args.output_dir, f'{dataset_prefix}.csv')
cluster_pkl = os.path.join(args.output_dir, f'{dataset_prefix}_cluster30.pkl')

REQUIRED_DATASET_COLUMNS = [
	"sample_id",
	"wt_pdb_id",
	"wt_chain_id",
	"mut_pdb_id",
	"mut_chain_id",
	"mut_pos_seq_index",
	"mut_pos_pdb_number",
	"wt_aa_type",
	"mut_aa_type",
	"wt_sequence",
	"mut_sequence",
	"cluster_id_30",
	"release_date",
]

PAIR_COLUMNS_BEFORE_CLUSTER = [col for col in REQUIRED_DATASET_COLUMNS if col != "cluster_id_30"]
INTERNAL_PAIR_COLUMNS = PAIR_COLUMNS_BEFORE_CLUSTER + ["mut_chain", "wt_chain"]

def log_skip(reason, pdb_id=None, chain_id=None, extra=None):
	logger.warning(
		"skip | reason=%s | pdb_id=%s | chain_id=%s | extra=%s",
		reason,
		pdb_id,
		chain_id,
		extra,
	)

MAX_LEN = 512 if args.multi_site else 2048
MIN_LEN = 32
N_CPUS = max(1, len(os.sched_getaffinity(0)) - 2) if args.num_workers < 0 else args.num_workers

# utils
@func_set_timeout(60)
def safe_read_mmcif(path):
	return PandasMmcif().read_mmcif(path)

@func_set_timeout(60)
def safe_read_pdb(path):
	return PandasPdb().read_pdb(path)

def filter_residues(df, chain_id=None):
	if chain_id is not None:
		df = df[df['chain_id'] == chain_id]
	df = df.copy()
	if 'alt_loc' not in df.columns:
		df['alt_loc'] = ''
	df['alt_loc'] = (
		df['alt_loc']
		.fillna('')
		.astype(str)
		.str.strip()
    .replace({'.': '', '?': '', 'None': ''})
	)	
	df = df[
		(df['atom_name'].isin(ATOMS)) &
		(df['alt_loc'].isin(VALID_ALT_LOCS)) &
		(df['residue_name'].isin(AA3_SET))
	].copy()
	residue_key_cols = ['chain_id', 'residue_number']
	if 'insertion' in df.columns:
		residue_key_cols.append('insertion')
	ca_residues = df[df['atom_name'] == 'CA'][residue_key_cols].drop_duplicates()
	df = df.merge(ca_residues, on=residue_key_cols, how='inner')
	return df

def read_pdb(pdb_id, chain_id):
	try:
		pdb, _ = load_structure_file(pdb_id, args)
		return filter_residues(pdb.df['ATOM'], chain_id=chain_id)
	except FunctionTimedOut:
		log_skip("file_read_timeout", pdb_id=pdb_id, chain_id=chain_id)
		return pd.DataFrame()
	except Exception as e:
		log_skip("file_read_error", pdb_id=pdb_id, chain_id=chain_id, extra=str(e))
		return pd.DataFrame()

def load_structure_file(pdb_id, args):
	pdb_id = pdb_id.lower()
	if args.pdb_format == 'pdb':
		pdb_path = os.path.join(args.pdb_root, args.pdb_version, 'pdb', f'pdb{pdb_id}.ent.gz')
		pdb = safe_read_pdb(pdb_path)
		has_multi_model = len(pdb.get_model_start_end()) > 1
	elif args.pdb_format == 'mmcif':
		pdb_path = os.path.join(args.pdb_root, args.pdb_version, 'pdb', f'{pdb_id}.cif.gz')
		pdb = safe_read_mmcif(pdb_path)
		has_multi_model = pdb.df['ATOM']['pdbx_PDB_model_num'].nunique() > 1
		pdb.df['ATOM'] = pdb.df['ATOM'].rename(columns=rename_dict)
		pdb.df['ATOM']['residue_number'] = pd.to_numeric(pdb.df['ATOM']['residue_number'], errors='coerce')
	else:
		raise ValueError(f"Unknown pdb_format: {args.pdb_format}")
		
	return pdb, has_multi_model

def get_seq_and_mapping(pdb_df):
	if pdb_df.empty:
		return None, None
	residue_key_cols = ['residue_number']
	if 'insertion' in pdb_df.columns:
		residue_key_cols.append('insertion')
		
	unique_residues = pdb_df.drop_duplicates(subset=residue_key_cols).sort_values(by=residue_key_cols)[residue_key_cols + ['residue_name']]
	
	seq = ''.join(unique_residues['residue_name'].apply(three_to_one))
	
	id_mapping = {}
	for idx, row in enumerate(unique_residues.itertuples(index=False)):
		res_num = getattr(row, 'residue_number')
		ins_code = getattr(row, 'insertion', '')

		if pd.isna(ins_code) or ins_code in ('?', '.', 'None', 'nan'): 
			ins_code = ''
		else:
			ins_code = str(ins_code).strip()
		id_mapping[(res_num, ins_code)] = idx
	return seq, id_mapping

def split_pdb_chain_id(pdb_chain_id):
	parts = str(pdb_chain_id).split('_', 1)
	if len(parts) != 2 or not parts[0] or not parts[1]:
		raise ValueError(f"Invalid pdb_chain_id: {pdb_chain_id}")
	return parts[0].lower(), parts[1]

def get_structure_path(pdb_id):
	pdb_id = str(pdb_id).lower()
	if args.pdb_format == 'pdb': return os.path.join(args.pdb_root, args.pdb_version, 'pdb', f'pdb{pdb_id}.ent.gz')
	if args.pdb_format == 'mmcif': return os.path.join(args.pdb_root, args.pdb_version, 'pdb', f'{pdb_id}.cif.gz')
	raise ValueError(f"Unknown pdb_format: {args.pdb_format}")

def normalize_structure_date(raw_date):
	"""
	Return ISO data YYYY-MM-DD, or empty string if parsing fails
	Handles:
		PDB legacy data: 28-FEB-83
		mmCIF date: 1983-02-28
	"""
	if raw_date is None: return ""
	date_str = str(raw_date).strip().strip("'\"")
	if date_str in {"", "?", ".", "None", "nan"}: return ""
	date_str = date_str.split()[0]
	if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str): return date_str

	for fmt in ("%d-%b-%y", "%d-%b-%Y"):
		try:
			return datetime.strptime(date_str.upper(), fmt).strftime("%Y-%m-%d")
		except ValueError:
			continue
	return ""

@func_set_timeout(30)
def get_release_date(pdb_id):
	path = get_structure_path(pdb_id)
	try:
		release_date = ""
		deposition_date = ""
		with gzip.open(path, "rt") as f:
			if args.pdb_format == "pdb":
				for line in f:
					if line.startswith("HEADER"):
						deposition_date = normalize_structure_date(line[50:59])
					elif line.startswith("REVDAT"):
						date_candidate = normalize_structure_date(line[13:22])
						if date_candidate:
							if not release_date or date_candidate < release_date:
								release_date = date_candidate
				return release_date or deposition_date
			if args.pdb_format == "mmcif":
				for line in f:
					if line.startswith("_pdbx_database_status.date_of_PDB_release"):
						fields = line.split(maxsplit=1)
						if len(fields) == 2:
							release_date = normalize_structure_date(fields[1])
							if release_date: return release_date
					if line.startswith("_pdbx_database_status.recvd_initial_deposition_date"):
						fields = line.split(maxsplit=1)
						if len(fields) == 2: deposition_date = normalize_structure_date(fields[1])
				return deposition_date
	except FunctionTimedOut:
		log_skip("release_date_read_timeout", pdb_id=pdb_id)
	except Exception as e:
		log_skip("release_date_read_error", pdb_id=pdb_id, extra=str(e))
	return ""

def hamming_distance(seq_a, seq_b):
	if len(seq_a) != len(seq_b): return None
	return sum(a != b for a, b in zip(seq_a, seq_b))

def validate_required_dataset_df(df):
	missing = [col for col in REQUIRED_DATASET_COLUMNS if col not in df.columns]
	if missing:
		raise ValueError(f"Final dataset missing columns: {missing}")
	null_cols = []
	for col in REQUIRED_DATASET_COLUMNS:
		s = df[col]
		if s.isna().any() or s.astype(str).str.strip().eq("").any():
			null_cols.append(col)

	if null_cols:
		raise ValueError(f"Final dataset has null/empty values in columns: {null_cols}")

	bad_rows = []
	for row in df.itertuples(index=False):
		wt_seq = row.wt_sequence
		mut_seq = row.mut_sequence
		pos = int(row.mut_pos_seq_index)

		if len(wt_seq) != len(mut_seq):
			bad_rows.append((row.sample_id, "length_mismatch"))
			continue
		if hamming_distance(wt_seq, mut_seq) != 1:
			bad_rows.append((row.sample_id, "not_single_mutation"))
			continue
		if pos < 0 or pos >= len(mut_seq):
			bad_rows.append((row.sample_id, "mutation_position_out_of_range"))
			continue
		if wt_seq[pos] != row.wt_aa_type:
			bad_rows.append((row.sample_id, "wt_residue_mismatch"))
			continue
		if mut_seq[pos] != row.mut_aa_type:
			bad_rows.append((row.sample_id, "mut_residue_mismatch"))
			continue
		if not str(row.release_date).strip():
			bad_rows.append((row.sample_id, "missing_release_date"))
			continue
	if bad_rows:
		raise ValueError(f"Invalid final dataset rows: {bad_rows[:20]}")
	return True

def write_fasta(records, output_path):
	with open(output_path, 'w') as f:
		for record in records:
			if record:
				f.write(record)

def run_cmd(cmd, check=True):
	subprocess.run(cmd, check=check)

def reconstruct_wt_sequence(mut_seq, positions, wt_types):
	wt_seq = list(mut_seq)
	for pos, wt_type in zip(positions, wt_types):
		wt_seq[pos] = wt_type
	return "".join(wt_seq)

def validate_expected_difference_positions(mut_seq, wt_seq, expected_diff_positions):
	if len(wt_seq) != len(mut_seq):
		return False, "multi_site_candidate_wt_length_mismatch", None
	actual_diff_positions = {i for i, (mut_aa, wt_aa) in enumerate(zip(mut_seq, wt_seq)) if mut_aa != wt_aa}
	if actual_diff_positions != expected_diff_positions:
		if actual_diff_positions - expected_diff_positions:
			return False, "multi_site_candidate_wt_extra_differences", actual_diff_positions
		return False, "multi_site_candidate_wt_missing_expected_difference", actual_diff_positions
	return True, None, actual_diff_positions

def format_mutation_set(group):
	if group is None or len(group) == 0:
		log_skip("multi_site_empty_group")
		return None

	group = group.drop_duplicates().copy()
	if group.empty:
		log_skip("multi_site_empty_group")
		return None

	pdb_id = group["ID"].iloc[0]
	chain_id = group["CHAIN"].iloc[0]

	if len(group) < 2:
		log_skip("multi_site_single_record_in_multi_mode", pdb_id=pdb_id, chain_id=chain_id)
		return None

	if not group["WT"].isin(AA3_SET).all() or not group["MUT"].isin(AA3_SET).all():
		log_skip("nonstandard_amino_acid_in_mutation_set", pdb_id=pdb_id, chain_id=chain_id)
		return None

	for _, dup_group in group.groupby("POS"):
		if len(dup_group[["WT", "MUT"]].drop_duplicates()) > 1:
			log_skip(
				"multi_site_duplicate_conflicting_position",
				pdb_id=pdb_id,
				chain_id=chain_id,
				extra=f"idx0={dup_group['POS'].iloc[0]}",
			)
			return None

	for _, dup_group in group.groupby("MUT_RES_NUM"):
		if len(dup_group[["WT", "MUT"]].drop_duplicates()) > 1:
			log_skip(
				"multi_site_duplicate_conflicting_position",
				pdb_id=pdb_id,
				chain_id=chain_id,
				extra=f"pdb_resnum={dup_group['MUT_RES_NUM'].iloc[0]}",
			)
			return None

	group = group.sort_values(by=["POS", "MUT_RES_NUM"]).drop_duplicates(subset=["POS", "WT", "MUT", "MUT_RES_NUM"]).copy()
	positions = [int(pos) for pos in group["POS"].tolist()]
	mut_resnums = [int(resnum) for resnum in group["MUT_RES_NUM"].tolist()]
	wt_aas = [three_to_one(wt) for wt in group["WT"].tolist()]
	mut_aas = [three_to_one(mut) for mut in group["MUT"].tolist()]

	return {
		"wt_aas": ";".join(wt_aas),
		"mut_aas": ";".join(mut_aas),
		"mut_pos_idx0": ";".join(str(pos) for pos in positions),
		"mut_resnums_pdb": ";".join(str(resnum) for resnum in mut_resnums),
		"mutation_set_idx0": ";".join(f"{wt}{pos}{mut}" for wt, pos, mut in zip(wt_aas, positions, mut_aas)),
		"mutation_set_pdb": ";".join(f"{wt}{resnum}{mut}" for wt, resnum, mut in zip(wt_aas, mut_resnums, mut_aas)),
		"positions": positions,
		"wt_types_1": wt_aas,
		"mut_types_1": mut_aas,
		"num_mutations": len(positions),
	}

# def _run_multi_site_self_checks():
# 	mut_seq = "ACDEFG"
# 	positions = [1, 4]
# 	wt_types = ["A", "Y"]
# 	assert reconstruct_wt_sequence(mut_seq, positions, wt_types) == "AADEYG"

# 	ok, reason, actual = validate_expected_difference_positions(mut_seq, "AADEYG", {1, 4})
# 	assert ok and reason is None and actual == {1, 4}

# 	ok, reason, _ = validate_expected_difference_positions(mut_seq, "AADHYG", {1, 4})
# 	assert not ok and reason == "multi_site_candidate_wt_extra_differences"

# 	ok, reason, _ = validate_expected_difference_positions(mut_seq, "ACDEYG", {1, 4})
# 	assert not ok and reason == "multi_site_candidate_wt_missing_expected_difference"

# 	ok, reason, _ = validate_expected_difference_positions(mut_seq, "AADFFG", {1, 4})
# 	assert not ok and reason == "multi_site_candidate_wt_extra_differences"

# 	assert ['mut_chain', 'wtaa_pos_mutaa', 'wt_chain', 'mut_resnum_pdb'] == ['mut_chain', 'wtaa_pos_mutaa', 'wt_chain', 'mut_resnum_pdb']

# _run_multi_site_self_checks()

# Filter PDB files containing mutation information
## In PDB files, mutation information is in the SEQADV line, containing the ENGINEERED MUTATION' keyword.
## extract PDB files containing mutation information
if args.re_symlink_mut:
	def find_engineered_mutation_file(pdb_file):
		with gzip.open(pdb_file, 'rt') as gz_file:
			for line in gz_file:
				line_lower = line.lower()
				is_pdb_mutation = line.startswith('SEQADV') and 'engineered mutation' in line_lower
				is_mmcif_mutation = not line.startswith('_') and 'engineered mutation' in line_lower
				if is_pdb_mutation or is_mmcif_mutation:
					run_cmd(['ln', '-s', pdb_file, pdb_file.replace(os.path.join(args.pdb_root, args.pdb_version, 'pdb'), os.path.join(args.pdb_root, args.pdb_version, 'mutated_data'))])
					break

	os.makedirs(os.path.join(args.pdb_root, args.pdb_version, 'mutated_data'), exist_ok=True)
	pdb_files = glob.glob(os.path.join(args.pdb_root, args.pdb_version, 'pdb', '*.gz'))
	with Pool(N_CPUS) as pool:
		_ = list(tqdm.tqdm(pool.imap(find_engineered_mutation_file, pdb_files), total=len(pdb_files)))

# parse mutation information
## 1. mutation site, wt type, mutant type
# (BIOPANDAS RESIDUE NUMBER VERSION) SEQADV  1ABC A    10  ASP PDB    1ABC A   10  TYR ENGINEERED MUTATION


def parse_seqadv_line(line):
	"""
	Parse a PDB SEQADV record using official fixed-column positions.
	Returned field order is kept compatible with the current df_seqadv columns:
	['ID', 'MUT', 'CHAIN', 'POS', 'iCODE', 'DBREF', 'DBREFID', 'WT', 'UNK_POS']
	"""
	line_lower = line.lower()
	if 'engineered mutation' not in line_lower:
		return None
	if line.startswith('SEQADV'):
		pdb_id = line[7:11].strip().lower()        # columns 8-11
		mut_res3 = line[12:15].strip()            # columns 13-15, PDB residue
		chain_id = line[16:17].strip()            # column 17
		pdb_resnum = line[18:22].strip()          # columns 19-22
		icode = line[22:23].strip()               # column 23
		database = line[24:28].strip()            # columns 25-28
		db_accession = line[29:38].strip()        # columns 30-38
		wt_res3 = line[39:42].strip()             # columns 40-42, database residue
		db_seqnum = line[43:48].strip()           # columns 44-48
		try:
			if not pdb_id or not mut_res3 or not chain_id or not pdb_resnum or not wt_res3 or int(pdb_resnum) <= 0:
				return None
		except ValueError:
			return None
		return [pdb_id, mut_res3, chain_id, pdb_resnum, icode, database, db_accession, wt_res3, db_seqnum]
	elif not line.startswith('_'):
		try:
			tokens = shlex.split(line)
		except ValueError:
			tokens = line.split()
		if len(tokens) >= 12:
			pdb_id = tokens[1].strip().lower()
			mut_res3 = tokens[2].strip()
			chain_id = tokens[3].strip()
			pdb_resnum = tokens[11].strip()
			if pdb_resnum in ('?', '.', 'None'):
				pdb_resnum = tokens[4].strip()
			icode = tokens[5].strip()
			database = tokens[6].strip()
			db_accession = tokens[7].strip()
			wt_res3 = tokens[8].strip()
			db_seqnum = tokens[9].strip()

			icode = "" if icode in ('?', '.', 'None') else icode
			database = "" if database in ('?', '.', 'None') else database
			db_accession = "" if db_accession in ('?', '.', 'None') else db_accession
			db_seqnum = "" if db_seqnum in ('?', '.', 'None') else db_seqnum

			try:
				if not pdb_id or not mut_res3 or not chain_id or not pdb_resnum or not wt_res3 or int(pdb_resnum) <= 0:
					return None
			except ValueError:
				return None
			return [pdb_id, mut_res3, chain_id, pdb_resnum, icode, database, db_accession, wt_res3, db_seqnum]
	return None

def parse_seqadv_records(pdb_file):
	seqadv_lines = []
	with gzip.open(pdb_file, 'rt') as gz_file:
		for line in gz_file:
			parsed = parse_seqadv_line(line)
			if parsed is not None:
				seqadv_lines.append(parsed)
	return seqadv_lines

if args.re_group_seqadv:
	pdb_dir = os.path.join(args.pdb_root, args.pdb_version, 'mutated_data')
	pdb_files = glob.glob(os.path.join(pdb_dir, '*.gz'))

	with Pool(N_CPUS) as pool:
		seqadv_lines = list(tqdm.tqdm(pool.imap(parse_seqadv_records, pdb_files), total=len(pdb_files)))

	seqadv_lines = [item for sublist in seqadv_lines for item in sublist]
	df_seqadv = pd.DataFrame(seqadv_lines, columns=['ID', 'MUT', 'CHAIN', 'POS', 'iCODE', 'DBREF', 'DBREFID', 'WT', 'UNK_POS'])
	df_seqadv['ID'] = df_seqadv['ID'].str.lower()
	df_seqadv = df_seqadv[df_seqadv['MUT'].isin(common_aas) & df_seqadv['WT'].isin(common_aas)].copy()

	## 2. set single-site mutation or multisite-mutation
	df_seqadv = df_seqadv.drop_duplicates(subset=['ID', 'CHAIN', 'POS', 'iCODE', 'WT', 'MUT']).copy()

	if not args.multi_site:
		n_unique_mut = df_seqadv.groupby(['ID', 'CHAIN']).size().rename('n_mut').reset_index()
		single_chains = n_unique_mut[n_unique_mut['n_mut'] == 1][['ID', 'CHAIN']]
		df_seqadv = df_seqadv.merge(single_chains, on=['ID', 'CHAIN'], how='inner')
	corrected_df_seqadv = []
	grouped_seqadv = list(df_seqadv.groupby(['ID', 'CHAIN']))
	with open(os.path.join(args.output_dir, 'grouped_seqadv.pkl'), 'wb') as f:
		pickle.dump(grouped_seqadv, f)
else:
	with open(os.path.join(args.output_dir, 'grouped_seqadv.pkl'), 'rb') as f:
		grouped_seqadv = pickle.load(f)

## 3. modify mutation position
## 4. remove mutations that are in unmodeled regions
## 5. 5z8w, 7why, 7a7y, 7a7w: the pdb_id in the 'SEQADV' line is inconsistent with the actual pdb_id
## 6. 7ttb: there are 'APHE' and 'BTYR' at the same ATOM position
# remove positions that are in unmodeled regions
def process_mutation_group(group_data):
	(pdb_id, chain_id), group = group_data

	if pdb_id in EXCLUDE_PDBS + LARGE_PDBS:
		# continue
		log_skip("manual_exclusion_known_annotation_issue", pdb_id=pdb_id, chain_id=chain_id)
		return []
	try:
		pdb, has_multi_model = load_structure_file(pdb_id, args)
	except FunctionTimedOut:
		log_skip("file_read_timeout", pdb_id=pdb_id, chain_id=chain_id)
		return []
	except Exception as e:
		log_skip("file_read_error", pdb_id=pdb_id, chain_id=chain_id, extra=str(e))
		return []
	if has_multi_model:
		# continue
		log_skip("multi_model_mutant_structure", pdb_id=pdb_id, chain_id=chain_id)
		return []
	df = filter_residues(pdb.df['ATOM'], chain_id=chain_id)
	seq, id_mapping = get_seq_and_mapping(df)
	if seq is None or id_mapping is None:
		log_skip("empty_or_unmapped_mutant_chain", pdb_id=pdb_id, chain_id=chain_id)
		return []

	if not set(seq).issubset(AA1_SET):
		log_skip("nonstandard_amino_acid_in_mutant", pdb_id=pdb_id, chain_id=chain_id)
		return []
	
	if len(seq) < MIN_LEN or len(seq) > MAX_LEN:
		# continue
		log_skip(
			"mutant_sequence_length_out_of_range",
			pdb_id=pdb_id,
			chain_id=chain_id,
			extra=f"length={len(seq)}",
		)
		return []
	valid_records = []
	for line in group.itertuples(index=False, name=None):
		pos = int(line[3])
		mut_type = line[1]
		icode = line[4]

		if (pos, icode) not in id_mapping:
			# continue
			log_skip(
				"mutation_position_unmodeled",
				pdb_id=pdb_id,
				chain_id=chain_id,
				extra=line,
			)
			continue
		correct_index = id_mapping[(pos, icode)]
		expected_mut_aa = three_to_one(mut_type)
		observed_aa = seq[correct_index]

		if observed_aa != expected_mut_aa:
			log_skip(
				"mutation_residue_mismatch",
				pdb_id=pdb_id,
				chain_id=chain_id,
				extra=line,
			)
			continue
		updated_line = list(line)
		updated_line[3] = correct_index
		updated_line.append(pos)
		valid_records.append(updated_line)
	return valid_records
	# corrected_df_seqadv.append(line)

if args.re_mutations:
	results = []
	with Pool(N_CPUS) as pool:
		for res in tqdm.tqdm(pool.imap_unordered(process_mutation_group, grouped_seqadv, chunksize=1), total=len(grouped_seqadv)):
			results.append(res)

	corrected_df_seqadv = [item for sublist in results for item in sublist]

	df = pd.DataFrame(corrected_df_seqadv,  columns=['ID', 'MUT', 'CHAIN', 'POS', 'iCODE', 'DBREF', 'DBREFID', 'WT', 'UNK_POS', 'MUT_RES_NUM'])
	df = df.drop_duplicates()
	if args.multi_site:
		df.to_csv(os.path.join(args.output_dir, 'mutations_ms.csv'), index=False)
	else:
		df.to_csv(os.path.join(args.output_dir, 'mutations.csv'), index=False)


# generate wild-type sequence
# using mutation data, replace the mutation site with the original amino acid
# (BIOPANDAS VERSION) generate mutated sequences fasta file
### CHANGES: 
### 1. Use Biopandas to parse PDB files
### 2. Get sequence method: only return residues with CA atoms
### 3. Filter out sequences shorter than 32
### 4. Check if mutation position, mutation type and wild type match
mutation_file = os.path.join(args.output_dir, 'mutations.csv') if not args.multi_site else os.path.join(args.output_dir, 'mutations_ms.csv')
muts = pd.read_csv(mutation_file, dtype=str, keep_default_na=False)
mutated_chains = set(muts['ID'].str.lower() + '_' + muts['CHAIN'].astype(str))

def build_wt_mut_fasta_record(row):
	pdb_id = row['ID']
	mut_type = row['MUT']
	chain_id = row['CHAIN']
	pos = row['POS']
	wt_type = row['WT']
	mut_df = read_pdb(pdb_id, chain_id)
	mut_seq, _ = get_seq_and_mapping(mut_df)
	if mut_seq is None:
		print(f"Failed to get sequence for {pdb_id} {chain_id}")
		return None
	try:
		assert mut_seq[pos] == three_to_one(mut_type), f"{pdb_id} {chain_id} Mutation type {mut_type} does not match the sequence {mut_seq[pos]} at position {pos}"
	except AssertionError as e:
		print(e)
		return None
	wt_seq = list(mut_seq)
	wt_seq[pos] = three_to_one(wt_type)
	wt_seq = "".join(wt_seq)
	return f">{pdb_id}_{chain_id}\n{mut_seq}\n", f">{pdb_id}_{chain_id}\n{wt_seq}\n"

def build_wt_mut_fasta_record_group(item):
	(pdb_id, chain_id), group = item
	formatted = format_mutation_set(group)
	if formatted is None:
		return None
	mut_df = read_pdb(pdb_id, chain_id)
	mut_seq, _ = get_seq_and_mapping(mut_df)
	if mut_seq is None:
		print(f"Failed to get sequence for {pdb_id} {chain_id}")
		return None
	for pos, mut_type in zip(formatted['positions'], formatted['mut_types_1']):
		if pos >= len(mut_seq) or mut_seq[pos] != mut_type:
			log_skip(
				"multi_site_mutant_sequence_mismatch",
				pdb_id=pdb_id,
				chain_id=chain_id,
				extra=f"idx0={pos}; expected={mut_type}; observed={mut_seq[pos] if pos < len(mut_seq) else 'OUT_OF_RANGE'}",
			)
			return None
	wt_seq = reconstruct_wt_sequence(mut_seq, formatted['positions'], formatted['wt_types_1'])
	return f">{pdb_id}_{chain_id}\n{mut_seq}\n", f">{pdb_id}_{chain_id}\n{wt_seq}\n"

if args.re_seqfasta:
	if args.multi_site:
		grouped = list(muts.groupby(['ID', 'CHAIN']))
		results = list(progress_map(build_wt_mut_fasta_record_group, grouped, n_cpu=N_CPUS))
	else:
		results = list(progress_map(build_wt_mut_fasta_record, muts.to_dict('records'), n_cpu=N_CPUS))
	mut_results = [result[0] for result in results if result is not None]
	wt_results = [result[1] for result in results if result is not None]
	write_fasta(wt_results, os.path.join(args.output_dir, 'wt_seqs.fasta'))
	write_fasta(mut_results, os.path.join(args.output_dir, 'mut_seqs.fasta'))

# search wild-type pdb with 100% sequence identity
# 1. extract the sequences of all proteins in PDB, and generate a fasta file
# 2. use mmseqs2 to create PDB DataBase (wholepdb_db)
# 3. searching 100% sequence identity using wt_seqs.fasta and wholepdb_db
# 4. parse searching output
# (BIOPANDAS VERSION) extract the sequences of all proteins in PDB, and generate a fasta file
def extract_chain_sequences(filepath):
	"""
	Parse the PDB file and extract amino acid sequences from all chains.
	"""
	# Extract PDB ID from filename
	pdb_id = os.path.basename(filepath).split('.')[0][-4:].lower()
	try:
		# Read PDB file - only once per file
		pdb, has_multi_model = load_structure_file(pdb_id, args)
	except (OSError, gzip.BadGzipFile, ValueError) as e:
		logger.exception(
			"read_reference_pdb_failed | pdb_id=%s | file=%s | error_type=%s",
			pdb_id,
			filepath,
			type(e).__name__,
		)
		return []
	if has_multi_model:
		log_skip("multi_model_reference_structure", pdb_id=pdb_id, extra=filepath)
		return []
	try:
		atom_df = pdb.df['ATOM']
		atom_df = filter_residues(atom_df)
		
		# Get unique chains
		chains = atom_df['chain_id'].unique()
		
		results = []
		for chain_id in chains:
			chain_df = atom_df[atom_df['chain_id'] == chain_id]
			if chain_df.empty:
				continue
			sequence, _ = get_seq_and_mapping(chain_df)
			if not all(res in common_aas_1 for res in sequence):
				continue
			# Filter by sequence length
			if 24 < len(sequence) < 2100:
				results.append(f">{pdb_id}_{chain_id}\n{sequence}\n")
			else:
				log_skip(
					"reference_sequence_length_out_of_range",
					pdb_id=pdb_id,
					chain_id=chain_id,
					extra=f"length={len(sequence)}",
				)		
		return results
	except (KeyError, TypeError, ValueError) as e:
		logger.exception(
			"extract_reference_sequence_failed | pdb_id=%s | file=%s | error_type=%s",
			pdb_id,
			filepath,
			type(e).__name__,
		)
		return []

def process_batch(file_batch):
	"""Process a batch of files and return combined results"""
	batch_results = []
	for filepath in file_batch:
		try:
			results = extract_chain_sequences(filepath)
			if results:
				batch_results.extend(results)
		except Exception as e:
			logger.exception(
				"unexpected_batch_processing_error | file=%s | error_type=%s",
				filepath,
				type(e).__name__,
			)
	return batch_results

if args.re_wholefasta:
	# Define directories
	output_file = os.path.join(args.output_dir, 'wholepdb.fasta')

	# Get all PDB files
	all_pdb_files = sorted(glob.glob(os.path.join(args.pdb_root, args.pdb_version, 'pdb', '*.gz')))
	# mut_pdb_files = set(glob.glob(os.path.join(args.pdb_root, args.pdb_version, 'mutated_data', 'pdb*.gz')))
	# pdb_files = list(all_pdb_files - mut_pdb_files)
	if not all_pdb_files:
		raise FileNotFoundError(f"No structure files found under {args.pdb_root}/{args.pdb_version}/pdb")
	# For debugging
	# pdb_files = random.sample(pdb_files, min(1000, len(pdb_files)))

	# Determine number of processes and batch size
	batch_size = max(1, min(50, len(all_pdb_files) // max(1, N_CPUS * 2)))  # Optimized batch size

	# Create batches of files
	file_batches = [all_pdb_files[i:i + batch_size] for i in range(0, len(all_pdb_files), batch_size)]

	# Process files in parallel with progress tracking
	print(f"Processing {len(all_pdb_files)} PDB files with {N_CPUS} processes in {len(file_batches)} batches")
	with Pool(processes=N_CPUS) as pool:
		# Process with imap to show progress
		all_results = []
		completed = 0
		total = len(file_batches)
		
		for batch_result in pool.imap_unordered(process_batch, file_batches):
			completed += 1
			if batch_result:
				all_results.extend(batch_result)
			# Print progress
			if completed % 10 == 0 or completed == total:
				print(f"Progress: {completed}/{total} batches ({completed*100//total}%) - Sequences found: {len(all_results)}")
	# Write results to file
	print(f"Writing {len(all_results)} sequences to {output_file}")
	write_fasta(all_results, output_file)

if args.re_genmatchesm8:
	# searching 100% sequence identity using wt_seqs.fasta and wholepdb_db
	query_fasta = os.path.join(args.output_dir, "wt_seqs.fasta")
	target_fasta = os.path.join(args.output_dir, "wholepdb.fasta")
	output_file = os.path.join(args.output_dir, "matches.m8")

	tmp_dir = f"/tmp/mmseqs_tmp_{os.getpid()}"
	mmseqs_search_cmd = [
		"mmseqs", "easy-search",
		query_fasta,
		target_fasta,
		output_file,
		tmp_dir,
		"--min-seq-id", "1.0",
		"-c", "1.0",
		"--cov-mode", "0",
		"--format-output", "query,target,fident,alnlen,qlen,tlen,bits",
	]
	try:
		run_cmd(mmseqs_search_cmd, check=True)
	except subprocess.CalledProcessError as e:
		logger.exception(
			"mmseqs_easy_search_failed_first_attempt | query=%s | target=%s | output=%s | returncode=%s",
			query_fasta,
			target_fasta,
			output_file,
			e.returncode,
		)

		if os.path.isdir(tmp_dir):
			run_cmd(["chmod", "-R", "u+rwx", tmp_dir], check=False)

		run_cmd(mmseqs_search_cmd, check=True)

if args.re_gen_matching_dict:
	# parse the matches.m8 file
	seqadv_list = os.listdir(os.path.join(args.pdb_root, args.pdb_version, 'mutated_data'))
	seqadv_ids = [seqadv.split('.')[0][-4:] for seqadv in seqadv_list]

	result_file = os.path.join(args.output_dir, 'matches.m8')
	matching_dict = defaultdict(list)

	with open(result_file) as f:
		total_lines = sum(1 for _ in f)

	with open(result_file) as f:
		for line in tqdm.tqdm(f, total=total_lines):
			parts = line.strip().split('\t')
			query_id, target_id, identity = parts[0], parts[1], float(parts[2])
			alnlen = int(parts[3])
			qlen = int(parts[4])
			tlen = int(parts[5])
			if identity >= 0.999999 and alnlen == qlen == tlen and target_id not in mutated_chains:
				matching_dict[query_id].append(target_id)

	output_file = os.path.join(args.output_dir, 'matching_dict.json')
	with open(output_file, 'w') as json_file:
		json.dump(matching_dict, json_file)

	with open(os.path.join(args.output_dir, 'matching_dict.json'), 'r') as json_file:
		matching_dict = json.load(json_file)
	print("Total number of matches: ", len(matching_dict))

if args.re_internalcsv:
	mut_seqs = {}
	whole_seqs = {}

	with open(os.path.join(args.output_dir, 'mut_seqs.fasta')) as mut_fasta:
		for record in SeqIO.parse(mut_fasta, 'fasta'):
			mut_seqs[record.id] = str(record.seq)

	with open(os.path.join(args.output_dir, 'wholepdb.fasta')) as whole_fasta:
		for record in SeqIO.parse(whole_fasta, 'fasta'):
			whole_seqs[record.id] = str(record.seq)

	muts_info = pd.read_csv(mutation_file, dtype=str, keep_default_na=False)

	def process_match_single_site(items):
		mut, wts = items
		try:
			mut_id, mut_chain_id = split_pdb_chain_id(mut)
		except ValueError as e:
			logger.warning("skip_invalid_mut_chain_id | mut_chain=%s | error=%s", mut, str(e))
			return None
		
		mut_info = muts_info[
			(muts_info['ID'] == mut_id) & 
			(muts_info['CHAIN'] == mut_chain_id)
		]

		if len(mut_info) > 1 or mut_info.empty:
			logger.warning(
				"skip_ambiguous_mut_info | mut_chain=%s | num_records=%s",
				mut,
				len(mut_info)
			)
			return None
		
		wt_type = three_to_one(mut_info['WT'].values[0])
		mut_type = three_to_one(mut_info['MUT'].values[0])
		pos = int(mut_info['POS'].values[0])
		mut_pos_pdb_number = int(mut_info['MUT_RES_NUM'].values[0])
		release_date = get_release_date(mut_id)

		try:
			mut_seq = mut_seqs[mut]
		except KeyError:
			logger.warning("skip_match_missing_mutant_sequence | mut_chain=%s", mut)
			return None
		
		if pos >= len(mut_seq) or mut_seq[pos] != mut_type:
			logger.info(
				"reject_mutant_sequence_validation | mut_chain=%s | pos=%s | expected=%s | observed=%s",
				mut,
				pos,
				mut_type,
				mut_seq[pos] if pos < len(mut_seq) else 'OUT_OF_RANGE',
			)
			return None

		results = []
		for wt in wts:
			try:
				wt_id, wt_chain_id = split_pdb_chain_id(wt)
			except ValueError as e:
				logger.warning(
					"skip_invalid_wt_chain_id | mut_chain=%s | wt_chain=%s | error=%s",
					mut,
					wt,
					str(e)
				)
				continue
			try:
				wt_seq = whole_seqs[wt]
			except KeyError:
				logger.warning(
					"skip_match_missing_wt_sequence | mut_chain=%s | wt_chain=%s",
					mut,
					wt,
				)
				continue

			differences = hamming_distance(mut_seq, wt_seq)
			if (
				differences == 1 and
				len(wt_seq) == len(mut_seq) and
				mut_seq[pos] == mut_type and
				wt_seq[pos] == wt_type
			):
				sample_id = (
					f"{wt_id}_{wt_chain_id}"
					f"__{mut_id}_{mut_chain_id}"
					f"__{wt_type}{pos}{mut_type}"
				)
				results.append([
					sample_id,
					wt_id,
					wt_chain_id,
					mut_id,
					mut_chain_id,
					pos,
					mut_pos_pdb_number,
					wt_type,
					mut_type,
					wt_seq,
					mut_seq,
					release_date,
					f"{mut_id}_{mut_chain_id}",
					f"{wt_id}_{wt_chain_id}",
				])
			else:
				logger.info(
					"reject_wt_match | reason=not_single_valid_difference | mut_chain=%s | wt_chain=%s | differences=%s | pos=%s",
					mut,
					wt,
					differences,
					pos,
				)
		if len(results) >= 1:
			return results
		return None

	def process_match_multi_site(items):
		mut, wts = items
		try:
			mut_id, mut_chain_id = split_pdb_chain_id(mut)
		except ValueError as e:
			logger.warning("skip_invalid_mut_chain_id | mut_chain=%s | error=%s", mut, str(e))
			return None
		mut_info = muts_info[(muts_info['ID'] == mut_id) & (muts_info['CHAIN'] == mut_chain_id)]
		if mut_info.empty:
			return None
		formatted = format_mutation_set(mut_info)
		if formatted is None:
			return None

		expected_diff_positions = set(formatted['positions'])
		try:
			mut_seq = mut_seqs[mut]
		except KeyError:
			logger.warning(
				"skip_match_missing_mutant_sequence | mut_chain=%s",
				mut,
			)
			return None

		for pos, mut_type in zip(formatted['positions'], formatted['mut_types_1']):
			if pos >= len(mut_seq) or mut_seq[pos] != mut_type:
				log_skip(
					"multi_site_mutant_sequence_mismatch",
					pdb_id=mut_id,
					chain_id=mut_chain_id,
					extra=f"idx0={pos}; expected={mut_type}; observed={mut_seq[pos] if pos < len(mut_seq) else 'OUT_OF_RANGE'}",
				)
				return None

		results = []
		for wt in wts:
			try:
				wt_seq = whole_seqs[wt]
			except KeyError:
				logger.warning(
					"skip_match_missing_wt_sequence | mut_chain=%s | wt_chain=%s",
					mut,
					wt,
				)
				continue

			ok, reason, actual_diff_positions = validate_expected_difference_positions(mut_seq, wt_seq, expected_diff_positions)
			if not ok:
				logger.info(
					"reject_wt_match | reason=%s | mut_chain=%s | wt_chain=%s | expected_positions=%s | actual_positions=%s",
					reason,
					mut,
					wt,
					sorted(expected_diff_positions),
					None if actual_diff_positions is None else sorted(actual_diff_positions),
				)
				continue

			wrong_wt_residue = False
			for pos, wt_type, mut_type in zip(formatted['positions'], formatted['wt_types_1'], formatted['mut_types_1']):
				if wt_seq[pos] != wt_type:
					logger.info(
						"reject_wt_match | reason=multi_site_candidate_wt_wrong_wt_residue | mut_chain=%s | wt_chain=%s | pos=%s | expected_wt=%s | observed_wt=%s | mut_residue=%s",
						mut,
						wt,
						pos,
						wt_type,
						wt_seq[pos],
						mut_type,
					)
					wrong_wt_residue = True
					break
			if wrong_wt_residue:
				continue

			results.append([
				f"{mut_id}_{mut_chain_id}",
				formatted['mutation_set_idx0'],
				formatted['mutation_set_pdb'],
				wt,
				formatted['num_mutations'],
				formatted['mut_pos_idx0'],
				formatted['mut_resnums_pdb'],
				formatted['wt_aas'],
				formatted['mut_aas'],
			])

		if len(results) >= 1:
			return results
		return None

	def process_match(items):
		if args.multi_site:
			return process_match_multi_site(items)
		return process_match_single_site(items)

	results = list(progress_map(process_match, matching_dict.items(), n_cpu=N_CPUS))

	out_lines = [result for result in results if result is not None]
	out_lines = [item for sublist in out_lines for item in sublist]
	if args.multi_site:
		df = pd.DataFrame(
			out_lines,
			columns=[
				'mut_chain',
				'mutation_set_idx0',
				'mutation_set_pdb',
				'wt_chain',
				'num_mutations',
				'mut_pos_idx0',
				'mut_resnums_pdb',
				'wt_aas',
				'mut_aas'
			]
		)
		df.to_csv(pair_internal_csv, index=False)
	else:
		df = pd.DataFrame(out_lines, columns=INTERNAL_PAIR_COLUMNS)
		df = df.drop_duplicates(subset=['sample_id']).copy()  # Remove exact duplicate WT-mutant-chain pairs
		
		# Hard validation before clustering.
		for row in df.itertuples(index=False):
			if hamming_distance(row.wt_sequence, row.mut_sequence) != 1:
				raise ValueError(f"Invalid one-mutation pair before clustering: {row.sample_id}")
			pos = int(row.mut_pos_seq_index)
			if row.wt_sequence[pos] != row.wt_aa_type:
				raise ValueError(f"WT residue mismatch before clustering: {row.sample_id}")
			if row.mut_sequence[pos] != row.mut_aa_type:
				raise ValueError(f"Mutant residue mismatch before clustering: {row.sample_id}")

		df.to_csv(pair_internal_csv, index=False)

# 30% sequence identity clustering
# generate mutated sequences fasta file
muts_info = pd.read_csv(pair_internal_csv, dtype=str, keep_default_na=False)

muts_info = muts_info.drop_duplicates(subset=['mut_chain'])
mut_seqs = {}
with open(os.path.join(args.output_dir, "mut_seqs.fasta")) as mut_fasta:
	for record in SeqIO.parse(mut_fasta, 'fasta'):
		mut_seqs[record.id] = str(record.seq)

def build_cluster_fasta_record(mut_info):
	pdb_id, chain_id = split_pdb_chain_id(mut_info)
	mt_seq = mut_seqs.get(mut_info)

	if mt_seq is None:
		logger.warning(
			"skip_cluster_sequence_missing | mut_chain=%s | pdb_id=%s | chain_id=%s",
			mut_info,
			pdb_id,
			chain_id,
		)
		return None

	return f">{pdb_id}_{chain_id}\n{mt_seq}\n"

results = list(progress_map(build_cluster_fasta_record, muts_info['mut_chain'], n_cpu=N_CPUS))
results = [result for result in results if result is not None]

write_fasta(results, os.path.join(args.output_dir, "mut_seqs_v2.fasta"))

input_fasta = os.path.join(args.output_dir, "mut_seqs_v2.fasta")

cluster_out = os.path.join(args.output_dir, "mut_seqs_cluster")
tmp_dir = f"/tmp/mmseqs_cluster_tmp_{os.getpid()}"
mmseqs_cluster_cmd = [
	"mmseqs", "easy-cluster",
	input_fasta,
	cluster_out,
	tmp_dir,
	"--min-seq-id", "0.3",
	"-c", "0.8",
	"--cov-mode", "0",
]

try:
	run_cmd(mmseqs_cluster_cmd)
except subprocess.CalledProcessError as e:
	logger.exception(
		"mmseqs_easy_cluster_failed_first_attempt | input_fasta=%s | cluster_out=%s | returncode=%s",
		input_fasta,
		cluster_out,
		e.returncode,
	)

	if os.path.isdir(tmp_dir):
		run_cmd(["chmod", "-R", "u+rwx", tmp_dir], check=False)

	run_cmd(mmseqs_cluster_cmd)

# save the cluster results to pickle dict
cluster_tsv = os.path.join(args.output_dir, 'mut_seqs_cluster_cluster.tsv')
cluster_dict = defaultdict(list)
cluster_df = pd.read_csv(cluster_tsv, sep='\t', header=None)
for c, m in cluster_df.itertuples(index=False):
	cluster_dict[c].append(m)

with open(cluster_pkl, 'wb') as f:
	pickle.dump(cluster_dict, f)

# Build member -> representative-cluster mapping
member_to_cluster = {}
for representative, members in cluster_dict.items():
	for member in members:
		member_to_cluster[member] = representative

if args.multi_site:
	raise NotImplementedError("multi_site mode is not finalized. Do not write a final dataset from internal columns.")
else:
	pair_df = pd.read_csv(pair_internal_csv, dtype=str, keep_default_na=False)
	bad_wt_chain = pair_df[pair_df['wt_chain_id'].astype(str).str.len() == 0]
	if not bad_wt_chain.empty:
		raise ValueError(
        	"Empty wt_chain_id found before final validation. "
			f"Examples: {bad_wt_chain['wt_chain'].head(10).tolist()}"
		)
	pair_df["cluster_id_30"] = pair_df['mut_chain'].map(member_to_cluster)
	missing_cluster = pair_df[pair_df['cluster_id_30'].isna()]
	if not missing_cluster.empty:
		raise ValueError(
			"Some mutant chains are missing cluster_id_30. "
			f"Examples: {missing_cluster['mut_chain'].head(10).tolist()}"
		)
	final_df = pair_df[REQUIRED_DATASET_COLUMNS].copy()

	# Enforce types
	final_df['mut_pos_seq_index'] = final_df['mut_pos_seq_index'].astype(int)
	final_df['mut_pos_pdb_number'] = final_df['mut_pos_pdb_number'].astype(int)
	final_df['wt_pdb_id'] = final_df['wt_pdb_id'].astype(str).str.lower()
	final_df['mut_pdb_id'] = final_df['mut_pdb_id'].astype(str).str.lower()

	validate_required_dataset_df(final_df)
	final_df.to_csv(pair_csv, index=False)
	logger.info(
		"final_dataset_written | path=%s | rows=%s | columns=%s",
		pair_csv,
		len(final_df),
		list(final_df.columns)
	)
	print(f"Final dataset written to: {pair_csv}")
	print(f"Rows: {len(final_df)}")
	print(f"Columns: {list(final_df.columns)}")