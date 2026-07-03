def load_or_compute_dssp_annotation(
    pdb_id: str,
    chain_id: str,
    residue_number: int,
    dssp_dir: Path,
    pdb_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:

    pdb_path = find_pdb_file(str(pdb_dir), pdb_id)

    tmpdir, tmp_path = _decompress_to_temp(pdb_path)
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(tmp_path))
        model = next(structure.get_models())

        dssp_result = None
        dssp_reason = None
        for exe_name in ["mkdssp", "dssp"]:
            try:
                from Bio.PDB.DSSP import DSSP

                dssp_result = DSSP(model, str(tmp_path), dssp=exe_name)
                dssp_reason = exe_name
                break
            except Exception as exc:
                dssp_reason = str(exc)

        if dssp_result is not None:
            chain_cache = {}
            for key in dssp_result.keys():
                chain = key[0]
                resseq = int(key[1][1])
                icode = str(key[1][2]).strip()
                values = dssp_result[key]

                rsa_value = values[3]
                try:
                    rsa = float(rsa_value)
                except (TypeError, ValueError):
                    rsa = float("nan")

                chain_cache.setdefault(chain, {})
                chain_cache[chain][str(resseq)] = {
                    "rsa": rsa,
                    "secondary_structure_code": values[2] if values[2] else "unknown",
                    "annotation_status": f"dssp:{dssp_reason}",
                    "icode": icode,
                }
            if chain_id in chain_cache:
                _save_json(cache_path, chain_cache[chain_id])
                if cache_key in chain_cache[chain_id]:
                    return chain_cache[chain_id][cache_key]

        try:
            freesasa_cache = _compute_freesasa_annotation(structure, chain_id)
            result = freesasa_cache.get((int(residue_number), ""))
            if result is not None:
                response = {
                    "rsa": result["rsa"],
                    "secondary_structure_code": "unknown",
                    "annotation_status": "freesasa_rsa_only",
                }
                cache[cache_key] = response
                _save_json(cache_path, cache)
                _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} DSSP unavailable; used FreeSASA fallback")
                return response
        except Exception as exc:
            _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} FreeSASA unavailable: {exc}")

        _warn(warnings, f"{pdb_id}:{chain_id}:{residue_number} annotation unknown; DSSP unavailable: {dssp_reason}")
        return {"rsa": float("nan"), "secondary_structure_code": "unknown", "annotation_status": f"unknown:{dssp_reason}"}
    finally:
        tmpdir.cleanup()