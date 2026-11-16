#!/usr/bin/env python3
"""Batch-map top-contribution 15-mers to InterPro-annotated DNA-binding domains.

Install: pip install pandas requests
Run: python interpro_dbd_batch.py --input highest_15mer_per_protein.csv
For stricter coordinate QC: add --verify-sequence (extra UniProt requests).

Coordinates are 1-based, both ends inclusive. No DBD boundaries are extended.
"""

import argparse
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://www.ebi.ac.uk/interpro/api"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{}.fasta"
LOCAL = threading.local()

# DBD classification based on functionally recognizable domain names.
# 3: explicit DNA-binding description; 2: established DBD families/motifs.
# Keywords are a screening heuristic, not experimental confirmation.
EXPLICIT_DNA = re.compile(
    r"(?:\bDNA[\s_/-]*(?:binding|bind(?:ing)?|bd|recognition|contact)"
    r"|\bDNA[-\s_]*associated[-\s_]*HTH)", re.I)
FAMILIES = [
    r"\bbHLH\b", r"\bHLH[_\s-]*(?:DNA|domain|dom)\b",
    r"helix[-\s_]*loop[-\s_]*helix",
    r"\bbZIP\b", r"basic[-\s_]*leucine[-\s_]*zipper",
    r"\bhomeo(?:domain|box)\b", r"\bHTH\b",
    r"helix[-\s_]*turn[-\s_]*helix",
    r"\b(?:zf|znf)[-\s_]*C2H2\b", r"\bC2H2[-\s_]*(?:type|zinc|finger)",
    r"zinc[-\s_]*finger[-\s_]*C2H2",
    r"\bGATA[-\s_]*(?:zinc|type|finger|domain)",
    r"\bforkhead\b", r"\bwinged[-\s_]*helix\b",
    r"\bETS[-\s_]*(?:domain|DNA|family)",
    r"\bMYB[-\s_]*(?:DNA|domain|repeat)",
    r"\bHMG[-\s_]*(?:box|DNA)", r"\bMADS[-\s_]*box\b",
    r"\bRunt[-\s_]*domain\b", r"\bARID\b",
    r"\b(?:AP2|ERF)[-\s_]*(?:domain|DNA)",
    r"\bWRKY\b", r"\bNAC[-\s_]*domain\b",
    r"\bB3[-\s_]*(?:DNA|domain)", r"\bT[-\s_]*box\b",
    r"\bPOU[-\s_]*(?:domain|specific|homeo)",
    r"\bpaired[-\s_]*box\b", r"\bSAND[-\s_]*domain\b",
    r"\b(?:DM|HSF|STAT|NF[-\s_]*Y)[-\s_]*DNA",
    r"\b(?:nuclear[-\s_]*receptor|steroid[-\s_]*receptor)[-\s_]*(?:DNA|zinc)",
    r"\b(?:C4[-\s_]*zinc[-\s_]*finger|zinc[-\s_]*cluster)",
    r"\bRel[-\s_]*homology[-\s_]*domain\b",
    r"\bTEA[-\s_]*domain\b",
]
FAMILY_RE = re.compile("|".join("(?:" + p + ")" for p in FAMILIES), re.I)
# Explicit RNA/DNA dual annotations alone aren't specific evidence of DBD.
AMBIGUOUS = re.compile(r"RNA[\s_/-]*(?:and|/|&)[\s_/-]*DNA|DNA[\s_/-]*(?:and|/|&)[\s_/-]*RNA", re.I)


def session():
    if not hasattr(LOCAL, "session"):
        s = requests.Session()
        retry = Retry(total=4, connect=4, read=4, backoff_factor=1.1,
                      status_forcelist=(429, 500, 502, 503, 504),
                      respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        s.mount("https://", adapter)
        s.headers.update({"Accept": "application/json",
                          "User-Agent": "InterPro-15mer-mapping/1.0 (research script)"})
        LOCAL.session = s
    return LOCAL.session


def get_text(url):
    resp = session().get(url, timeout=(15, 55))
    resp.raise_for_status()
    return resp.text


def cached_entries(pid, db, cache_dir):
    folder = cache_dir / db
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (pid + ".json")
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    url = f"{API}/entry/{db}/protein/uniprot/{quote(pid)}/?page_size=200"
    results, seen = [], set()
    while url:
        if url in seen or len(seen) >= 50:
            raise RuntimeError("Pagination loop or >50 pages")
        seen.add(url)
        response = session().get(url, timeout=(15, 55))
        if response.status_code == 204:
            break
        response.raise_for_status()
        page = response.json()
        if not isinstance(page.get("results"), list):
            raise ValueError("Unexpected InterPro response")
        results.extend(page["results"])
        url = page.get("next")
    target.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return results


def check_domain_name(name, short_name):
    joined = " | ".join(x for x in (name, short_name) if x).replace("_", " ")
    if AMBIGUOUS.search(joined) and not FAMILY_RE.search(joined):
        return 0, ""
    if EXPLICIT_DNA.search(joined):
        return 3, "explicit_DNA_binding"
    if FAMILY_RE.search(joined):
        return 2, "known_DBD_family"
    return 0, ""


def extract_hits(items, db, pid):
    output = []
    for ent in items:
        meta = ent.get("metadata") or {}
        name = str(meta.get("name") or "")
        short = str(meta.get("short_name") or (ent.get("extra_fields") or {}).get("short_name") or "")
        dbtype = str(meta.get("type") or "").lower()
        score, why = check_domain_name(name, short)
        if not score:
            continue
        # Family annotations can span whole proteins, so don't use them as DBD intervals.
        eligible = dbtype in ("domain", "homologous_superfamily", "repeat", "site")
        accession = str(meta.get("accession") or "")
        for protein in ent.get("proteins", ent.get("protein_subset", [])) or []:
            length = protein.get("protein_length")
            for loc in protein.get("entry_protein_locations") or []:
                for frag in loc.get("fragments") or []:
                    start, end = frag.get("start"), frag.get("end")
                    try:
                        start, end = int(start), int(end)
                    except (TypeError, ValueError):
                        continue
                    if start < 1 or end < start:
                        continue
                    output.append({
                        "ProteinID": pid, "SourceDB": db, "InterProID": accession,
                        "DomainName": name or short, "DomainType": dbtype,
                        "DomainStart": start, "DomainEnd": end,
                        "DomainRange": f"{start}-{end}", "DomainLength": end - start + 1,
                        "DomainStrength": score, "MatchReason": why,
                        "EligibleDBD": eligible, "ProteinLengthDB": length,
                    })
    # dedup repeated fragments in same entry
    return list({(h["SourceDB"], h["InterProID"], h["DomainStart"], h["DomainEnd"]): h
                 for h in output}.values())


def verify_peptide(pid, start, end, peptide, cache_dir):
    folder = cache_dir / "fasta"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (pid + ".fasta")
    if target.exists():
        fasta = target.read_text(encoding="utf-8")
    else:
        fasta = get_text(UNIPROT_FASTA.format(quote(pid)))
        target.write_text(fasta, encoding="utf-8")
    sequence = "".join(line.strip() for line in fasta.splitlines()
                       if line and not line.startswith(">")).upper()
    if not sequence:
        return "SEQUENCE_CHECK_FAILED", None
    return ("SEQUENCE_OK" if sequence[start - 1:end] == peptide else "SEQUENCE_MISMATCH",
            len(sequence))


def process_row(row, cache_dir, verify_sequence):
    r = row.to_dict()
    pid = str(r["ProteinID"]).strip().upper()
    r["ProteinID"] = pid
    st, en = int(r["Start"]), int(r["End"])
    peptide = str(r["Peptide15mer"]).strip().upper()
    r.update({"SelectedDBD": "", "DBDRange": "", "DBDStart": "", "DBDEnd": "",
              "DBDSource": "", "DBDAccession": "", "DBDType": "",
              "OverlapAA": 0, "Overlap": "0/15", "OverlapFraction": 0.0,
              "DistanceToDBD": "", "DBDCandidates": 0, "DBDFamilyOnly": 0,
              "DomainCoordinateQC": "", "SequenceQC": "NOT_CHECKED",
              "AnnotationStatus": "", "Error": ""})
    if not re.fullmatch(r"[A-Z0-9-]+", pid):
        r.update(AnnotationStatus="INVALID_ID", Error="Unexpected UniProt accession")
        return r, []
    if st < 1 or en - st + 1 != 15 or len(peptide) != 15:
        r.update(AnnotationStatus="INVALID_WINDOW", Error="Start/End/peptide is not 15 aa")
        return r, []

    hits, errs = [], []
    found_any_entry = False
    for db in ("interpro", "pfam", "smart"):
        if db != "interpro" and any(h["EligibleDBD"] for h in hits):
            break   # only query member databases if InterPro lacks usable DBD hits
        try:
            items = cached_entries(pid, db, cache_dir)
            if items:
                found_any_entry = True
            hits.extend(extract_hits(items, db, pid))
        except Exception as exc:
            errs.append(f"{db}:{str(exc)[:150]}")

    r["DBDFamilyOnly"] = sum(not h["EligibleDBD"] for h in hits)
    good = [h for h in hits if h["EligibleDBD"]]
    r["DBDCandidates"] = len(good)

    for h in hits:
        h["PeptideStart"], h["PeptideEnd"] = st, en
        h["Peptide15mer"] = peptide
        h["OverlapAA"] = max(0, min(en, h["DomainEnd"]) - max(st, h["DomainStart"]) + 1)
        h["Overlap"] = f"{h['OverlapAA']}/15"
        h["DistanceAA"] = max(h["DomainStart"] - en, st - h["DomainEnd"], 0)
        length = h["ProteinLengthDB"]
        h["LengthMatchesCSV"] = ((str(length) == str(int(r["OriginalLength"])))
                                 if pd.notna(length) else None)

    if good:
        # First maximize overlap; then favour specific, confident, narrow annotations.
        best = sorted(good, key=lambda h: (
            -h["OverlapAA"], -h["DomainStrength"],
            {"domain": 0, "site": 1, "repeat": 2, "homologous_superfamily": 3}.get(h["DomainType"], 4),
            h["DomainLength"], h["DistanceAA"],
            {"interpro": 0, "pfam": 1, "smart": 2}.get(h["SourceDB"], 3),
            h["InterProID"]))[0]
        overlap = best["OverlapAA"]
        r.update(
            SelectedDBD=best["DomainName"], DBDRange=best["DomainRange"],
            DBDStart=best["DomainStart"], DBDEnd=best["DomainEnd"],
            DBDSource=best["SourceDB"], DBDAccession=best["InterProID"],
            DBDType=best["DomainType"], OverlapAA=overlap,
            Overlap=f"{overlap}/15", OverlapFraction=overlap / 15,
            DistanceToDBD=best["DistanceAA"],
            DomainCoordinateQC=("LENGTH_MISMATCH" if best["LengthMatchesCSV"] is False
                                else "LENGTH_MATCH" if best["LengthMatchesCSV"] is True
                                else "LENGTH_UNKNOWN"),
            AnnotationStatus=("FULL_OVERLAP" if overlap == 15 else
                              "PARTIAL_OVERLAP" if overlap > 0 else "NO_OVERLAP"))
    elif hits:
        r["AnnotationStatus"] = "FAMILY_ONLY_NOT_USED"
    elif errs and not found_any_entry:
        r["AnnotationStatus"] = "QUERY_FAILED"
    else:
        r["AnnotationStatus"] = "NO_DBD_ANNOTATION"

    if errs:
        r["Error"] = " ; ".join(errs)
        # An incomplete set of API queries is not reliable evidence for absence of DBD.
        if not good:
            r["AnnotationStatus"] = "INCOMPLETE_QUERY"

    if good and r["DomainCoordinateQC"] == "LENGTH_MISMATCH":
        r["AnnotationStatus"] = "NEEDS_REVIEW_LENGTH"

    if verify_sequence:
        try:
            qc, seq_length = verify_peptide(pid, st, en, peptide, cache_dir)
            r["SequenceQC"] = qc
            if qc != "SEQUENCE_OK" or seq_length != int(r["OriginalLength"]):
                r["AnnotationStatus"] = "NEEDS_REVIEW_SEQUENCE"
        except Exception as exc:
            r["SequenceQC"] = "CHECK_FAILED"
            r["Error"] += " ; FASTA:" + str(exc)[:150]
            r["AnnotationStatus"] = "NEEDS_REVIEW_SEQUENCE"

    return r, hits


def main():
    parser = argparse.ArgumentParser(description="Automatically annotate InterPro DBD/15-mer overlap")
    parser.add_argument("--input", default="highest_15mer_per_protein_human_attentiontfactor.csv")
    parser.add_argument("--output", default="interpro_dbd_results_human_attentiontfactor.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default="interpro_cache")
    parser.add_argument("--verify-sequence", action="store_true",
                        help="Check peptide at stated coordinates against UniProt FASTA (slower)")
    args = parser.parse_args()

    source = pd.read_csv(args.input, dtype={"ProteinID": str})
    required = {"ProteinID", "Start", "End", "Peptide15mer", "OriginalLength"}
    missing = required - set(source.columns)
    if missing:
        parser.error("Missing CSV columns: " + ", ".join(sorted(missing)))

    cache_dir = Path(args.cache)
    results, candidates = [], []

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        futures = [pool.submit(process_row, row, cache_dir, args.verify_sequence)
                   for _, row in source.iterrows()]
        for n, future in enumerate(as_completed(futures), 1):
            result, hits = future.result()
            results.append(result)
            candidates.extend(hits)
            if n % 25 == 0 or n == len(futures):
                print(f"Processed {n}/{len(futures)} proteins", flush=True)

    result_df = pd.DataFrame(results)
    # Keep same order as the input CSV, including any repeated accession rows.
    if "GlobalRank" in result_df:
        result_df.sort_values("GlobalRank", inplace=True)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    outpath = Path(args.output)
    candidate_path = outpath.with_name(outpath.stem + "_all_candidates.csv")
    pd.DataFrame(candidates).to_csv(candidate_path, index=False, encoding="utf-8-sig")

    print("\nAnnotation status counts:")
    for status, count in Counter(result_df["AnnotationStatus"]).most_common():
        print(f"  {status}: {count}")
    print(f"\nResults: {args.output}\nCandidates: {candidate_path}")
    print("Warning: domain-name heuristics are not a curated DBD ontology; review ambiguous entries.")


if __name__ == "__main__":
    main()