"""EvoAtlas data acquisition: NCBI fetch + local alignment + demo fallback."""
import os, time, requests
from Bio import Entrez, SeqIO
from Bio.Align import PairwiseAligner
from Bio.AlignIO import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np

Entrez.email = "evoatlas@example.com"

def fetch_ncbi_sequences(query: str, db: str = "protein", max_seqs: int = 200,
                          cache_path: str = None) -> list:
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached sequences from {cache_path}")
        return list(SeqIO.parse(cache_path, "fasta"))
    print(f"Searching NCBI {db}: '{query}'")
    handle = Entrez.esearch(db=db, term=query, retmax=max_seqs)
    record = Entrez.read(handle)
    ids = record["IdList"]
    print(f"Found {len(ids)} sequences, fetching...")
    if not ids:
        raise ValueError(f"No sequences found for query: {query}")
    all_records = []
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        handle = Entrez.efetch(db=db, id=",".join(batch), rettype="fasta", retmode="text")
        records = list(SeqIO.parse(handle, "fasta"))
        all_records.extend(records)
        time.sleep(0.4)
    if cache_path:
        SeqIO.write(all_records, cache_path, "fasta")
    return all_records


def download_demo_spike_sequences(n: int = 100, out_dir: str = "data/") -> list:
    os.makedirs(out_dir, exist_ok=True)
    cache = f"{out_dir}/spike_sequences.fasta"
    try:
        records = fetch_ncbi_sequences(
            query=(
                'SARS-CoV-2[Organism] AND "surface glycoprotein"[Protein Name] '
                'AND 500:1300[Sequence Length] AND refseq[Filter]'
            ),
            db="protein", max_seqs=n, cache_path=cache
        )
        if len(records) >= 5:
            return records
    except Exception as e:
        print(f"NCBI fetch failed ({e}), using embedded demo sequences")

    fallback_seqs = {
        "WT_Wuhan":    "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLE",
        "Alpha_B117":  "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLE",
        "Delta_B16172":"NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLRSYGFQPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTPSSKRFQPFQQFGRDIADTTDAVRDPQTLE",
        "Omicron_BA1": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSATKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSKHIDAKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTPSSKRFQPFQQFGRDIADTTDAVRDPQTLE",
        "XBB15":       "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSATKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSKHIDAKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTPSSKRFQPFQQFGRDIADTTDAVRDPQTLE",
    }
    records = [
        SeqRecord(Seq(seq), id=name, description=name)
        for name, seq in fallback_seqs.items()
    ]
    SeqIO.write(records, cache, "fasta")
    return records


def align_sequences(records: list) -> tuple:
    """Global pairwise alignment building an MSA matrix."""
    valid = [r for r in records
             if len(str(r.seq).replace("-", "").replace("X", "").replace("N", "")) > 50]
    print(f"Aligning {len(valid)} sequences...")
    if len(valid) < 2:
        raise ValueError("Need at least 2 valid sequences")
    ref = max(valid, key=lambda r: len(r.seq))
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    aligned_seqs, seq_ids = [], []
    for rec in valid[:200]:
        if rec.id == ref.id:
            aligned_seqs.append(str(rec.seq))
        else:
            try:
                alignments = aligner.align(str(ref.seq), str(rec.seq))
                aligned_seqs.append(str(alignments[0]).split("\n")[2])
            except Exception:
                s = str(rec.seq)
                L = len(str(ref.seq))
                aligned_seqs.append(s[:L].ljust(L, "-") if len(s) >= L
                                    else s + "-" * (L - len(s)))
        seq_ids.append(rec.id)
    align_len = len(aligned_seqs[0])
    msa = np.array([list(s.ljust(align_len, "-")) for s in aligned_seqs])
    print(f"Alignment: {msa.shape[0]} × {msa.shape[1]}")
    return msa, seq_ids
