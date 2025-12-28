from Bio import Entrez

Entrez.email = "nimrod.shai@mail.huji.ac.il"

# 2. Define the Accession ID
genome_id = "BA000007.3"

print(f"Downloading {genome_id} from NCBI...")

# 3. Fetch the FASTA file (The Raw DNA)
with Entrez.efetch(db="nucleotide", id=genome_id, rettype="fasta", retmode="text") as handle:
    with open("data/test_genome.fasta", "w") as out_file:
        out_file.write(handle.read())
print("Saved: test_genome.fasta")

# 4. Fetch the GenBank Full Record (Contains 'Ground Truth' gene locations)
#    (Note: 'gb' format is often easier to parse in BioPython than GFF)
with Entrez.efetch(db="nucleotide", id=genome_id, rettype="gb", retmode="text") as handle:
    with open("data/test_annotations.gb", "w") as out_file:
        out_file.write(handle.read())
print("Saved: test_annotations.gb")