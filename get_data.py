from Bio import Entrez

Entrez.email = "nimrod.shai@mail.huji.ac.il"

# Define the Accession ID
genome_id = "BA000007.3"

print(f"Downloading {genome_id} from NCBI...")


# Fetch the GenBank Full Record (Contains 'Ground Truth' gene locations)
with Entrez.efetch(db="nucleotide", id=genome_id, rettype="gb", retmode="text") as handle:
    with open("data/test_annotations.gb", "w") as out_file:
        out_file.write(handle.read())
print("Saved: test_annotations.gb")