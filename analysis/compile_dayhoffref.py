import os

individual_dir = "/home/kevyan/generations/dayhoffref/dayhoff_generations/"
individual_files = os.listdir(individual_dir)

out_file = "/home/kevyan/generations/dayhoffref/dayhoffref.fasta"
with open(out_file, 'w') as out:
    for individual_file in individual_files:
        name = individual_file.replace(".fasta", "")
        name = name.replace("jamba-", "")
        name = name.replace("10mbothfilter", "bbr-novel-sc")
        print(name)
        with open(os.path.join(individual_dir, individual_file), 'r') as infile:
            for line in infile:
                if line.startswith(">"):
                    out.write(">" + name + "_" + line[1:])
                else:
                    out.write(line)
