
with open('/data/pre_dedup/final_clusters.fasta', 'r') as s:
    with open('/data/pre_dedup/final_reps.fasta', 'w') as r:
        prev = None
        seq = False
        count = 0
        for line in s:
            if line == prev:
                r.write(line)
                seq = True
                count+=1
            elif seq:
                r.write(line)
                seq = False
            prev = line
        print(count)