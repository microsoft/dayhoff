import random
import json

with open("/data/post_dedup/dedup_clusters.fasta", 'r') as f, open("/data/gigaref/private/consensus.fasta", 'w') as private, open("/data/gigaref/consensus.fasta", 'w') as consensus:
    current_cluster = []
    prev = None
    id = None
    seq = None
    size = 0
    indices = []
    index = 0
    private_index = 0
    public_index = 0
    private_clu_json = {'test': []}
    private_rep_json = {'test': []}
    clu_json = {'train': [], 'test': []}
    clu_no_singles_json = {'train': [], 'test': []}
    rep_json = {'train': [], 'test': []}
    rep_no_singles_json = {'train': [], 'test': []}

    for line in f:
        if prev == line:
            if current_cluster:
                if size > 1:
                    rand = random.random()
                    if (rand < 4.1e-5):
                        private.writelines(current_cluster)
                        private_clu_json['test'].append([i + private_index for i in range(len(indices))])
                        private_rep_json['test'].append(private_index)
                        private_index += len(indices)
                    elif(rand < 8.2e-5):
                        consensus.writelines(current_cluster)
                        clu_json['test'].append([i + public_index for i in range(len(indices))])
                        clu_no_singles_json['test'].append([i + public_index for i in range(len(indices))])
                        rep_json['test'].append(public_index) 
                        rep_no_singles_json['test'].append(public_index)
                        public_index += len(indices)
                    else:
                        consensus.writelines(current_cluster)
                        clu_json['train'].append([i + public_index for i in range(len(indices))])
                        clu_no_singles_json['train'].append([i + public_index for i in range(len(indices))])
                        rep_json['train'].append(public_index) 
                        rep_no_singles_json['train'].append(public_index)
                        public_index += len(indices)
                else:
                    consensus.writelines(current_cluster)
                    clu_json['train'].append([i + public_index for i in range(len(indices))])
                    rep_json['train'].append(public_index)
                    public_index += len(indices)
            current_cluster = []
            size = 0
            indices = []
        if line.startswith('>'):
            if id and seq:
                current_cluster.append(id)
                current_cluster.append(seq)
                size += 1
            id = line
            seq = None
        else:
            indices.append(index)
            seq = line
            index+=1
        prev = line

    print("Done reading file")

    if id and seq:
        current_cluster.append(id)
        current_cluster.append(seq)
        size += 1
    if current_cluster:
        if size > 1:
            rand = random.random()
            if (rand < 4.1e-5):
                private.writelines(current_cluster)
                private_clu_json['test'].append([i + private_index for i in range(len(indices))])
                private_rep_json['test'].append(private_index)
                private_index += len(indices)
            elif(rand < 8.2e-5):
                consensus.writelines(current_cluster)
                clu_json['test'].append([i + public_index for i in range(len(indices))])
                clu_no_singles_json['test'].append([i + public_index for i in range(len(indices))])
                rep_json['test'].append(public_index) 
                rep_no_singles_json['test'].append(public_index)
                public_index += len(indices)
            else:
                consensus.writelines(current_cluster)
                clu_json['train'].append([i + public_index for i in range(len(indices))])
                clu_no_singles_json['train'].append([i + public_index for i in range(len(indices))])
                rep_json['train'].append(public_index) 
                rep_no_singles_json['train'].append(public_index)
                public_index += len(indices)
        else:
            consensus.writelines(current_cluster)
            clu_json['train'].append([i + public_index for i in range(len(indices))])
            rep_json['train'].append(public_index)
            public_index += len(indices)

    print("Finished last cluster")

    consensus.flush()
    consensus.close()
    private.flush()
    private.close()

    print("Closed files")

    with open("/data/gigaref/private/clustered_splits.json", 'w') as f:
        json.dump(private_clu_json, f)

    with open("/data/gigaref/private/splits.json", 'w') as f:
        json.dump(private_rep_json, f)

    with open("/data/gigaref/with_singletons/clustered_splits.json", 'w') as f:
        json.dump(clu_json, f)

    with open("/data/gigaref/with_singletons/splits.json", 'w') as f:    
        json.dump(rep_json, f)

    with open("/data/gigaref/no_singletons/clustered_splits.json", 'w') as f:
        json.dump(clu_no_singles_json, f)

    with open("/data/gigaref/no_singletons/splits.json", 'w') as f:
        json.dump(rep_no_singles_json, f)

    print("Done writing json files")