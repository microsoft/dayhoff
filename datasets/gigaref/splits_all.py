import json

#dirs = ["/data/gigaref/with_singletons/", "/data/gigaref/no_singletons/", "/data/gigaref/private/"]
dirs = ["/data/gigaref/private/"]

for dir in dirs:
    with open(dir+"clustered_splits.json", 'r') as f:
        data = json.load(f)

    train_indices = []
    test_indices = []

    if not 'private' in dir:
        for cluster in data['train']:
            train_indices.extend(cluster)

    for cluster in data['test']:
        test_indices.extend(cluster)

    if 'private' in dir:
        new_data = {
            'test': test_indices
        }
    else:
        new_data = {
            'train': train_indices,
            'test': test_indices
        }

    with open(dir+"splits_all.json", 'w') as f:
        json.dump(new_data, f)