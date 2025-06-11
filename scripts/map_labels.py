import json
import sys

def load_mapping(path='imagenet_class_index.json'):
    with open(path) as f:
        data = json.load(f)
    return {int(k): v[1] for k,v in data.items()}

if __name__ == '__main__':
    mapping = load_mapping()
    preds = [int(x) for x in sys.argv[1:]]
    labels = [mapping.get(i, 'Unknown') for i in preds]
    print(labels)