import yaml

def main(path='configs/topology.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        topo = yaml.safe_load(f)
    print('Topology loaded:', topo)

if __name__ == '__main__':
    main()
