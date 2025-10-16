from grain2mesh import Grain2Mesh

if __name__ == '__main__':
    g2m = Grain2Mesh()
    g2m.read_config('./example/binary_config.json')
    g2m.run()
