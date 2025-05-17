import pdb
import sys

seed_dict = {'abu-airport-4': 7105, 'abu-beach-3': 6140, 'abu-urban-2': 9905,
             'abu-urban-3': 8180, 'abu-urban-4': 2413, 'hydice': 8123}


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def get_params(net):
    '''Returns parameters that we want to optimize over.
    '''
    params = []
    params += [x for x in net.parameters()]

    return params


def img2mask(img):
    img = img[0].sum(0)
    img = img - img.min()
    img = img / img.max()
    img = img.detach().cpu().numpy()

    return img