from distutils.core import setup as csetup
from distutils.extension import Extension

def get_extensions():
    extensions = []
    return extensions


def setup_package():
    metadata = dict()
    csetup(**metadata)


if __name__ == '__main__':
    setup_package()


