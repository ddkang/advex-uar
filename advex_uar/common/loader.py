import os
import shutil
import tempfile

import torchvision

class StridedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, *args, **kwargs):
        self.stride = kwargs['stride']
        del kwargs['stride']

        self.new_root = tempfile.mkdtemp()
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        classes = classes[::self.stride]
        for cls in classes:
            os.symlink(os.path.join(root, cls), os.path.join(self.new_root, cls))

        super().__init__(self.new_root, *args, **kwargs)

    def __del__(self):
        shutil.rmtree(self.new_root)
