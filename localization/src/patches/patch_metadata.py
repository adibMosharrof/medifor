
class PatchMetadata():
    def __init__(self, shape, patch_window_shape, id):
        self.original_image_shape = shape
        self.patch_window_shape = patch_window_shape
        self.probe_file_id = id