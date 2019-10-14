class MediforData():
    ref = None
    sys = None
    folder_name = None
    
    def __init__(self, ref, sys, folder_name=""):
        self.ref = ref
        self.sys = sys
        self.folder_name = folder_name