from picml.custom import custom_objects as cobj


class ModelInstance:
    """
    Should extend with save and integrations
    TODO: convert batch_size(as extension)
    """
    
    def __init__(self, model, custom_objects=None):
        self.model = model
        self.custom_objects = custom_objects

    def save(self, path):
        tdir = tempfile.TemporaryDirectory()
        if self.custom_objects is not None:
            dill.dump(self.custom_objects, open(os.path.join(tdir.name, "meta.pkl"), "wb"))
        keras.models.save_model(self.model, os.path.join(tdir.name, "model.h5"))
        zip_file = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
        for root, _, files in os.walk(tdir.name):
            for file in files:
                zip_file.write(os.path.join(root, file), file)        
        [print(zinfo) for zinfo in zip_file.filelist]
        zip_file.close()
        
    @classmethod
    def load(cls, path):
        tdir = tempfile.TemporaryDirectory()
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(tdir.name)
        zip_ref.close()
        if os.path.isfile(os.path.join(tdir.name, "meta.pkl")):
            meta = dill.load(open(os.path.join(tdir.name, "meta.pkl"), "rb"))
            meta.update(cobj)
            mdl = keras.models.load_model(os.path.join(tdir.name, "model.h5"), meta)
        else:
            meta = None
            mdl = keras.models.load_model(os.path.join(tdir.name, "model.h5"), cobj)
        return ModelInstance(mdl, meta)
