
class History:
    def __init__(self, *keys):
        self.data = {key: [] for key in keys}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def to_dict(self):
        return self.data
    
    def plot(self):
        pass
        
    def wandb(self):
        pass