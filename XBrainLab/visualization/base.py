class Visiualizer:
    def __init__(self, eval_record, epoch_data, figsize=(6.4, 4.8), dpi=100, fig=None):
        self.eval_record = eval_record
        self.epoch_data = epoch_data
        self.figsize = figsize
        self.dpi = dpi
        self.fig = fig
    
    def get_plt(self, absolute):
        raise NotImplementedError()

    def get_gradient(self, labelIndex):
        #return self.eval_record.gradient[labelIndex][self.eval_record.label == labelIndex]
        return self.eval_record.gradient[labelIndex]