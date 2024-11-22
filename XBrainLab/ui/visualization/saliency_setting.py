import tkinter as tk

from ..base import TopWindow, ValidateException
from ..widget import EditableTreeView
from ..script import Script

class SetSaliencyWindow(TopWindow):
    # return nested dict
    command_label = 'Set Saliency Methods'
    def __init__(self, parent, saliency_params=None):
        super().__init__(parent, 'Saliency Setting')

        self.saliency_params = saliency_params # nested dict of {'method':{'param':value}}
        algo_map ={} # dict of {'method':['param']}

        support_saliency_methods = ['SmoothGrad', 'SmoothGrad_Squared', 'VarGrad']
        for support_saliency_method in support_saliency_methods:
            if support_saliency_method.startswith('Gradient'):
                algo_map[support_saliency_method] = None
            elif support_saliency_method in ['SmoothGrad', 'SmoothGrad_Squared', 'VarGrad']:
                algo_map[support_saliency_method] = ['nt_samples', 'nt_samples_batch_size', 'stdevs']
            else:
                raise NotImplementedError

        columns = ('param', 'value')
        self.params_frames = []
        self.params_trees = {}
        for i, saliency_method in enumerate(support_saliency_methods):
            new_frame = tk.LabelFrame(self, text=f'{saliency_method} Parameters')
            new_tree = EditableTreeView(new_frame, editableCols=['#2'], columns=columns, show='headings')
            self.params_frames.append(new_frame)
            self.params_trees[saliency_method] = new_tree
            new_tree.pack(fill=tk.BOTH, expand=True)
            new_frame.grid(row=i, column=0, columnspan=2, sticky='NEWS')
        PADDING = 5

        tk.Button(self, text='Confirm', command=self.confirm).grid(
            row=len(support_saliency_methods), column=0, columnspan=2, pady=PADDING
        )

        self.algo_map = algo_map
        self.confirm_update = False
        
        self.algo_display()

    def algo_display(self):
        for algo, params_list in self.algo_map.items():
            self.params_trees[algo].delete(*self.params_trees[algo].get_children()) # clear table
            if self.saliency_params is None:
                for param in params_list:
                    if param == 'nt_samples':
                        value = 5
                    elif param == 'nt_samples_batch_size':
                        value = 'None'
                    elif param == 'stdevs':
                        value = 1.0
                    self.params_trees[algo].insert('', index='end', values=(param, value))
            else:
                for param in params_list:
                    value = self.saliency_params[algo][param]
                    self.params_trees[algo].insert('', index='end', values=(param, value))
    def confirm(self):
        saliency_params = {}
        reason = None
        for algo, params_tree in self.params_trees.items():
            params_tree.submitEntry()
            saliency_params[algo] = {}
            try:
                for item in params_tree.get_children():
                    param, value = params_tree.item(item)['values']
                    if param.startswith('nt_samples'):
                        assert value == 'None' or str(value).isdigit(), "invalid value"
                        if value != 'None':
                            value = int(value)
                        else:
                            value = None
                    else:
                        if value != '':
                            if value =='None':
                                value = None
                            elif value == 'True':
                                value = True
                            elif value == 'False':
                                value = False
                            else:
                                value = float(value)
                        
                    saliency_params[algo][param] = value
            except Exception:
                reason = 'Invalid parameter'
            if reason:
                raise ValidateException(window=self, message=reason)
        self.saliency_params = saliency_params

        # script
        self.script_history = Script()
        self.script_history.add_cmd(f'saliency_params={self.saliency_params!r}') 

        self.confirm_update = True
        self.destroy()

    def _get_result(self):
        return self.confirm_update, self.saliency_params
    
    def _get_script_history(self):
        return self.script_history
