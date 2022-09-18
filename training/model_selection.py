import tkinter as tk
from .. import model_base
from ..base import TopWindow
from ..widget import EditableTreeView
import tkinter.filedialog
import inspect
import os

ARG_DICT_SKIP_SET = set(['self', 'n_classes', 'channels', 'samples', 'sfreq'])

class ModelHolder:
    def __init__(self, target_model, model_parms_map, pretrained_weight_path):
        self.target_model = target_model
        self.model_parms_map = model_parms_map
        self.pretrained_weight_path = pretrained_weight_path

    def get_model(self, args):
        model = self.target_model(**self.model_parms_map, **args)
        if self.pretrained_weight_path:
            model.load_state_dict(torch.load(self.pretrained_weight_path))
        return model

class ModelSelectionWindow(TopWindow):
    def __init__(self, parent):
        super().__init__(parent, 'Model Selection')
        self.pretrained_weight_path = None
        self.model_holder = None

        ## fetch model list from db
        model_map = inspect.getmembers(model_base, inspect.isclass)
        model_map = {m[0]: m[1] for m in model_map}
        model_list = list(model_map.keys())

        ## model option menu
        model_opt_label = tk.Label(self, text='model: ')
        selected_model_name = tk.StringVar(self)
        selected_model_name.trace('w', self.on_model_select) # callback
        model_opt = tk.OptionMenu(self, selected_model_name, *model_list)
        
        ## model parms
        parms_frame = tk.LabelFrame(self, text='model parmeters')
        columns = ('parm', 'value')
        parms_tree = EditableTreeView(parms_frame, editableCols=['#2'], columns=columns, show='headings')
        parms_tree.pack(fill=tk.BOTH, expand=True)

        ## pretrained weight
        pretrained_weight_label = tk.Label(self, text='Pretrained weight: ')
        pretrained_weight_disp_label = tk.Label(self)
        pretrained_weight_btn = tk.Button(self, text='load', command=self.load_pretrained_weight)
        
        model_opt_label.grid(row=1, column=0)
        model_opt.grid(row=1, column=1)
        parms_frame.grid(row=2, column=0, columnspan=3, sticky='NEWS')
        pretrained_weight_label.grid(row=3, column=0)
        pretrained_weight_disp_label.grid(row=3, column=1)
        pretrained_weight_btn.grid(row=3, column=2)
        tk.Button(self, text='Confirm', command=self.confirm).grid(row=4, column=2)
        self.columnconfigure([0,1,2], weight=1)
        self.rowconfigure([2], weight=1)

        self.model_map = model_map
        self.selected_model_name = selected_model_name
        self.parms_tree = parms_tree
        self.parms_frame = parms_frame
        self.pretrained_weight_disp_label = pretrained_weight_disp_label
        self.pretrained_weight_btn = pretrained_weight_btn

        selected_model_name.set(model_list[0])

    def on_model_select(self, *args):
        """Update model params when model is selected"""
        target = self.model_map[self.selected_model_name.get()]
        self.parms_tree.delete(*self.parms_tree.get_children()) # clear table
        contain = False
        if target:
            sigs = inspect.signature(target.__init__)
            parms = sigs.parameters
            for parm in parms:
                if parm in ARG_DICT_SKIP_SET:
                    continue
                value = value = '' if parms[parm].default == inspect._empty else parms[parm].default
                self.parms_tree.insert('', index='end', values=(parm, value))
                contain = True
        if contain:
            self.parms_frame.grid()
        else:
            self.parms_frame.grid_remove()
    
    def load_pretrained_weight(self):
        if self.pretrained_weight_path:
            # perform clear if already set
            self.pretrained_weight_path = None
            # update display
            self.pretrained_weight_disp_label.config(text='')
            self.pretrained_weight_btn.config(text='load')
            return
        
        filepath = tk.filedialog.askopenfilename(parent=self, filetypes=(("model/weights", "*"),))
        if filepath:
            self.pretrained_weight_path = filepath
            filename = os.path.basename(filepath)
            self.pretrained_weight_disp_label.config(text=filename)
            self.pretrained_weight_btn.config(text='clear')

    def confirm(self):
        self.parms_tree.submitEntry() # close editing tables
        reason = None    
        
        target_model = self.model_map[self.selected_model_name.get()]
        model_parms_map = {}
        for item in self.parms_tree.get_children():
            parm, value = self.parms_tree.item(item)['values']
            model_parms_map[parm] = value
        
        self.model_holder = ModelHolder(target_model, model_parms_map, self.pretrained_weight_path )
        self.destroy()

    def _get_result(self):
        return self.model_holder

if __name__ == '__main__':
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    window = ModelSelectionWindow(root)

    print (window.get_result())
    root.destroy()