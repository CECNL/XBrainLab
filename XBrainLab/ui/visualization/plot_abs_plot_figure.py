import tkinter as tk

from ..widget import PlotFigureWindow

class PlotABSFigureWindow(PlotFigureWindow):
    def __init__(
        self, 
        parent, 
        trainers, 
        plot_type, 
        figsize=None, 
        title='Plot', 
        plan_name=None, 
        real_plan_name=None, 
        absolute=None, 
        spectrogram=None
    ):
        super().__init__(
            parent, trainers, plot_type, figsize, title, plan_name, real_plan_name
        )
        self.absolute_var = tk.BooleanVar(self)
        self.absolute_var.trace('w', self.absolute_callback)
        self.spectrogram_var = tk.BooleanVar(self)
        self.spectrogram_var.trace('w', self.spectrogram_callback)

        self.abs_btn = tk.Checkbutton(
            self.selector_frame, text='absolute value',var=self.absolute_var
        )
        self.spec_btn = tk.Checkbutton(
            self.selector_frame, text='spectrogram value',var=self.spectrogram_var
        )

        self.abs_btn.pack()
        self.spec_btn.pack()

        # checkbutton spectrogram
        if absolute is not None:
            self.absolute_var.set(absolute)

        if spectrogram is not None:
            self.spectrogram_var.set(spectrogram)

    def add_plot_command(self):
        if not hasattr(self, 'absolute_var') or not hasattr(self, 'spectrogram_var'):
            return
        self.script_history.add_import(
            "from XBrainLab.visualization import SaliencyMapViz"
        )
        self.script_history.add_ui_cmd((
            f"study.show_grad_plot(plot_type={self.plot_type.__name__}, "
            f"plan_name={repr(self.selected_plan_name.get())}, "
            f"real_plan_name={repr(self.selected_real_plan_name.get())}, "
            f"absolute={repr(self.absolute_var.get())}, "
            f"spectrogram={repr(self.spectrogram_var.get())})"
        ))

    def absolute_callback(self, *args):
        self.add_plot_command()
        self.recreate_fig()

    def spectrogram_callback(self, *args):
        if self.spectrogram_var.get():
            self.abs_btn.config(state=tk.DISABLED)
        else:
            self.abs_btn.config(state=tk.NORMAL)
        self.add_plot_command()
        self.recreate_fig()
    
    
    def _create_figure(self):
        if not hasattr(self, 'absolute_var') or not hasattr(self, 'spectrogram_var'):
            return None
       
        eval_record = self.plan_to_plot.get_eval_record()
        if not eval_record:
            return None
        
        epoch_data = self.trainer.get_dataset().get_epoch_data()
        plot_visualizer = self.plot_type(
            eval_record, epoch_data, **self.get_figure_params()
        )
        figure = plot_visualizer.get_plt(
            absolute=self.absolute_var.get(), 
            spectrogram=self.spectrogram_var.get(), 
            sfreq=epoch_data.get_model_args()['sfreq']
        )
        return figure
