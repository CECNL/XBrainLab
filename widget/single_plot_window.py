from ..base import TopWindow
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

class SinglePlotWindow(TopWindow):
    PLOT_COUNTER = 0
    def __init__(self, parent, figsize=None, dpi=None):
        super().__init__(parent, 'Plot')
        if figsize is None:
            figsize = (6.4, 4.8)
        if dpi is None:
            dpi = 100
        
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.plot_number = f'SinglePlotWindow-{SinglePlotWindow.PLOT_COUNTER}'
        SinglePlotWindow.PLOT_COUNTER += 1
        # create dummy figure
        figure = plt.figure(num=self.plot_number, figsize=figsize, dpi=dpi)
        self.set_figure(figure, figsize, dpi)

        # resize to figure size
        self.update_idletasks()
        width = self.figure_canvas.get_tk_widget().winfo_width()
        heigh = self.figure_canvas.get_tk_widget().winfo_height()
        target_width, target_height = [int(s * self.fig_parm['dpi']) for s in self.fig_parm['figsize']]
        target_width -= width
        target_height -= heigh
        self.geometry(f"{self.winfo_width() + target_width}x{self.winfo_height() + target_height}")

    def active_figure(self):
        plt.figure(self.plot_number)

    def get_figure_parms(self):
        return self.fig_parm
    
    def clear_figure(self):
        plt.clf()
        self.redraw()

    def empty_data_figure(self):
        self.clear_figure()
        plt.text(.5, .5, 'No data is available.', ha='center', va='center')

    def set_figure(self, figure, figsize, dpi):
        fig_frame = tk.Frame(self)
        # create Figure_to_CanvasTkAgg object
        figure_canvas = FigureCanvasTkAgg(figure, fig_frame)
        # create the toolbar
        NavigationToolbar2Tk(figure_canvas, fig_frame)
        
        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        fig_frame.grid(row=1, column=0, sticky='news')

        self.figure_canvas = figure_canvas
        if hasattr(self, 'fig_parm'):
            plt.close(self.fig_parm['fig'])
        self.fig_parm = {'fig': figure, 'figsize': figsize, 'dpi': dpi}

    def redraw(self):
        self.fig_parm['fig'].tight_layout()
        self.figure_canvas.draw()