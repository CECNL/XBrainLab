import tkinter as tk
import traceback

import matplotlib

from ..base import TopWindow

try:
    matplotlib.use('TkAgg')
except Exception:
    traceback.print_exc()

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class SinglePlotWindow(TopWindow):
    PLOT_COUNTER = 0
    def __init__(self, parent, figsize=None, dpi=None, title='Plot'):
        super().__init__(parent, title)
        if figsize is None:
            figsize = (6.4, 4.8)
        if dpi is None:
            dpi = 100
        self.figsize = figsize
        self.dpi = dpi

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.figure_canvas = None
        self.plot_number = None
        self.init_figure()

        # resize to figure size
        self.update_idletasks()
        width = self.figure_canvas.get_tk_widget().winfo_width()
        heigh = self.figure_canvas.get_tk_widget().winfo_height()
        target_width, target_height = (
            int(s * self.fig_param['dpi']) for s in self.fig_param['figsize']
        )
        target_width -= width
        target_height -= heigh
        self.geometry(f"{self.winfo_width() + target_width}"
                       "x"
                       f"{self.winfo_height() + target_height}"

        )

    def active_figure(self):
        plt.figure(self.plot_number)

    def init_figure(self):
        # could crash system called in threads
        # if self.plot_number is not None:
        #     self.active_figure()
        #     plt.close()

        self.plot_number = f'SinglePlotWindow-{SinglePlotWindow.PLOT_COUNTER}'
        SinglePlotWindow.PLOT_COUNTER += 1
        # create dummy figure
        figure = plt.figure(num=self.plot_number, figsize=self.figsize, dpi=self.dpi)
        self.set_figure(figure, self.figsize, self.dpi)
        self.active_figure()


    def get_figure_params(self):
        self.init_figure()
        return self.fig_param

    def clear_figure(self):
        plt.clf()
        self.redraw()

    def show_drawing(self):
        self.clear_figure()
        plt.text(.5, .5, 'Drawing.', ha='center', va='center')
        self.redraw()

    def empty_data_figure(self):
        self.clear_figure()
        plt.text(.5, .5, 'No data is available.', ha='center', va='center')
        self.redraw()

    def set_figure(self, figure, figsize, dpi):
        if self.figure_canvas:
            self.figure_canvas.get_tk_widget().destroy()

        fig_frame = tk.Frame(self)
        # create Figure_to_CanvasTkAgg object
        figure_canvas = FigureCanvasTkAgg(figure, fig_frame)
        # create the toolbar
        NavigationToolbar2Tk(figure_canvas, fig_frame)

        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        fig_frame.grid(row=1, column=0, sticky='news')

        self.figure_canvas = figure_canvas
        # could crash system called in threads
        # if hasattr(self, 'fig_param'):
        #     plt.close(self.fig_param['fig'])
        self.fig_param = {'fig': figure, 'figsize': figsize, 'dpi': dpi}

    def redraw(self):
        self.fig_param['fig'].tight_layout()
        self.figure_canvas.draw()
