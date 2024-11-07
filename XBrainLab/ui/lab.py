from __future__ import annotations

import contextlib
import traceback

from XBrainLab import Study
from XBrainLab.evaluation import Metric
from XBrainLab.training import TrainingPlanHolder
from XBrainLab.visualization import PlotType, VisualizerType

from . import tk_patch
from .dash_board import DashBoard
from .evaluation import EvaluationTableWindow
from .script import Script
from .visualization import (
    PlotABSFigureWindow,
    PlotEvalRecordFigureWindow,
    PlotTopoABSFigureWindow,
)
from .widget import PlotFigureWindow


class XBrainLab:
    """Class for XBrainLab study workflow.

    Attributes:
        ui: :class:`XBrainLab.ui.dash_board.DashBoard` or None.
            The UI of XBrainLab.
        study: :class:`XBrainLab.Study`.
            The study instance.
        script_history: :class:`XBrainLab.ui.script.Script` or None.
            The script history generated from UI.
    """
    def __init__(self, study=None):
        self.ui = None
        if study is None:
            study = Study()
        self.study = study
        self.script_history = Script()
        tk_patch.patch()


    def get_script(self) -> Script | None:
        """Return script history generated from UI."""
        return self.script_history

    def get_study(self) -> Study:
        """Return current study."""
        return self.study

    def show_ui(self, interact=False):
        """Show UI.

        Args:
            interact: Whether to run in interactive mode.
                      If True, the UI will run in a new thread.
        """
        # close previous ui
        try:
            if(self.ui):
                self.ui.destroy(force=True)
        except Exception:
            pass
        self.ui = None

        try:
            self.ui = DashBoard(self.study, self.script_history)
            if not interact:
                self.ui_loop()
        except Exception as e:
            traceback.print_exc()
            # recyle ui
            with contextlib.suppress(Exception):
                self.ui.destroy(force=True)

            # show error message
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                tk.messagebox.showerror(parent=root, title='Error', message=e)
                root.destroy()
            except Exception as e:
                traceback.print_exc()
                raise e
        else:
            return self.ui
    def ui_loop(self) -> None:
        """Run UI loop in main thread."""
        if self.ui is None or not self.ui.window_exist:
            self.show_ui(interact=False)
        else:
            self.ui.mainloop()
            self.ui = None

    def ui_func_wrapper(func) -> callable:
        """Decorator for UI related functions.

        Helper function to show UI and provide trainers to called function.
        """
        def wrap(*args, **kwargs):
            lab = args[0]
            study = lab.study
            trainers = None
            if study.trainer:
                trainers = study.trainer.get_training_plan_holders()
            if not lab.ui:
                lab.show_ui(interact=True)
            func(*args, **kwargs, trainers=trainers)
            lab.ui_loop()

        return wrap

    @ui_func_wrapper
    def show_plot(
        self,
        plot_type: PlotType,
        plan_name: str,
        real_plan_name: str,
        trainers: list[TrainingPlanHolder]
    ) -> None:
        """Show figure window.

        Args:
            plot_type: The plot type.
            plan_name: The name of training plan.
            real_plan_name: The name of real plan under training plan.
            trainers: The list of :class:`XBrainLab.training.TrainingPlanHolder`.
        """
        PlotFigureWindow(parent=self.ui, trainers=trainers,
            plot_type=plot_type, plan_name=plan_name, real_plan_name=real_plan_name)

    @ui_func_wrapper
    def show_grad_plot(
        self,
        plot_type: VisualizerType,
        plan_name: str,
        real_plan_name: str,
        saliency_name: str,
        absolute: bool,
        trainers: list[TrainingPlanHolder]
    ) -> None:
        """Show gradient figure window.

        Args:
            plot_type: The plot type.
            plan_name: The name of training plan.
            real_plan_name: The name of real plan under training plan.
            absolute: Whether to use absolute value.
            trainers: The list of :class:`XBrainLab.training.TrainingPlanHolder`.
        """
        PlotABSFigureWindow(
            parent=self.ui, trainers=trainers,
            plot_type=plot_type, plan_name=plan_name, real_plan_name=real_plan_name, saliency_name=saliency_name,
            absolute=absolute
        )

    @ui_func_wrapper
    def show_grad_topo_plot(
        self,
        plot_type: VisualizerType,
        plan_name: str,
        real_plan_name: str,
        saliency_name: str,
        absolute: bool,
        trainers: list[TrainingPlanHolder]
    ) -> None:
        """Show gradient topographic figure window.

        Args:
            plot_type: The plot type.
            plan_name: The name of training plan.
            real_plan_name: The name of real plan under training plan.
            absolute: Whether to use absolute value.
            trainers: The list of :class:`XBrainLab.training.TrainingPlanHolder`.
        """
        PlotTopoABSFigureWindow(
            parent=self.ui, trainers=trainers,
            plot_type=plot_type, plan_name=plan_name, saliency_name=saliency_name,
            real_plan_name=real_plan_name, absolute=absolute
        )

    @ui_func_wrapper
    def show_grad_eval_plot(
        self,
        plot_type: VisualizerType,
        plan_name: str,
        real_plan_name: str,
        saliency_name: str,
        trainers: list[TrainingPlanHolder]
    ) -> None:
        """Show evaluation figure window.

        Args:
            plot_type: The plot type.
            plan_name: The name of training plan.
            real_plan_name: The name of real plan under training plan.
            trainers: The list of :class:`XBrainLab.training.TrainingPlanHolder`.
        """
        PlotEvalRecordFigureWindow(
            parent=self.ui, trainers=trainers,
            plot_type=plot_type, plan_name=plan_name, real_plan_name=real_plan_name, saliency_name=saliency_name
        )


    @ui_func_wrapper
    def show_performance(
        self,
        metric: Metric,
        trainers: list[TrainingPlanHolder]
    ) -> None:
        """Show performance window.

        Args:
            metric: The metric type.
            trainers: The list of :class:`XBrainLab.training.TrainingPlanHolder`.
        """
        EvaluationTableWindow(self.ui, trainers, metric)
