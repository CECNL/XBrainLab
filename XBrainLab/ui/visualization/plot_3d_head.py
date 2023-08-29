import os
import shutil
import mne
import numpy as np
import pyvista as pv
import argparse
import matplotlib.pyplot as plt

class Saliency3D:
    def __init__(self, eval_record, epoch_data, selected_event_name):
        # set parameters
        self.selected_event_name = selected_event_name
        self.save = False
        self.showChannel = True
        self.showHead = True
        self.cmap = plt.cm.get_cmap("viridis")
        self.neighbor = 5

        # load 3d model
        mesh = pv.read('./XBrainLab/visualization/3Dmodel/head.ply')
        mesh2 = pv.read('./XBrainLab/visualization/3Dmodel/brain.ply')

        # get saliency
        labelIndex = epoch_data.event_id[self.selected_event_name]
        self.saliency = eval_record.gradient[labelIndex]#[eval_record.label == labelIndex]
        self.saliency = self.saliency.mean(axis=0)

        for i in range(self.saliency.shape[-1]):
            self.saliency[:, i] = (self.saliency[:, i] - min(self.saliency[:, i])) / (max(self.saliency[:, i]) - min(self.saliency[:, i]))
        self.max_time = self.saliency.shape[-1]

        # get channel pos
        ch_pos = epoch_data.get_montage_position() # close plt
        electrode = epoch_data.get_channel_names()

        # get electrode pos in 3d
        pos_on_3d = []
        trans = [-0.0004, 0.00917, mesh.bounds[5] - 0.10024]  # trans Cz to [0, 0, 0]
        for ele in electrode:
            center = ch_pos[electrode.index(ele)] + trans
            if center[1] > 0:
                center[2] += 0.007
            pos_on_3d.append(center)
        self.pos_on_3d = np.asarray(pos_on_3d)

        self.chs = [pv.Sphere(radius=0.005, center=self.pos_on_3d[i, :]) for i in range(self.saliency.shape[0])]

        # set plotter
        self.plotter = pv.Plotter(window_size=[750, 750])
        self.head = mesh
        self.brain = mesh2.scale((0.001, 0.001, 0.001), inplace=False).triangulate()
        self.head1 = self.head.clip_closed_surface('z', origin=[0, 0, self.pos_on_3d[:, 2].min()])
        self.scalar = np.zeros(self.head1.n_points)
        self.channelActor = []
        self.headActor = None
        self.param = {
            'timestamp': 1,
            'save': self.save,
        }
        self.saveBox = CheckboxObj(self.param['save'])
        self.channelBox = CheckboxObj(self.showChannel)
        self.headBox = CheckboxObj(self.showHead)
        self.update(save=0)

    def __call__(self, key, value):
        self.param[key] = value
        self.update(save=self.saveBox.ctrl)

    def update(self, save=0):
        self.param['save'] = save
        for i in range(self.head1.n_points):
            dist = [np.linalg.norm(self.head1.points[i] - ch) for ch in self.pos_on_3d]
            dist_idx = np.argsort(dist)[:self.neighbor]  # color interpolated by #neighbor closest points
            dist = [dist[id] for id in dist_idx]
            dist = 1 - dist / np.sum(dist)
            self.scalar[i] = np.sum([self.saliency[dist_idx[j], self.param['timestamp'] - 1] * dist[j] for j in range(self.neighbor)])
        try:
            self.plotter.update_scalars(self.scalar, self.head1)
        except:
            pass

        if self.channelActor != []:
            for actor in self.channelActor:
                actor.SetVisibility(self.channelBox.ctrl)
        if save:
            self.plotter.save_graphic(f"event-{self.selected_event_name}_time-{self.param['timestamp']}_saliency3d.svg")

        if self.headBox.ctrl:
            if self.headActor == None:
                self.headActor = self.plotter.add_mesh(self.head, opacity=0.3)
        else:
            self.plotter.remove_actor(self.headActor)
            self.headActor = None

    def get3dHeadPlot(self):
        self.plotter.add_slider_widget(
            callback=lambda val: self('timestamp', int(val)),
            rng=[1, self.max_time], value=1,
            title="Timestamp",
            pointa=(0.025, 0.08),  # left bottom
            pointb=(0.31, 0.08),  # right bottom
            style='modern'
        )

        self.plotter.add_checkbox_button_widget(
            self.saveBox,
            value=self.save,
            position=(25, 150),
            size=20, border_size=2,
            color_on='white',
            color_off='grey',
        )
        self.plotter.add_text('Save', position=(60, 147), color='white', shadow=True, font_size=8)

        self.plotter.add_checkbox_button_widget(
            self.channelBox,
            value=self.showChannel,
            position=(25, 200),
            size=20, border_size=2,
            color_on='white',
            color_off='grey',
        )
        self.plotter.add_text('Show channel', position=(60, 197), color='white', shadow=True, font_size=8)

        self.plotter.add_checkbox_button_widget(
            self.headBox,
            value=self.showHead,
            position=(25, 250),
            size=20, border_size=2,
            color_on='white',
            color_off='grey',
        )
        self.plotter.add_text('Show head', position=(60, 247), color='white', shadow=True, font_size=8)

        self.plotter.camera_position = 'xy'
        self.plotter.camera.zoom(0.8)

        self.channelActor = [self.plotter.add_mesh(ch) for ch in self.chs]
        self.plotter.add_mesh(self.head1, opacity=0.7, scalars=self.scalar, cmap=self.cmap)
        self.plotter.add_mesh(self.brain)

        self.plotter.show_bounds()

        if self.param['save']:
            self.plotter.save_graphic(f"event-{self.selected_event_name}_time-{self.param['timestamp']}_saliency3d.svg")

        return self.plotter


class CheckboxObj:
    def __init__(self, init_val):
        self.ctrl = init_val

    def __call__(self, state):
        self.ctrl = state