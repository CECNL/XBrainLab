import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull


class Saliency3D:
    def __init__(self, eval_record, epoch_data, selected_event_name):
        # set parameters
        self.selected_event_name = selected_event_name
        self.save = False
        self.showChannel = True
        self.showHead = True
        self.cmap = plt.cm.get_cmap("coolwarm")
        self.neighbor = 3

        # load 3d model
        mesh = pv.read('./XBrainLab/visualization/3Dmodel/head.ply')
        mesh2 = pv.read('./XBrainLab/visualization/3Dmodel/brain.ply')

        # get saliency
        labelIndex = epoch_data.event_id[self.selected_event_name]
        # [eval_record.label == labelIndex]
        self.saliency = eval_record.gradient[labelIndex]
        self.saliency = self.saliency.mean(axis=0)
        self.scalar_bar_range = [self.saliency.min(), self.saliency.max()]

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

        self.chs = [
            pv.Sphere(radius=0.005, center=self.pos_on_3d[i, :])
            for i in range(self.saliency.shape[0])
        ]

        # set plotter
        self.plotter = pv.Plotter(window_size=[750, 750])
        self.plotter.background_color = "lightslategray"
        self.head = mesh
        self.brain = mesh2.scale((0.001, 0.001, 0.001), inplace=False).triangulate()
        # self.head1 = self.head.clip_closed_surface(
        #     'z', origin=[0, 0, self.pos_on_3d[:, 2].min()]
        # ) # upper half head
        self.head1 = ChannelConvexHull(self.pos_on_3d)
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
            dist_idx = np.argsort(dist)[:self.neighbor] # id of #neighbor cloest points
            dist = np.array([dist[idx] for idx in dist_idx])
            self.scalar[i] = InverseDistWeightedSum(
                dist, self.saliency[dist_idx, self.param['timestamp'] - 1]
            )
        try:
            self.plotter.update_scalars(self.scalar, self.head1)
            self.plotter.update_scalar_bar_range(self.scalar_bar_range, '')
        except Exception:
            pass

        if self.channelActor != []:
            for actor in self.channelActor:
                actor.SetVisibility(self.channelBox.ctrl)
        if save:
            self.plotter.save_graphic(
                f"event-{self.selected_event_name}_"
                f"time-{self.param['timestamp']}_saliency3d.svg"
            )

        if self.headBox.ctrl:
            if self.headActor is None:
                self.headActor = self.plotter.add_mesh(
                    self.head, opacity=0.3, color='w'
                )
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
        self.plotter.add_text(
            'Save', position=(60, 147), color='white', shadow=True, font_size=8
        )

        self.plotter.add_checkbox_button_widget(
            self.channelBox,
            value=self.showChannel,
            position=(25, 200),
            size=20, border_size=2,
            color_on='white',
            color_off='grey',
        )
        self.plotter.add_text(
            'Show channel', position=(60, 197), color='white', shadow=True, font_size=8
        )

        self.plotter.add_checkbox_button_widget(
            self.headBox,
            value=self.showHead,
            position=(25, 250),
            size=20, border_size=2,
            color_on='white',
            color_off='grey',
        )
        self.plotter.add_text(
            'Show head', position=(60, 247), color='white', shadow=True, font_size=8
        )

        self.plotter.camera_position = 'xy'
        self.plotter.camera.zoom(0.8)

        self.channelActor = [self.plotter.add_mesh(ch, color='w') for ch in self.chs]
        self.plotter.add_mesh(
            self.head1, opacity=0.7,
            scalars=self.scalar, cmap=self.cmap, show_scalar_bar=False
        )
        self.plotter.add_scalar_bar('', interactive=False, vertical=False)
        self.plotter.update_scalar_bar_range(self.scalar_bar_range, '')
        self.plotter.add_mesh(self.brain, color='w')

        self.plotter.show_bounds()

        if self.param['save']:
            self.plotter.save_graphic(
                f"event-{self.selected_event_name}_"
                f"time-{self.param['timestamp']}_saliency3d.svg"
            )

        return self.plotter


class CheckboxObj:
    def __init__(self, init_val):
        self.ctrl = init_val

    def __call__(self, state):
        self.ctrl = state

def InverseDistWeightedSum(dist, val):
    assert len(dist) == len(val)
    dist = dist + 1e-12
    return np.sum(val/dist)/(np.sum([1/d for d in dist]))

def ChannelConvexHull(ch_pos):
    # faster than pyvista delaunay? :
    # https://gist.github.com/flutefreak7/bd621a9a836c8224e92305980ed829b9
    hull = ConvexHull(ch_pos)
    faces = np.hstack(
        (np.ones((len(hull.simplices), 1))*3, hull.simplices)
    ).astype(np.int32)
    poly = pv.PolyData(hull.points, faces.ravel())
    return poly
    # cloud = pv.PolyData(ch_pos)
    # return cloud.delaunay_3d()
