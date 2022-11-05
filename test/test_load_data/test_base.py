from XBrainLab.load_data.base import LoadBase, DataType
from XBrainLab.load_data.base.base import EditRaw
from XBrainLab.dataset.data_holder.raw import Raw

import pytest
import tkinter as tk

@pytest.fixture(scope="function")
def base(root):
    window = LoadBase(root, "")
    yield window
    window.destroy()

class TestLoadBase:
    @pytest.mark.parametrize("status,out", [(False, 'active'), (True, 'disabled')])
    def test_reset(self, base, status, out):
        base.raw_data_list = [1]
        base.raw_data_len_var.set(100)
        base.lock_config_status = status
        base.event_ids_var.set('test_string')
        base.reset()
        assert base.type_raw['state'] == out
        assert base.type_epoch['state'] == out
        assert len(base.raw_data_list) == 0
        assert base.raw_data_len_var.get() == 0
        assert base.event_ids_var.get() == 'None'

    @pytest.mark.parametrize("raw_list_data_dict,raw_dict,msg", [
        (None, {'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, None),
        ({'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, {'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, None),
        ({'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, {'get_nchan':8,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, 'channel'),
        ({'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, {'get_nchan':9,'get_sfreq':8,'is_raw':True,'get_epoch_duration':9}, 'frequency'),
        ({'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, {'get_nchan':9,'get_sfreq':9,'is_raw':False,'get_epoch_duration':9}, 'type'),
        ({'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':9}, {'get_nchan':9,'get_sfreq':9,'is_raw':True,'get_epoch_duration':8}, None),
        ({'get_nchan':9,'get_sfreq':9,'is_raw':False,'get_epoch_duration':9}, {'get_nchan':9,'get_sfreq':9,'is_raw':False,'get_epoch_duration':9}, None),
        ({'get_nchan':9,'get_sfreq':9,'is_raw':False,'get_epoch_duration':9}, {'get_nchan':9,'get_sfreq':9,'is_raw':False,'get_epoch_duration':8}, 'duration'),
        ])
    def test_check_loaded_data_consistency_with_data(self, mocker, base, raw_list_data_dict, raw_dict, msg):
        raw = Raw(None, None)
        if raw_list_data_dict:
            raw_list_data = Raw(None, None)
            base.raw_data_list = [raw_list_data]
            for func, ret in raw_list_data_dict.items():
                mocker.patch.object(raw_list_data, func, return_value=ret)
                mocker.patch.object(raw, func, return_value=raw_dict[func])
        if msg:
            with pytest.raises(Exception) as e_info:
                base.check_loaded_data_consistency(raw)
            assert msg in str(e_info)
        else:
            base.check_loaded_data_consistency(raw)
            assert True
    
    @pytest.mark.parametrize("raw_data_list,filepath,ret", [
        (['test', 'test2'], 'test', True),
        (['test', 'test2'], 'error', False),
        ([], 'error', False),
    ])
    def test_get_loaded_raw(self, mocker, base, raw_data_list, filepath, ret):
        for raw_data in raw_data_list:
            base.raw_data_list.append(Raw(raw_data, None))
        assert (base.get_loaded_raw(filepath) is not None) == ret
    
    def test__load(self, base):
        with pytest.raises(NotImplementedError):
            base._load()

    @pytest.mark.parametrize("filename,pattern,target_subject,target_session", [
        ('A01_sess1_2.mat', '(?P<subject>.*)_(?P<session>.*)_.*', 'A01', 'sess1'),
        ('A01_sess1_2.mat', '(?P<subject>.*)_', 'A01_sess1', None),
        ('A01_sess1_2.mat', '.*_(?P<session>.*)_.*', None, 'sess1'),
        ('A01_sess1_2.mat', '(.*)_(.*).*', None, None),
        ('A01_sess1_2.mat', 'invalid.*', None, None)
        ]
    )
    def test__parse_filename(self, base, filename, pattern, target_subject, target_session):
        session, subject = base._parse_filename(filename, pattern)
        assert session == target_session
        assert subject == target_subject

    @pytest.mark.parametrize("selected_files", [['test'], []])
    @pytest.mark.parametrize("get_loaded_raw", [True, False])
    @pytest.mark.parametrize("data", [None, False, Raw('', None)])
    def test_load(self, base, mocker, mock_askopenfilenames, selected_files, get_loaded_raw, data):
        next(mock_askopenfilenames(selected_files))
        mocker.patch.object(base, 'get_loaded_raw', return_value=get_loaded_raw)
        mocker.patch.object(base, '_load', return_value=data)
        mocker.patch.object(base, 'check_loaded_data_consistency')
        if data is not None and data:
            mocker.patch.object(data, 'get_row_info')
        mocker.patch.object(base.data_attr_treeview, 'insert')
        mocker.patch.object(base, 'update_panel')
        if len(selected_files) == 0 or get_loaded_raw or data == False:
            base.load()
            assert len(base.raw_data_list) == 0
        else:
            if data is None:
                with pytest.raises(Exception) as e_info:
                    base.load()
                assert len(base.raw_data_list) == 0
                assert 'Unable' in str(e_info)
            else:
                base.load()
                assert len(base.raw_data_list) == 1
                assert base.data_attr_treeview.insert.call_count == 1
        assert base.update_panel.call_count == 1

    @pytest.mark.parametrize("raw_data_list", [[True], []])
    @pytest.mark.parametrize("ori_type",  [d.value for d in DataType])
    @pytest.mark.parametrize("data_type", [d.value for d in DataType])
    def test_check_data_type(self, base, tk_warning, raw_data_list, ori_type, data_type):
        base.type_ctrl.set(ori_type)
        base.raw_data_list = raw_data_list
        if raw_data_list and ori_type != data_type:
            with pytest.raises(Exception) as e_info:
                base.check_data_type(data_type)
            assert 'at the same time' in str(e_info)
        else:
            base.check_data_type(data_type)
            if ori_type == data_type:
                assert len(tk_warning) == 0
            else:
                assert len(tk_warning) > 0
            assert base.type_ctrl.get() == data_type
    
    @pytest.mark.parametrize("raw_data_list", [[Raw(None, None)], []])
    @pytest.mark.parametrize("lock_config_status", [True, False])
    @pytest.mark.parametrize("event_list", [['a', 'b', 'c'], []])
    def test_update_panel(self, base, mocker, raw_data_list, lock_config_status, event_list):
        ori_len = len(raw_data_list)
        if ori_len:
            mocker.patch.object(raw_data_list[0], 'get_event_list', return_value=(None, event_list))
        mocker.patch.object(base, 'reset')
        base.raw_data_list = raw_data_list
        base.lock_config_status = lock_config_status
        base.update_panel()
        if ori_len == 0:
            assert base.reset.call_count == 1
        else:
            assert base.reset.call_count == 0
            assert base.raw_data_len_var.get() == ori_len
            if event_list and ori_len:
                assert base.event_ids_var.get() == 'a\nb\nc'
            else:
                assert base.event_ids_var.get() == 'None'
    
    @pytest.mark.parametrize("raw_data", [None, Raw(None, None)])
    @pytest.mark.parametrize("del_row", [True, False])
    def test_edit(self, mocker, base, raw_data, del_row):
        if raw_data:
            base.raw_data_list.append(raw_data)
            mocker.patch.object(raw_data, 'get_row_info')

        mocker.patch.object(base, 'get_loaded_raw', return_value=raw_data)
        mocker.patch.object(base.data_attr_treeview, 'delete')
        mocker.patch.object(base.data_attr_treeview, 'item')
        mocker.patch.object(base, 'update_panel')
        
        mocker.patch.object(EditRaw, '__init__', return_value=None)
        mocker.patch.object(EditRaw, 'get_result', return_value=del_row)
        base.edit(None)
        if raw_data:
            assert EditRaw.__init__.call_count == 1
            if del_row:
                assert len(base.raw_data_list) == 0
                assert base.data_attr_treeview.delete.call_count == 1
                assert base.data_attr_treeview.item.call_count == 0
            else:
                assert len(base.raw_data_list) == 1
                assert base.data_attr_treeview.delete.call_count == 0
                assert base.data_attr_treeview.item.call_count == 1
            assert base.update_panel.call_count == 1
        else:
            assert EditRaw.__init__.call_count == 0
            assert len(base.raw_data_list) == 0
            assert base.data_attr_treeview.delete.call_count == 0
            assert base.data_attr_treeview.item.call_count == 0
            assert base.update_panel.call_count == 0

    @pytest.mark.parametrize("raw_data_list", [[], [Raw('test', None)]])
    @pytest.mark.parametrize("event_list", [['a', 'b', 'c'], []])
    def test_confirm(self, base, mocker, raw_data_list, event_list):
        if raw_data_list:
            mocker.patch.object(raw_data_list[0], 'get_event_list', return_value=(None, event_list))
        base.raw_data_list = raw_data_list
        if len(raw_data_list) == 0:
            with pytest.raises(Exception) as e_info:
                base.confirm()
            assert 'loaded' in str(e_info)
        else:
            if event_list:
                base.confirm()
                assert base.ret_val == raw_data_list
            else:
                with pytest.raises(Exception) as e_info:
                    base.confirm()
                assert 'No label' in str(e_info)
