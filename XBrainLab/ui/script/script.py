import time


def get_id():
    return time.time()

class Script:
    def __init__(self):
        self.reset()

    def reset(self):
        self.import_list = set()
        self.command_list = []
        self.ui_command_list = []

    def add_cmd(self, data, newline=False):
        if newline:
            self.newline()
        self.command_list.append((get_id(), data))

    def add_ui_cmd(self, data, newline=False):
        if newline:
            self.ui_newline()
        self.ui_command_list.append((get_id(), data))

    def add_script(self, script):
        if script is None:
            return
        self.import_list.update(script.import_list)
        self.command_list += script.command_list
        self.ui_command_list += script.ui_command_list

    def set_cmd(self, data):
        self.clear_cmd()
        self.add_cmd(data)

    def clear_cmd(self):
        self.command_list.clear()

    def newline(self):
        self.command_list.append((get_id(), ""))

    def ui_newline(self):
        self.ui_command_list.append((get_id(), ""))

    def add_import(self, data):
        self.import_list.add(data)

    def __iadd__(self, lhs):
        if not lhs:
            return self
        self.import_list.update(lhs.import_list)

        if lhs.command_list:
            self.newline()
        self.command_list += lhs.command_list
        self.ui_command_list += lhs.ui_command_list
        return self

    def get_str(self):
        import_str = '\n'.join(self.import_list)
        # self.command_list.sort(key=lambda x: x[0])
        command_str = '\n'.join([x[1] for x in self.command_list])
        return import_str + '\n\n' + command_str

    def get_ui_str(self):
        import_str = '\n'.join(self.import_list)
        self.ui_command_list.sort(key=lambda x: x[0])
        ui_command_str = '\n'.join([x[1] for x in self.ui_command_list])
        return import_str + '\n\n' + ui_command_str

    def get_all_str(self):
        all_list = self.ui_command_list + self.command_list
        all_list.sort(key=lambda x: x[0])

        import_str = '\n'.join(self.import_list)
        all_list_str = '\n'.join([x[1] for x in all_list])
        return import_str + '\n\n' + all_list_str
