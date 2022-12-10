class Script:
    def __init__(self):
        self.import_list = set()
        self.command_list = []

    def add_cmd(self, data):
        self.command_list.append(data)

    def add_script(self, script):
        self.import_list.update(script.import_list)
        self.command_list += script.command_list

    def set_cmd(self, data):
        self.clear_cmd()
        self.add_cmd(data)

    def clear_cmd(self):
        self.command_list.clear()
    
    def newline(self, pseudo=True):
        if pseudo:
            self.command_list.append("\n")
        else:
            self.command_list.append("")

    def add_import(self, data):
        self.import_list.add(data)

    def __iadd__(self, lhs):
        if not lhs:
            return self
        self.import_list.update(lhs.import_list)
        
        if lhs.command_list:
            self.newline(pseudo=False)
        self.command_list += lhs.command_list
        if lhs.command_list:
            self.newline(pseudo=False)
        return self

    def get_str(self):
        import_str = '\n'.join(self.import_list)
        command_str = '\n'.join(self.command_list)
        return import_str + '\n\n' + command_str