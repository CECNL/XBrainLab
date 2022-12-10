def validate_type(instance, type_class, message_name):
    if not isinstance(type_class, (list, tuple)):
        type_class = (type_class, )
    if not isinstance(instance, type_class):
        if len(type_class) == 1:
            type_class = type_class[0]
            type_name = f"{type_class.__module__}.{type_class.__name__}"
        else:
            type_name_list = []
            for c in type_class:
                type_name_list.append(f"{c.__module__}.{c.__name__}")
            type_name = ' or '.join(type_name_list)
        raise TypeError(
            f"{message_name} must be an instance of {type_name}, "
            f"got {type(instance)} instead.")

def validate_list_type(instance_list, type_class, message_name):
    validate_type(instance_list, list, message_name)
    for instance in instance_list:
        validate_type(instance, type_class, f"Items of {message_name}")

def validate_issubclass(class_name, type_class, message_name):
    if not issubclass(class_name, type_class):
        type_name = f"{type_class.__module__}.{type_class.__name__}"
        raise TypeError(
            f"{message_name} must be an instance of {type_name}, "
            f"got {class_name.__module__}.{class_name.__name__} instead.")
