from __future__ import annotations


def _get_type_name(type_class: type) -> str:
    """Return the formatted name of a type."""
    return f"{type_class.__module__}.{type_class.__name__}"

def validate_type(instance: object,
                  type_class: type | tuple[type],
                  message_name: str) -> None:
    """Validate the type of an instance.

    Args:
        instance: The instance to be validated.
        type_class: Acceptable type(s) of the instance.
                    Can be a single type or a tuple of types.
        message_name: The name of the instance to be displayed in the error message.

    Raises:
        TypeError: If the instance is not an instance of the acceptable type(s).
    """
    if not isinstance(type_class, (list, tuple)):
        type_class = (type_class, )
    if not isinstance(instance, type_class):
        if len(type_class) == 1:
            type_class = type_class[0]
            type_name = _get_type_name(type_class)
        else:
            type_name_list = [_get_type_name(c) for c in type_class]
            type_name = ' or '.join(type_name_list)
        raise TypeError(
            f"{message_name} must be an instance of {type_name}, "
            f"got {type(instance)} instead.")

def validate_list_type(instance_list: list,
                       type_class: type | tuple[type],
                       message_name: str) -> None:
    """Validate the type of a list of instances.

    Args:
        instance_list: The list of instances to be validated.
        type_class: Acceptable type(s) of the instances.
                    Can be a single type or a tuple of types.
        message_name: The name of the instances to be displayed in the error message.

    Raises:
        TypeError:
            If any instance in the list is not an instance of the acceptable type(s).
    """
    validate_type(instance_list, list, message_name)
    for instance in instance_list:
        validate_type(instance, type_class, f"Items of {message_name}")

def validate_issubclass(class_name: type,
                        type_class: type | tuple[type],
                        message_name: str) -> None:
    """Validate if a class is a subclass of a type.

    Args:
        class_name: The class to be validated.
        type_class: Acceptable type of the class.
        message_name: The name of the class to be displayed in the error message.

    Raises:
        TypeError: If the class is not a subclass of the acceptable type.
    """
    if not isinstance(type_class, (list, tuple)):
        type_class = (type_class, )
    if not issubclass(class_name, type_class):
        if len(type_class) == 1:
            type_name = _get_type_name(type_class[0])
        else:
            type_name_list = [_get_type_name(c) for c in type_class]
            type_name = ' or '.join(type_name_list)

        raise TypeError(
            f"{message_name} must be an instance of {type_name}, "
            f"got {_get_type_name(class_name)} instead.")
