#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import dataclasses
import typing as typ

T = typ.TypeVar("T")  # Generic type for any dataclass

def fill_dataclass(
        flat_dict: typ.Dict,
        class_name: typ.Type[T]
) -> T:
    """
    Fills dataclass extracting common arguments coming from a dict
    """
    if not dataclasses.is_dataclass(class_name):
        raise ValueError(f"{class_name} is not a dataclass!")
    
    field_values = {}
    
    # NB: assumes all fields in a flat dictionary, and no overlap of fields btw nested dataclasses
    for field in dataclasses.fields(class_name):
        if field.name in flat_dict:
            field_values[field.name] = flat_dict[field.name]
        elif dataclasses.is_dataclass(field.type):
            nested_field_names = {f.name for f in dataclasses.fields(field.type)}
            nested_dict = {k: v for k, v in flat_dict.items() if k in nested_field_names}
            # Recursively construct the nested dataclass if relevant fields exist
            if nested_dict:
                field_values[field.name] = fill_dataclass(nested_dict, field.type)
    
    return class_name(**field_values)
