import difflib


def _name_getter(dictionary: dict, previous_name, previous_names):
    for k, v in dictionary.items():
        if previous_name == "":
            previous_names.append(k)
        else:
            previous_names.append(str(previous_name) + "." + str(k))
    for k, v in dictionary.items():
        if isinstance(v, dict):
            _name_getter(
                v,
                str(k) if previous_name == "" else str(previous_name) + "." + str(k),
                previous_names,
            )


def merge_checker(base_dict, incoming_dict):
    base_names = []
    _name_getter(base_dict, "", base_names)
    incom_names = []
    _name_getter(incoming_dict, "", incom_names)
    undesired_attributes = sorted(set(incom_names) - set(base_names))

    def create_proposal(unwanted_string: str):
        return difflib.get_close_matches(unwanted_string, base_names, n=1)[0]

    if len(undesired_attributes) > 0:
        raise RuntimeError(
            f"\nUnwanted attributed identified compared with base config: \t"
            f"{', '.join([f'`{x}`: (possibly `{create_proposal(x)}`)' for x in undesired_attributes])}"
        )


if __name__ == "__main__":
    base = {1: {"a": 1, "b": 2}, 2: ["C", "D"]}
    inc_dict = {1: {"a": "replace", "ef": 2}}
    merge_checker(base, incoming_dict=inc_dict)
