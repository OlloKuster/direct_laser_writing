def filter_loader(filter: str):
    if filter == "None":
        return lambda x: x
