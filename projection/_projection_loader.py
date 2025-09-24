def projection_loader(projection: str, *args):
    if projection == "None":
        return lambda x: x