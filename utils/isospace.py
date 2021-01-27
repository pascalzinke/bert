class IsoSpaceEntity:
    __types = [
        "PAD",
        "NONE",
        "PLACE",
        "PATH",
        "SPATIAL_ENTITY",
        "LOCATION",
        "MOTION",
        "SPATIAL_SIGNAL",
        "MOTION_SIGNAL",
        "MEASURE"
    ]

    __tag2label = {t: i for i, t in enumerate(__types)}

    def __init__(self, tag):
        self.label = (
            self.tag_to_label(tag.tag)
            if tag.tag in IsoSpaceEntity.__types
            else 0)
        self.id = tag.get("id")
        self.text = tag.get("text")
        start, end = tag.get("start"), tag.get("end")
        self.start = int(start) if start else None
        self.end = int(end) if end else None
        self.interval = (
            range(self.start, self.end)
            if self.start and self.end
            else [])

    @staticmethod
    def tag_to_label(tag):
        return next(
            label for label, t in enumerate(IsoSpaceEntity.__types) if tag == t)

    @staticmethod
    def label_to_tag(label):
        return IsoSpaceEntity.__types[label]

    @staticmethod
    def n_types():
        return len(IsoSpaceEntity.__types)

    @staticmethod
    def is_padding(label):
        return IsoSpaceEntity.label_to_tag(label) == "PAD"
