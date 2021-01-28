PAD = "PAD"
NONE = "NONE"


class Attribute:
    def __init__(self, index, name, values):
        self.index = index
        self.name = name
        self.__values = [PAD, NONE] + values
        self.__value_dict = {value: i for i, value in enumerate(self.__values)}
        self.n = len(self.__values)
        self.none = self.encode(NONE)
        self.pad = self.encode(PAD)

    def encode(self, value):
        value = value if value in self.__values else NONE
        return self.__value_dict[value]

    def decode(self, i):
        return self.__values[i]

    def is_padding(self, i):
        return self.decode(i) == PAD


Tag = Attribute(1, "tag", [
    "PLACE",
    "PATH",
    "SPATIAL_ENTITY",
    "LOCATION",
    "MOTION",
    "SPATIAL_SIGNAL",
    "MOTION_SIGNAL",
    "MEASURE"
])

Dimensionality = Attribute(2, "dimensionality", [
    "POINT",
    "AREA",
    "VOLUME"
])

Form = Attribute(3, "form", [
    "NOM",
    "NAM"
])

SemanticType = Attribute(4, "semantic type", [
    "TOPOLOGICAL",
    "DIRECTIONAL",
    "DIR_TOP"
])

MotionType = Attribute(5, "motion type", [
    "PATH",
    "COMPOUND",
    "MANNER"
])

MotionClass = Attribute(6, "motion class", [
    "REACH",
    "CROSS",
    "MOVE",
    "MOVE_INTERNAL",
    "MOVE_EXTERNAL",
    "FOLLOW",
    "DEVIATE"
])


class Entity:
    def __init__(self, tag):
        if tag is None:
            self.tag = Tag.none
            self.dimensionality = Dimensionality.none
            self.form = Form.none
            self.semantic_type = SemanticType.none
            self.motion_type = MotionType.none
            self.motion_class = MotionClass.none

        else:
            self.id = tag.get("id")
            self.text = tag.get("text")

            # extract training labels
            self.tag = Tag.encode(tag.tag)
            self.dimensionality = Dimensionality.encode(
                tag.get("dimensionality"))
            self.form = Form.encode(tag.get("form"))
            self.semantic_type = SemanticType.encode(tag.get("semantic_type"))
            self.motion_type = MotionType.encode(tag.get("motion_type"))
            self.motion_class = MotionClass.encode(tag.get("motion_class"))

            # Used for mapping NLP token to isospace entities
            start, end = tag.get("start"), tag.get("end")
            self.start = int(start) if start else None
            self.end = int(end) if end else None
            self.interval = (
                range(self.start, self.end)
                if self.start and self.end
                else [])
