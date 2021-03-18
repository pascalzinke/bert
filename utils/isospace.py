PAD = "PAD"
NONE = "NONE"


class Attribute:
    """
    Helper class for creating attributes with a list of possible values
    Values can be encoded to integers for usage with bert
    Values "PAD" and "NONE" are automatically added
    """
    def __init__(self, name, values):
        self.id = name
        self.values = [PAD, NONE] + values
        self.__value_dict = {value: i for i, value in enumerate(self.values)}
        self.num_values = len(self.values)
        self.none = self.encode(NONE)
        self.pad = self.encode(PAD)
        self.encoded_values = [self.encode(v) for v in self.values]

    def encode(self, value):
        # Encodes attribute value from string to integer
        value = value if value in self.values else NONE
        return self.__value_dict[value]

    def decode(self, i):
        # Decodes attribute value from integer to string
        return self.values[i]

    def is_padding(self, i):
        return self.decode(i) == PAD


# Initialize IsoSpace attributes
SpatialElement = Attribute("spatial_element", [
    "PLACE",
    "PATH",
    "SPATIAL_ENTITY",
    "MOTION",
    "SPATIAL_SIGNAL",
    "MOTION_SIGNAL",
    "MEASURE"
])
Dimensionality = Attribute("dimensionality", [
    "POINT",
    "AREA",
    "VOLUME"
])
Form = Attribute("form", [
    "NOM",
    "NAM"
])
SemanticType = Attribute("semantic_type", [
    "TOPOLOGICAL",
    "DIRECTIONAL",
    "DIR_TOP"
])
MotionType = Attribute("motion_type", [
    "PATH",
    "COMPOUND",
    "MANNER"
])
MotionClass = Attribute("motion_class", [
    "REACH",
    "CROSS",
    "MOVE",
    "MOVE_INTERNAL",
    "MOVE_EXTERNAL",
    "FOLLOW",
    "DEVIATE"
])


class IsoSpaceElement:
    """
    Helper class for creating IsoSpace Elements from XML tags
    """
    def __init__(self, tag):
        # Create empty entity if no
        if tag is None:
            self.spatial_element = SpatialElement.none
            self.dimensionality = Dimensionality.none
            self.form = Form.none
            self.semantic_type = SemanticType.none
            self.motion_type = MotionType.none
            self.motion_class = MotionClass.none

        else:
            self.id = tag.get("id")
            self.text = tag.get("text")

            # Extract Spatial Element type and attributes
            self.spatial_element = SpatialElement.encode(tag.tag)
            self.dimensionality = Dimensionality.encode(
                tag.get("dimensionality"))
            self.form = Form.encode(tag.get("form"))
            self.semantic_type = SemanticType.encode(tag.get("semantic_type"))
            self.motion_type = MotionType.encode(tag.get("motion_type"))
            self.motion_class = MotionClass.encode(tag.get("motion_class"))

            # Used for mapping tokens to IsoSpace Elements
            start, end = tag.get("start"), tag.get("end")
            self.start = int(start) if start else None
            self.end = int(end) if end else None
            self.interval = (
                range(self.start, self.end)
                if self.start and self.end
                else [])
