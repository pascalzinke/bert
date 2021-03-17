from utils.annotate import TextAnnotator
from utils.data import get_test_data

annotator = TextAnnotator()

text = get_test_data()

annotated_text = annotator.annotate(text)

print(annotated_text)
