from utils.annotate import TextAnnotator
from utils.data import get_test_data

annotator = TextAnnotator()

text = get_test_data()

text1 = annotator.annotate(text)
text2 = annotator.annotate("Peter runs to the house across the street.")

print(text1)
print(text2)
