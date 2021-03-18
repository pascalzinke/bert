# ISOBERT

### What is ISOBERT?

ISOBERT is a IsoSpace Text Annotator based on the BERT Transformer Model. It
uses a custom BERT Model to classify *Spatial Elements* and can also classify
attributes of *Spatial Elements*.

### How to train

An example of how to train the model can be found in the `train.py` file. All
IsoSpace attributes can be trained and fine-tuned separately. The constructor of
the  `Trainer` Class expects an IsoSpace attribute and a `TrainerConfig`, which
consists of a device to train on (GPU or CPU), and a Train- and Eval-
`DataLoader`. The `Trainer` Class also has some optional parameters for
fine-tuning. The model can now be trained with the `train()` method. The model
is automatically saved after training.

Since *Spatial Elements* are classified differently from attributes, the
`keep_none` flag needs to be set here.

### How to annotate

An example on how to annotate text can be found in the `annotate.py` file. The
`TextAnnotater` class accepts a list of attributes to annotate. By default, all
attributes are annotated. Since *Spatial Elements* are required for attribute
classification, they will always be annotated and do **not** need to be
specified. The `annotate()` method expects a text and returns an
`AnnotatedText`, which has a list of sentences containing annotated words. Each
`AnnotatedWord` has a text, the *Spatial Elements* and a dictionary of
attributes and their values.