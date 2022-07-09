from transformers import SegformerForSemanticSegmentation


def segformer_b0():
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2)
    return model


def segformer_b1():
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1", num_labels=2)
    return model


def segformer_b2():
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=2)
    return model


def segformer_b3():
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=2)
    return model
