from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torchinfo


def _segformer(config_name, load_pretrained=True):
    if load_pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(config_name, num_labels=2)
    else:
        config = SegformerConfig.from_pretrained(config_name)
        config.num_labels = 2
        model = SegformerForSemanticSegmentation(config)
    return model


def segformer_b0(load_pretrained):
    return _segformer(config_name="nvidia/mit-b0", load_pretrained=load_pretrained)


def segformer_b1(load_pretrained):
    return _segformer(config_name="nvidia/mit-b1", load_pretrained=load_pretrained)


def segformer_b2(load_pretrained):
    return _segformer(config_name="nvidia/mit-b2", load_pretrained=load_pretrained)


def segformer_b3(load_pretrained):
    return _segformer(config_name="nvidia/mit-b3", load_pretrained=load_pretrained)
