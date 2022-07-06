from transformers import SegformerForSemanticSegmentation


def huggingface_segformer(config, use_imagenet_pretrained=True):
    model = SegformerForSemanticSegmentation(config=config)
    if use_imagenet_pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=config.num_labels)
    return model