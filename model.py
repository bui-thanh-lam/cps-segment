from transformers import SegformerForSemanticSegmentation, SegformerConfig


model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")
print(model)