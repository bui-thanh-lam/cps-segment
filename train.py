from transformers import SegformerForSemanticSegmentation, SegformerConfig

from dataset import SSLSegmentationDataset
from trainer import NCPSTrainer, SupervisedTrainer
from utils import TARGET_TRANSFORMS, TRAIN_TRANSFORMS


unlabelled_dataset = SSLSegmentationDataset(
    image_dir='datasets/SemiDataset25/unlabelled/image',
    transform=TRAIN_TRANSFORMS
)
labelled_dataset = SSLSegmentationDataset(
    image_dir='datasets/SemiDataset25/labelled/image',
    mask_dir='datasets/SemiDataset25/labelled/mask',
    transform=TRAIN_TRANSFORMS,
    target_transform=TARGET_TRANSFORMS,
    return_image_name=True
)
val_dataset = SSLSegmentationDataset(
    image_dir='datasets/TestDataset/CVC-300/images',
    mask_dir='datasets/TestDataset/CVC-300/masks',
    transform=TRAIN_TRANSFORMS,
    target_transform=TARGET_TRANSFORMS
)
full_labelled_dataset = SSLSegmentationDataset(
    image_dir='datasets/TrainDataset/image',
    mask_dir='datasets/TrainDataset/mask',
    transform=TRAIN_TRANSFORMS,
    target_transform=TARGET_TRANSFORMS
)
test_dataset = SSLSegmentationDataset(
    image_dir='datasets/TestDataset/CVC-300/images',
    mask_dir='datasets/TestDataset/CVC-300/masks',
    transform=TRAIN_TRANSFORMS,
    target_transform=TARGET_TRANSFORMS,
    return_image_name=True
)
config = SegformerConfig.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
config.num_labels = 2

print("==== Start training, supervised (smaller dataset) mode ====")
small_sup_trainer = SupervisedTrainer(model_config=config, n_epochs=10)
small_sup_trainer.fit(full_labelled_dataset, test_dataset=labelled_dataset)
print("Evaluate on CVC-300:")
print(small_sup_trainer.evaluate(val_dataset))
# small_sup_trainer.predict(test_dataset, "datasets/TestDataset/CVC-300/output/small/")

# print("==== Start training, fully supervised mode ====")
# full_sup_trainer = SupervisedTrainer(model_config=config, n_epochs=10)
# full_sup_trainer.fit(full_labelled_dataset)
# print("Evaluate on CVC-300:")
# print(full_sup_trainer.evaluate(val_dataset))
# full_sup_trainer.predict(test_dataset, "datasets/TestDataset/CVC-300/output/full/")

# print("==== Start training, semi supervised mode ====")
# semi_sup_trainer = NCPSTrainer(model_config=config, n_epochs=10)
# semi_sup_trainer.fit(labelled_dataset, unlabelled_dataset)
# print("Evaluate on CVC-300:")
# print(semi_sup_trainer.evaluate(val_dataset))
# semi_sup_trainer.predict(test_dataset, "datasets/TestDataset/CVC-300/output/semi/")
