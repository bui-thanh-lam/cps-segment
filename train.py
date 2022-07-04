from transformers import SegformerForSemanticSegmentation, SegformerConfig

from dataset import SSLSegmentationDataset
from trainer import NCPSTrainer, SupervisedTrainer
from utils import *


unlabelled_dataset = SSLSegmentationDataset(
    image_dir='datasets/SemiDataset25/unlabelled/image',
    input_transform=TRAIN_INPUT_TRANSFORMS,
    shared_transform=TRAIN_SHARED_TRANSFORMS
)
labelled_dataset = SSLSegmentationDataset(
    image_dir='datasets/SemiDataset25/labelled/image',
    mask_dir='datasets/SemiDataset25/labelled/mask',
    input_transform=TRAIN_INPUT_TRANSFORMS,
    target_transform=TRAIN_TARGET_TRANSFORMS,
    shared_transform=TRAIN_SHARED_TRANSFORMS,
)
full_labelled_dataset = SSLSegmentationDataset(
    image_dir='datasets/TrainDataset/image',
    mask_dir='datasets/TrainDataset/mask',
    input_transform=TRAIN_INPUT_TRANSFORMS,
    target_transform=TRAIN_TARGET_TRANSFORMS,
    shared_transform=TRAIN_SHARED_TRANSFORMS
)
val_dataset = SSLSegmentationDataset(
    image_dir='datasets/TestDataset/CVC-300/images',
    mask_dir='datasets/TestDataset/CVC-300/masks',
    input_transform=VAL_INPUT_TRANSFORMS,
    shared_transform=VAL_SHARED_TRANSFORMS
)
cvc_300 = SSLSegmentationDataset(
    image_dir='datasets/TestDataset/CVC-300/images',
    mask_dir='datasets/TestDataset/CVC-300/masks',
    input_transform=VAL_INPUT_TRANSFORMS,
    shared_transform=VAL_SHARED_TRANSFORMS,
    return_image_name=True
)
config = SegformerConfig.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
config.num_labels = 2

# print("==== Start training, supervised (smaller dataset) mode ====")
# small_sup_trainer = SupervisedTrainer(
#     model_config=config, 
#     n_epochs=N_EPOCHS,
#     learning_rate=LEARNING_RATE,
#     batch_size=BATCH_SIZE
# )
# small_sup_trainer.fit(
#     labelled_dataset, 
#     val_dataset=val_dataset,
# )
# print("Evaluate on CVC-300:")
# print(small_sup_trainer.evaluate(val_dataset))
# small_sup_trainer.predict(test_dataset, "datasets/TestDataset/CVC-300/output/small/")
# small_sup_trainer.save("datasets/TestDataset/CVC-300/output/small/")

print("==== Start training, fully supervised mode ====")
full_sup_trainer = SupervisedTrainer(
    model_config=config, 
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE    
)
full_sup_trainer.fit(
    full_labelled_dataset,
    val_dataset=val_dataset
)
print("Evaluate on CVC-300:")
print(full_sup_trainer.evaluate(val_dataset))
full_sup_trainer.predict(cvc_300, "datasets/TestDataset/CVC-300/output/full/")
full_sup_trainer.save("datasets/TestDataset/CVC-300/output/full/")

# print("==== Start training, semi supervised mode ====")
# semi_sup_trainer = NCPSTrainer(
#     model_config=config, 
#     n_epochs=N_EPOCHS,
#     learning_rate=LEARNING_RATE,
#     batch_size=BATCH_SIZE    
# )
# semi_sup_trainer.fit(
#     labelled_dataset, 
#     unlabelled_dataset, 
#     val_dataset=val_dataset,
# )
# print("Evaluate on CVC-300:")
# print(semi_sup_trainer.evaluate(val_dataset))
# semi_sup_trainer.predict(test_dataset, "datasets/TestDataset/CVC-300/output/semi/")
# semi_sup_trainer.save("datasets/TestDataset/CVC-300/output/semi/")
