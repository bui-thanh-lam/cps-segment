import argparse
from dataset import SSLSegmentationDataset
from trainer import NCPSTrainer
from utils import *

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Train Segformer")
    parser.add_argument("--model_config", type=str, default="segformer_b0")
    parser.add_argument("--use_cutmix", type=bool, default=False)
    parser.add_argument("--pseudo_label_confidence_threshold", type=float, default=0.7)
    parser.add_argument("--use_multiple_teachers", type=bool, default=True)
    parser.add_argument("--prediction_mode", type=str, default="soft_voting")
    parser.add_argument("--labelled_image_dir", type=str, default="../datasets/SemiDataset25/labelled/image")
    parser.add_argument("--unlabelled_image_dir", type=str, default="../datasets/SemiDataset25/unlabelled/image")
    parser.add_argument("--mask_dir", type=str, default="../datasets/SemiDataset25/labelled/mask")
    parser.add_argument("--test_image_dir", type=str, default="../datasets/TestDataset/CVC-300/images")
    parser.add_argument("--test_mask_dir", type=str, default="../datasets/TestDataset/CVC-300/masks")
    parser.add_argument("--out_dir", type=str, default="../datasets/TestDataset/CVC-300/output/semi")
    args = parser.parse_args()

    _test_set = SSLSegmentationDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        feature_extractor_config=args.model_config,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
        return_image_name=True
    )
    test_set = SSLSegmentationDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        feature_extractor_config=args.model_config,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
    )
    unlabelled_dataset = SSLSegmentationDataset(
        image_dir=args.unlabelled_image_dir,
        feature_extractor_config=args.model_config,
        input_transform=TRAIN_INPUT_TRANSFORMS,
    )
    labelled_dataset = SSLSegmentationDataset(
        image_dir=args.labelled_image_dir,
        mask_dir=args.mask_dir,
        feature_extractor_config=args.model_config,
        input_transform=TRAIN_INPUT_TRANSFORMS,
        target_transform=TRAIN_TARGET_TRANSFORMS,
        shared_transform=TRAIN_SHARED_TRANSFORMS,
    )
    print("==== Start training, semi supervised mode ====")
    semi_sup_trainer = NCPSTrainer(
        model_config=args.model_config,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        n_labelled_examples_per_batch=BATCH_SIZE // 2,
        use_multiple_teachers=args.use_multiple_teachers,
        use_cutmix=args.use_cutmix,
        pseudo_label_confidence_threshold=args.pseudo_label_confidence_threshold,
    )
    semi_sup_trainer.fit(
        labelled_dataset,
        unlabelled_dataset,
        test_set
    )
    print("Evaluate on test set:")
    print(semi_sup_trainer.evaluate(test_set))
    semi_sup_trainer.predict(_test_set, args.out_dir, args.prediction_mode)
    semi_sup_trainer.save(args.out_dir)
