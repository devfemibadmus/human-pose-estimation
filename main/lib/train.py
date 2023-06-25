import argparse, time, os
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.keras import backend as K

# from files
from constants import *
from hourglass import HourglassNet
from util import *


def tensorflow_setup():
    print(f"TensorFlow detected the following GPU(s): {tf.config.list_physical_devices('GPU')}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def process_args():
    argparser = argparse.ArgumentParser(description='Training parameters')
    argparser.add_argument('-m',
                        '--model-save',
                        default=DEFAULT_MODEL_BASE_DIR,
                        help='base directory for saving model weights')
    argparser.add_argument('-e',
                        '--epochs',
                        default=DEFAULT_EPOCHS,
                        type=int,
                        help='number of epochs')
    argparser.add_argument('-b',
                        '--batch',
                        type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help='batch size')
    argparser.add_argument('-hg',
                        '--hourglass',
                        type=int,
                        default=DEFAULT_NUM_HG,
                        help='number of hourglass blocks')
    argparser.add_argument('-sub',
                        '--subset',
                        type=float,
                        default=1.0,
                        help='fraction of train set to train on, default 1.0')
    argparser.add_argument('-l',
                        '--loss',
                        default=DEFAULT_LOSS,
                        help='Loss function for model training')
    argparser.add_argument('-a',
                        '--augment',
                        default=DEFAULT_AUGMENT,
                        help='Strength of image augmentation')
    argparser.add_argument('--optimizer',
                        default=DEFAULT_OPTIMIZER,
                        help='name of optimizer to use')
    argparser.add_argument('--learning-rate',
                        type=float,
                        default=DEFAULT_LEARNING_RATE,
                        help='learning rate of optimizer')
    argparser.add_argument('--activation',
                        default=DEFAULT_ACTIVATION,
                        help='activation for output layer')
    # Resume model training arguments
    # TODO make a consistent way to generate and retrieve epoch checkpoint filenames
    argparser.add_argument('-r',
                        '--resume',
                        default=False,
                        type=bool,
                        help='resume training')
    argparser.add_argument('--resume-json',
                        default=None,
                        help='Model architecture for re-loading weights to resume training')
    argparser.add_argument('--resume-weights',
                        default=None,
                        help='Model weights file to resume training')
    argparser.add_argument('--resume-epoch',
                        default=None,
                        type=int,
                        help='Epoch to resume training')
    argparser.add_argument('--resume-subdir',
                        default=None,
                        help='Subdirectory containing architecture json and weights')
    argparser.add_argument('--resume-with-new-run',
                        type=bool,
                        default=False,
                        help='start a new session ID on resume. Default will be true if resume epoch is not the latest weight file.')
    # Misc
    argparser.add_argument('--notes',
                        default=None,
                        help='Any notes to save with the model path. Prefer no spaces')

    # Convert string arguments to appropriate type
    args = argparser.parse_args()

    # Validate arguments
    assert (args.subset > 0 and args.subset <= 1.0), "Subset must be fraction between 0 and 1.0"

    if args.resume:
        # Automatically locate architecture json and model weights
        if args.resume_subdir is not None:
            args.resume_json, args.resume_weights, args.resume_epoch = find_resume_json_weights_str(args.model_save, args.resume_subdir, args.resume_epoch)

        # If we are not resuming from the highest epoch in that subdir, start a new run
        # This is because Tensorboard does not overwrite epoch information on resume,
        # which may cause the graph to no longer be single-valued.
        # See https://github.com/tensorflow/tensorboard/issues/3732
        if not args.resume_with_new_run:
            args.resume_with_new_run = not is_highest_epoch_file(args.model_save, args.resume_subdir, args.resume_epoch)

        assert args.resume_json is not None and args.resume_weights is not None, \
            "Resume model training enabled, but no parameters received for: --resume-subdir, or both --resume-json and --resume-weights"


    if args.notes is not None:
        # Clean notes so it can be used in directory name
        args.notes = slugify(args.notes)

    # validate enum args
    assert validate_enum(LossFunctionOptions, args.loss)
    assert validate_enum(ImageAugmentationStrength, args.augment)
    assert validate_enum(OptimizerType, args.optimizer)
    assert validate_enum(OutputActivation, args.activation)

    return args


if __name__ == "__main__":
    args = process_args()

    print(f"\n\nSetup start: {time.ctime()}\n")
    setup_start = time.time()

    tensorflow_setup()

    hgnet = HourglassNet(
        num_classes=NUM_COCO_KEYPOINTS,
        num_stacks=args.hourglass,
        num_channels=NUM_CHANNELS,
        inres=INPUT_DIM,
        outres=OUTPUT_DIM,
        loss_str=args.loss,
        image_aug_str=args.augment,
        optimizer_str=args.optimizer,
        learning_rate=args.learning_rate,
        activation_str=args.activation
    )

    training_start = time.time()

    # TODO Save all model parameters in JSON for easy resuming and parsing later on
    if args.resume:
        print(f"\n\nResume training start: {time.ctime()}\n")

        hgnet.resume_train(args.batch, args.model_save, args.resume_json, args.resume_weights, \
            args.resume_epoch, args.epochs, args.resume_subdir, args.subset, new_run=args.resume_with_new_run)
    else:
        hgnet.build_model(show=True)

        print(f"\n\nTraining start: {time.ctime()}\n")
        print(f"Hourglass blocks: {args.hourglass:2d}, epochs: {args.epochs:3d}, batch size: {args.batch:2d}, subset: {args.subset:.2f}")

        hgnet.train(args.batch, args.model_save, args.epochs, args.subset, args.notes)

    print(f"\n\nTraining end: {time.ctime()}\n")

    training_end = time.time()

    setup_time = training_start - setup_start
    training_time = training_end - training_start

    print(f"Total setup time: {str(timedelta(seconds=setup_time))}")
    print(f"Total train time: {str(timedelta(seconds=training_time))}")
    print(f"Hourglass blocks: {args.hourglass:2d}, epochs: {args.epochs:3d}, batch size: {args.batch:2d}, subset: {args.subset:.2f}")
