import json
from absl import app
from absl import flags
import matplotlib.pyplot as plt

from data_loader_camus import DataLoaderCamus
from patch_gan import PatchGAN
from utils import set_backend

flags.DEFINE_string('dataset_path', None, 'Path of the dataset.')
flags.DEFINE_string('gpu', '0', 'Comma separated list of GPU cores to use for training.')
flags.DEFINE_boolean('test', False, 'Test model and generate outputs on the test set')
flags.DEFINE_string('config', None, 'Config file for training hyper-parameters.')
flags.DEFINE_boolean('use_wandb', False, 'Use wandb for logging')
flags.DEFINE_string('wandb_resume_id', None, 'Resume wandb process with the given id')
flags.DEFINE_string('ckpt_load', None, 'Path to load the model')
flags.DEFINE_float('train_ratio', 0.95,
                   'Ratio of training data used for training and the rest used for testing. Set this value to 1.0 if '
                   'the data in the test folder are to be used for testing.')
flags.DEFINE_float('valid_ratio', 0.02, 'Ratio of training data used for validation')
flags.mark_flag_as_required('dataset_path')
flags.mark_flag_as_required('config')

FLAGS = flags.FLAGS

plt.switch_backend('agg')


def main(argv):
    # Load configs from file
    config = json.load(open(FLAGS.config))
    set_backend(FLAGS.gpu)

    # Set name
    name = '{}_{}_'.format(config['INPUT_NAME'], config['TARGET_NAME'])
    for l in config['LABELS']:
        name += str(l)
    config['NAME'] += '_' + name

    # Organize augmentation hyper-parameters from config
    augmentation = dict()
    for key, value in config.items():
        if 'AUG_' in key:
            augmentation[key] = value

    # Initialize data loader
    data_loader = DataLoaderCamus(
        dataset_path=FLAGS.dataset_path,
        input_name=config['INPUT_NAME'],
        target_name=config['TARGET_NAME'],
        condition_name=config['CONDITION_NAME'],
        img_res=config['IMAGE_RES'],
        target_rescale=config['TARGET_TRANS'],
        input_rescale=config['INPUT_TRANS'],
        condition_rescale=config['CONDITION_TRANS'],
        labels=config['LABELS'],
        train_ratio=FLAGS.train_ratio,
        valid_ratio=FLAGS.valid_ratio,
        augment=augmentation
    )

    if FLAGS.use_wandb:
        import wandb
        resume_wandb = True if FLAGS.wandb_resume_id is not None else False
        wandb.init(config=config, resume=resume_wandb, id=FLAGS.wandb_resume_id, project='EchoGen')

    # Initialize GAN
    model = PatchGAN(data_loader, config, FLAGS.use_wandb)

    # load trained models if they exist
    if FLAGS.ckpt_load is not None:
        model.load_model(FLAGS.ckpt_load)

    if FLAGS.test:
        model.test()
    else:
        model.train()


if __name__ == '__main__':
    app.run(main)
