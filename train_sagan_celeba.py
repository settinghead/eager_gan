import tensorflow as tf
tf.enable_eager_execution()
from utils import tf_record_parser, normalizer
from utils import save_model
from tqdm import tqdm

print(tf.__version__)

tfe = tf.contrib.eager

import os
from celeb.model import Generator, Discriminator
from libs.loss import discriminator_loss, generator_loss

flags = tf.app.flags
flags.DEFINE_integer(name='z_size', default=128,
                     help="Input random vector dimension")
flags.DEFINE_float(name='learning_rate_generator', default=0.0001,
                   help="Learning rate for the generator net")
flags.DEFINE_float(name='learning_rate_discriminator', default=0.0004,
                   help="Learning rate for the discriminator net")
flags.DEFINE_integer(name='batch_size', default=128,
                     help="Size of the input batch")
flags.DEFINE_float(name='alpha', default=0.1,
                   help="Leaky ReLU negative slope")
flags.DEFINE_float(name='beta1', default=0.0,
                   help="Adam optimizer beta1")
flags.DEFINE_float(name='beta2', default=0.9,
                   help="Adam optimizer beta2")
flags.DEFINE_integer(name='total_train_steps', default=600000,
                     help="Total number of training steps")
flags.DEFINE_string(name='dtype', default="float32",
                    help="Training Float-point precision")
flags.DEFINE_integer(name='record_summary_after_n_steps', default=200,
                     help="Number of interval steps to recording summaries")
flags.DEFINE_integer(name='number_of_test_images', default=16,
                     help="Number of test images to generate during evaluation")
flags.DEFINE_string(name='model_id', default="no_spectral_norm",
                    help="Load this model if found")

REPORT_INTERVAL = 51200
train_dataset = tf.data.TFRecordDataset(["./dataset/celeba.tfrecord"])
train_dataset = train_dataset.map(tf_record_parser, num_parallel_calls=8)
train_dataset = train_dataset.map(
    lambda image: normalizer(
        image, dtype=flags.FLAGS.dtype), num_parallel_calls=8
)
train_dataset = train_dataset.shuffle(REPORT_INTERVAL)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(flags.FLAGS.batch_size)


generator_net = Generator(
    dtype=flags.FLAGS.dtype
)
discriminator_net = Discriminator(
    alpha=flags.FLAGS.alpha, dtype=flags.FLAGS.dtype
)

basepath = "./models/" + flags.FLAGS.model_id
logdir = os.path.join(basepath, "logs")
print("Base folder:", basepath)
tf_board_writer = tf.contrib.summary.create_file_writer(logdir)
tf_board_writer.set_as_default()

generator_optimizer = tf.train.AdamOptimizer(
    learning_rate=flags.FLAGS.learning_rate_generator, beta1=flags.FLAGS.beta1, beta2=flags.FLAGS.beta2)
discriminator_optimizer = tf.train.AdamOptimizer(
    learning_rate=flags.FLAGS.learning_rate_discriminator, beta1=flags.FLAGS.beta1, beta2=flags.FLAGS.beta2)

global_step = tf.train.get_or_create_global_step()

gen_checkpoint_dir = os.path.join(basepath, "generator")
gen_checkpoint_prefix = os.path.join(gen_checkpoint_dir, "model.ckpt")
gen_root = tfe.Checkpoint(optimizer=generator_optimizer,
                          model=generator_net,
                          optimizer_step=global_step)

disc_checkpoint_dir = os.path.join(basepath, "discriminator")
disc_checkpoint_prefix = os.path.join(disc_checkpoint_dir, "model.ckpt")
disc_root = tfe.Checkpoint(optimizer=discriminator_optimizer,
                           model=discriminator_net,
                           optimizer_step=global_step)

if os.path.exists(basepath):
    try:
        gen_root.restore(tf.train.latest_checkpoint(gen_checkpoint_dir))
        print("Generator model restored")
    except Exception as ex:
        print("Error loading the Generator model:", ex)

    try:
        disc_root.restore(tf.train.latest_checkpoint(disc_checkpoint_dir))
        print("Discriminator model restored")
    except Exception as ex:
        print("Error loading the Discriminator model:", ex)
    print("Current global step:", tf.train.get_or_create_global_step().numpy())
else:
    print("Model folder not found.")

# generate sample noise for evaluation
fake_input_test = tf.random_normal(shape=(
    flags.FLAGS.number_of_test_images, flags.FLAGS.z_size), dtype=flags.FLAGS.dtype)

# print("train_dateset", train_dataset)
t = tqdm(total=REPORT_INTERVAL)

for _, (batch_real_images) in enumerate(train_dataset):
    fake_input = tf.random_normal(
        shape=(flags.FLAGS.batch_size, flags.FLAGS.z_size), dtype=flags.FLAGS.dtype)

    with tf.contrib.summary.record_summaries_every_n_global_steps(flags.FLAGS.record_summary_after_n_steps):

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # run the generator with the random noise batch
            g_images = generator_net(fake_input, is_training=True)

            # run the discriminator with real input images
            # d_logits = discriminator_net(
            #     tf.concat([batch_real_images, g_images], axis=0), is_training=True)

            # d_logits_real = d_logits[:flags.FLAGS.batch_size]
            # d_logits_fake = d_logits[flags.FLAGS.batch_size:]

            d_logits_real = discriminator_net(
                batch_real_images, is_training=True)

            # run the discriminator with fake input images (images from the generator)
            d_logits_fake = discriminator_net(g_images, is_training=True)

            # compute the generator loss
            gen_loss = generator_loss(d_logits_fake)

            # compute the discriminator loss
            dis_loss = discriminator_loss(d_logits_real, d_logits_fake)

        tf.contrib.summary.scalar('generator_loss', gen_loss)
        tf.contrib.summary.scalar('discriminator_loss', dis_loss)
        tf.contrib.summary.image(
            'generator_image', tf.to_float(g_images), max_images=5)

        # get all the discriminator variables, including the tfe variables
        discriminator_variables = discriminator_net.variables
        # discriminator_variables.append(discriminator_net.attention.gamma)

        discriminator_grads = d_tape.gradient(
            dis_loss, discriminator_variables)

        # get all the discriminator variables, including the tfe variables
        generator_variables = generator_net.variables
        # generator_variables.append(generator_net.attention.gamma)

        generator_grads = g_tape.gradient(gen_loss, generator_variables)

        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator_variables),
                                                global_step=global_step)

        generator_optimizer.apply_gradients(zip(generator_grads, generator_variables),
                                            global_step=global_step)

    counter = global_step.numpy()
    t.update(flags.FLAGS.batch_size * 2)  # TODO: figure out why need to * 2

    if counter % (REPORT_INTERVAL // flags.FLAGS.batch_size) == 0:
        print("Current step:", counter)
        with tf.contrib.summary.always_record_summaries():
            generated_samples = generator_net(
                fake_input_test, is_training=False)
            tf.contrib.summary.image('test_generator_image', tf.to_float(
                generated_samples), max_images=16)
        t = tqdm(total=REPORT_INTERVAL)

    if counter % 15000 == 0:
        # save and download the mode
        save_model(gen_root, gen_checkpoint_prefix)
        save_model(disc_root, disc_checkpoint_prefix)

    if counter >= flags.FLAGS.total_train_steps:
        save_model(gen_root, gen_checkpoint_prefix)
        save_model(disc_root, disc_checkpoint_prefix)
        break
