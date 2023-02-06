import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
import time

from IPython import display

import generator
import dataset
import discriminator

import tensorflow_docs.vis.embed as embed

BUFFER_SIZE=60000
BATCH_SIZE=256

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

def saveModel(generator_optimizer, discriminator_optimizer, generator, discriminator):
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    return checkpoint

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator.model(noise, training=True)

      real_output = discriminator.model(images, training=True)
      fake_output = discriminator.model(generated_images, training=True)

      gen_loss = generator.generator_loss(fake_output)
      disc_loss = discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))

def train(dataset, epochs, generator, discriminator):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset.train_dataset:
            train_step(image_batch, generator, discriminator)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator.model,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

def make_gif():
    anim_file = 'dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    
    return anim_file

if __name__ == "__main__":
    train_dataset = dataset.Dataset()
    train_dataset.importExample()

    generator = generator.Generator()
    generator.make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator.model(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    discriminator = discriminator.Discriminator()
    discriminator.make_discriminator_model()
    decision = discriminator.model(generated_image)
    print (decision)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = saveModel(generator_optimizer, discriminator_optimizer, generator.model, discriminator.model)

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    train(train_dataset, EPOCHS, generator, discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    display_image(EPOCHS)

    anim_file = make_gif()

    embed.embed_file(anim_file)