
from email import generator
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.backend import stack
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Reshape
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
from keras.models import load_model
from keras.utils.vis_utils import plot_model

# from other files

from coalition_winkler import mixed_loss


PRINT_COMMENTS = True
SHOW_PLOTS = True


# model params
THRESHOLD = 0.5
N_COALITION = 2
N_TOTAL = 6
M = 4
EPSILON = 0.001

PROBS_RAW = [0.34, 0.41, 0.67, 0.81]
assert len(PROBS_RAW) == M
PROBS = tf.convert_to_tensor(PROBS_RAW)

BATCH_SIZE = 32

EPOCHS = 50
noise_dim = (N_TOTAL, M)
ALPHA = 0.7
NOISE = 0.03


cross_entropy = BinaryCrossentropy()


def make_discriminator_model():
    inputs = Input(shape=(N_COALITION, M), dtype=tf.float32)
    layer = Flatten()(inputs)
    layer = Dense(N_COALITION * M, activation='relu')(layer)
    layer = Dense(N_COALITION * M/2, activation='relu')(layer)
    outputs = Dense(N_COALITION, activation='sigmoid')(layer)

    model = Model(inputs=inputs, outputs=outputs, name="discriminator_model")
    return model


def make_generator_model():
    inputs = Input(shape=(N_COALITION, M), dtype=tf.float32)
    layer = Flatten()(inputs)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M)(layer)
    layer = Dense(N_COALITION * M, activation='sigmoid')(layer)
    # we have to concatenate the input biases in the final layer so they can appear in the loss function
    # but the input biases never actually change
    outputs = Concatenate(axis=0)([Reshape((1, N_COALITION, M))(layer),
                                   Reshape((1, N_COALITION, M))(inputs)])

    model = Model(inputs=inputs, outputs=outputs, name="generator_model")
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    # / tf.math.log(tf.cast(tf.size(fake_output), tf.float32))
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_data(generator, epoch, print_interval):
    print(f'epoch {epoch}')
    tests = []
    for index in range(M):
        test = np.zeros((1, M))
        test[0, index] = 1
        test = np.tile(test, [N_COALITION, 1])
        tests.append(test)
    predictions = generator(np.array(tests))
    predictions, _ = np.split(predictions, 2, axis=0)
    predictions = np.array([prediction[0] for prediction in predictions])
    reports = [[] for _ in range(M)]
    for prediction in predictions:
        for q in range(M):
            reports[q].extend(prediction[:, q])
        if epoch % print_interval == 0:
            print('alpha: ' + str(ALPHA))
            print(prediction)
            print('mean: ' + str(np.mean(prediction, axis=0)))
            print('std: ' + str(np.std(prediction, axis=0)))
            print('')
    mean_reports = np.mean(reports, axis=1)
    std_reports = np.std(reports, axis=1)
    return mean_reports, std_reports


def main(epochs):
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    generator_optimizer = Adam(3e-4, beta_1=0.95, epsilon=1e-8, amsgrad=True)
    discriminator_optimizer = Adam(2e-4, epsilon=1e-8, amsgrad=True)

    @tf.function
    def train_step():
        indices = np.random.randint(0, M, BATCH_SIZE)
        X = np.array([np.tile(np.reshape([float(j == index) for j in range(M)], (1, M)), (N_COALITION, 1))
                      for _, index in enumerate(indices)])

        y_outcomes = np.random.binomial(
            1, PROBS, (BATCH_SIZE, M)).astype(np.float32)
        y_noise = np.reshape(np.clip(np.random.normal(PROBS, NOISE,
                                                      (BATCH_SIZE * (N_TOTAL - N_COALITION), M)),
                                     EPSILON, 1-EPSILON),
                             (BATCH_SIZE, (N_TOTAL - N_COALITION) * M))
        y = np.concatenate([y_outcomes, y_noise], axis=1)

        real_reports = np.reshape(np.clip(np.random.normal(PROBS, NOISE,
                                                           (BATCH_SIZE * N_COALITION, M)),
                                          EPSILON, 1-EPSILON),
                                  (BATCH_SIZE, N_COALITION, M))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_reports = generator(X, training=True)
            generated_reports = tf.map_fn(
                lambda x: x[0], generated_reports)
            y = tf.cast(y, tf.float32)
            utility_loss = mixed_loss(ALPHA)(
                y, generated_reports)
            generated_reports, _ = tf.split(generated_reports, 2, axis=0)

            real_output = discriminator(real_reports, training=True)
            fake_output = discriminator(generated_reports, training=True)

            gen_loss = 0.01 * utility_loss + generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(epochs):
        # train the generator first
        indices = np.random.randint(0, M, BATCH_SIZE * 256)
        X = np.array([np.tile(np.reshape([float(j == index) for j in range(M)], (1, M)), (N_COALITION, 1))
                      for _, index in enumerate(indices)])

        y_outcomes = np.random.binomial(
            1, PROBS, (BATCH_SIZE * 256, M)).astype(np.float32)
        y_noise = np.reshape(np.clip(np.random.normal(PROBS, NOISE,
                                                      (BATCH_SIZE * 256 * (N_TOTAL - N_COALITION), M)),
                                     EPSILON, 1-EPSILON),
                             (BATCH_SIZE * 256, (N_TOTAL - N_COALITION) * M))
        y = np.concatenate([y_outcomes, y_noise], axis=1)
        generator.compile(loss=mixed_loss(ALPHA),
                          optimizer=Adam(learning_rate=0.0015, epsilon=1e-8, amsgrad=True))
        generator.fit(X, y, epochs=20, verbose=False)

        discriminator_losses = []
        generator_losses = []
        mean_reports = []
        std_reports = []
        for epoch in range(epochs):
            PRINT_INTERVAL = 10
            # these are actually all arrays
            mean_report, std_report = generate_and_save_data(
                generator, epoch, PRINT_INTERVAL)
            mean_reports.append(mean_report)
            std_reports.append(std_report)

            gen_losses = []
            disc_losses = []
            for _ in range(128):
                gen_loss, disc_loss = train_step()
                gen_losses.append(gen_loss.numpy())
                disc_losses.append(disc_loss.numpy())
            discriminator_losses.append(np.mean(disc_losses))
            generator_losses.append(np.mean(gen_losses))

        discriminator_losses = np.array(discriminator_losses)
        generator_losses = np.array(generator_losses)

        with open('gan.npz', 'wb') as f:
            np.savez(f, discriminator_losses=discriminator_losses,
                     generator_losses=generator_losses, mean_reports=mean_reports, std_reports=std_reports)

        plt.plot(np.array(discriminator_losses))
        plt.xlabel('Epoch')
        plt.ylabel('Discriminator loss')
        plt.show()
        plt.plot(np.array(generator_losses))
        plt.xlabel('Epoch')
        plt.ylabel('Generator loss')
        plt.show()

    train(epochs)


def post_process():
    with open('gan.npz', 'rb') as f:
        files = np.load(f)
        discriminator_losses = files['discriminator_losses']
        generator_losses = files['generator_losses']

        plt.plot(discriminator_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Discriminator loss')
        plt.show()
        plt.plot(generator_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Generator loss')
        plt.show()

        epochs = [i for i in range(len(generator_losses))]

        mean_reports = files['mean_reports']
        std_reports = files['std_reports']

        mean_reports = np.split(mean_reports, M, axis=1)
        std_reports = np.split(std_reports, M, axis=1)

        for mean_report, std_report in zip(mean_reports, std_reports):
            plt.plot(mean_report.ravel())
            # plt.fill_between(x=epochs, y1=mean_report.ravel() - std_report.ravel(),
            #                 y2=mean_report.ravel() + std_report.ravel(), color=color)

        plt.xlabel('Epoch')
        plt.ylabel('Average reports')
        plt.legend(['1', '2', '3', '4'], title='Borrower', loc='upper left')
        plt.ylim(0, 1)

        plt.show()


if __name__ == '__main__':
    main(150)
    post_process()
