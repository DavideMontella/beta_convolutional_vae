from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Set memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image 

import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import tensorflow_probability as tfp

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class BVAE(keras.Model):
    def __init__(self, encoder, decoder, beta, flatted_dim_for_loss, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
         
        self.flatted_dim_for_loss = flatted_dim_for_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= self.flatted_dim_for_loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss * self.beta
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

class BCVAE():
    '''
    This class make a Beta Convolutional Variational Auto-Encoder (BCVAE) starting from images given in input.
    Args:
        images (list): images list 
    Attributes:
        encoder (instance): function map from image to latent space
        dencoder (instance): function map from latent space to image 
    '''
    def __init__(self, images, latent_dim, epochs, batch_size=128 , beta=60, n_convolutions=3, 
                            n_hidden_layers=3, background_subtractor=False, resize_images = None, rgb_average=False, 
                                train_percentage=0.80, first_layer_filters=32, first_layer_neurons=500):

        # reparameterization trick
        # instead of sampling from Q(z|X), sample epsilon = N(0,I)
        # z = z_mean + sqrt(var) * epsilon
        def sampling(args):
            """Reparameterization trick by sampling from an isotropic unit Gaussian.
            # Arguments
                args (tensor): mean and log of variance of Q(z|X)
            # Returns
                z (tensor): sampled latent vector
            """

            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        self.latent_dim = latent_dim
        self.background_subtractor = background_subtractor
        self.background = None    
        self.rgb_average = rgb_average
        self.resize_images = resize_images

        if resize_images is not None:
            self.resize_images = resize_images if type(resize_images) is tuple else (resize_images,resize_images)         
        
        self.rgb_images = np.max(images[np.random.randint(len(images))]) > 1 

        self.images = self.images_standardization(images)
        
        image = self.images[0]
        for i in range(len(image.shape)):
           n = image.shape[i]
           if n % 2 != 0:
               print("The {}-dimension isn't divisible by two (it is {})".format(i, n))

        kernel_dim = 3
        strides = 2


        encoder_inputs = keras.Input(shape=image.shape)

        x = encoder_inputs
        filters = first_layer_filters
        filters_used = []
        dim_1 = image.shape[0]
        dim_2 = image.shape[1]
        dim_3 = 1
        if len(image.shape) == 3:
           dim_3 = image.shape[2]

        for i in range(n_convolutions):

            if (dim_1 / strides) != int(dim_1 / strides):
                print("Reached (therefore stopped) the max number of convolutions {} (cause dim_1)".format(i+1))
                break
            if (dim_2 / strides) != int(dim_2 / strides):
                print("Reached (therefore stopped) the max number of convolutions {} (cause dim_2)".format(i+1))
                break

            x = layers.Conv2D(filters, kernel_dim, activation="relu", strides=strides, padding="same")(x)
            filters_used.append(filters)
            filters *= 2
            dim_1 = int(dim_1 / strides)
            dim_2 = int(dim_2 / strides)

        if n_convolutions:
            x = layers.Flatten()(x)

        neurons = first_layer_neurons
        neurons_used = []
        for _ in range(n_hidden_layers):
            x = layers.Dense(neurons, activation="relu")(x)
            neurons_used.append(neurons)
            neurons = int(neurons/2)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))

        x = latent_inputs
        for i in range(n_hidden_layers-1,-1,-1):
             x = layers.Dense(neurons_used[i], activation="relu")(x)

        if n_convolutions:
            x = layers.Dense(dim_1 * dim_2 * filters_used[-1], activation="relu")(x)
            x = layers.Reshape((dim_1, dim_2, filters_used[-1]))(x)

            for i in range(len(filters_used)-1,-1,-1):
                x = layers.Conv2DTranspose(filters_used[i], kernel_dim, activation="relu", strides=strides, padding="same")(x)
        
        decoder_outputs = layers.Conv2DTranspose(dim_3, kernel_dim, activation="sigmoid", padding="same")(x)

        if dim_3 == 1:
            decoder_outputs = layers.Reshape(image.shape)(decoder_outputs)

        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

        x_train = self.images[:int(np.floor(len(self.images)*train_percentage))] 
        x_test = self.images[int(np.ceil(len(self.images)*train_percentage)):] 

        data = np.concatenate([x_train, x_test], axis=0)
        #data = np.expand_dims(data, -1).astype("float32") 

        flatted_dim = dim_1*dim_2*dim_3
        vae = BVAE(self.encoder, self.decoder, beta, flatted_dim_for_loss = flatted_dim)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(data, epochs=epochs, batch_size=batch_size)

    def images_standardization(self, images):
        if self.resize_images is not None:  
            images = np.array([np.array(Image.fromarray(images[i]).resize(self.resize_images)) for i in range(len(images))])
        
        if self.background_subtractor:   
            images = np.array(images)

            if self.background is None:
                cbsm = cv2.createBackgroundSubtractorMOG2(len(images))
                for i in range(len(images)):
                    cbsm.apply(images[i])

                self.background = cbsm.getBackgroundImage()
            
            images = np.abs(images - self.background)

        images = images / 255 if self.rgb_images else images
        
        if rgb_average:
            images = np.average(images, axis=3)  

        return np.array(standardizated_images)        

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def encoding(self, images):
        images = self.images_standardization(images)
        return self.encoder(images)
       
    def decoding(self, latents):
        return self.decoder(latents).numpy()  
    



class BCVAEAnalisys():
    def __init__(self, images, latent_dim = 7, epochs=30, 
                        beta = 80, n_convolutions = 3, n_hidden_layers= 3):

        self.latent_dim = latent_dim
        self.epochs = epochs
        self.beta = beta
        self.n_convolutions = n_convolutions
        self.n_hidden_layers= n_hidden_layers
        self.images = images
    
        self.abst = BCVAE(images, latent_dim, epochs, beta = beta, n_convolutions=n_convolutions, n_hidden_layers=n_hidden_layers)

    '''
    Description
        save a image that shows 5 random image recostructions
    '''
    def five_random_recostructions(self):  
        columns = 5
        rows = 2
        imgs = [self.images[np.random.randint(len(self.images))] for _ in range(columns)]
        imgs = imgs + [self.abst.decoder(self.abst.encoding(np.array([imgs[i],imgs[i]])))[0] for i in range(columns)]


        fig = plt.figure(figsize=(10., 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(rows, columns),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

        for ax, im in zip(grid, imgs):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            if np.all(im == imgs[0]):
                ax.set_ylabel("Original")
            if np.all(im == imgs[columns*(rows-1)]):
                ax.set_ylabel("Reconstruction")

        plt.show()
        plt.savefig("reconstructions_plot_{}_{}_{}_{}_{}_{}".format(len(self.images), self.latent_dim, self.epochs, self.beta, self.n_convolutions, self.n_hidden_layers))
    
    '''
    Description
        find the images pairs nearest in the latent space and cartesian space (distances at zero not included), 
        after check how much the two distances differs and if the distance is over the overbound (dist param) the 
        pair is added inconsistent distances
    '''
    def inconsistent_distance(self, dist=0.01): #0.01 is 1cm in REAL environment
        encoded_images = []
        for i in range(256,len(images)+256, 256):
            print((i-256,i))
            batch = images[i-256: i]
            encoded_batch = self.abst.encoding(np.array(batch))[0]
            encoded_images += list(encoded_batch)

        print("Images encoded.")

        idxs_inconsistent = []
        for i in range(len(encoded_images)-1):
            pre, post = encoded_images[i], encoded_images[i+1]
            if np.all(pre == post):
                idxs_inconsistent += [i]

        print("Number of sequential identical images {}".format(len(idxs_inconsistent)))

        filtered_dataset = [dataset[i] for i in range(len(dataset)) if i not in idxs_inconsistent]

        print("Total images: {}, survived images: {}".format(len(images), len(filtered_dataset)))

        filtered_encoded = [encoded_images[i] for i in range(len(dataset)) if i not in idxs_inconsistent]

        filtered_coord = [filtered_dataset[i][2][1]['cube'] for i in range(len(filtered_dataset))]

        dist = [np.linalg.norm(np.array(filtered_coord)-filtered_coord[i], axis=1) for i in range(len(filtered_coord))]

        min_dist_idxs = []
        for i in range(len(dist)):
            idx = 0 if i != 0 else 1 
            for j in range(len(dist[i])):
                if j == i:
                    continue
                if dist[i][j] < dist[i][idx]:
                    idx = j
            min_dist_idxs += [idx]

        encoded_dist = [np.linalg.norm(np.array(filtered_encoded)-filtered_encoded[i], axis=1) for i in range(len(filtered_encoded))]

        encoded_min_dist_idxs = [] 
        for i in range(len(encoded_dist)): 
            idx = 0 if i != 0 else 1  
            for j in range(len(encoded_dist[i])): 
                if j == i: 
                    continue 
                if encoded_dist[i][j] < encoded_dist[i][idx]: 
                    idx = j 
            encoded_min_dist_idxs += [idx] 

        f = lambda i: np.abs(np.linalg.norm(filtered_coord[i]-filtered_coord[encoded_min_dist_idxs[i]]) - np.linalg.norm(filtered_coord[i]-filtered_coord[min_dist_idxs[i]]) )

        dist_differences = [f(i) for i in range(len(filtered_coord))] 

        inconsistencies = np.where(np.array(dist_differences) >= dist)[0]  
    
        return len(inconsistencies)

      

#######################SKEWFIT THINGS##########################

  
class BCVAEWithSkewFit(BCVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_goals(self, images, n_sample=1, batch_size=256, alpha=-1, debug=False):
        
        partial_log_p_z = []
        partial_log_q_z_given_x = []
        partial_log_d_x_given_z = []
        log_p_z = np.array([])
        log_q_z_given_x = np.array([])
        log_d_x_given_z = np.array([])
        for i in range(batch_size, batch_size*(int(np.ceil(len(images)/batch_size)))+1, batch_size):
            print((i-batch_size, i))
            partial_images = images[i-batch_size:i]
            latents_distributions = self.encoder(partial_images)
            prior_distributions = tfp.distributions.Normal([[0]*self.latent_dim]*len(partial_images),[[1]*self.latent_dim]*len(partial_images))
            mus, log_vars = latents_distributions[0], latents_distributions[1]
            latents = latents_distributions[2]
            stds = np.power(np.exp(log_vars),0.5)
            vae_distributions = tfp.distributions.Normal(mus, stds)
            partial_log_p_z = prior_distributions.log_prob(latents).numpy().sum(axis=1)
            partial_log_q_z_given_x = vae_distributions.log_prob(latents).numpy().sum(axis=1)
            decoded_images = np.array([img.flatten() for img in self.decoder(latents).numpy()]).astype(np.float64)
            mus_decoded, stds_decoded = decoded_images, np.power([[1]*(partial_images.shape[1]*partial_images.shape[2]*partial_images.shape[3])]*partial_images.shape[0],0.5)
            decoder_distribution = tfp.distributions.Normal(mus_decoded, stds_decoded)
            flatten_images = [img.flatten() for img in partial_images]
            partial_log_d_x_given_z = decoder_distribution.log_prob(flatten_images).numpy().sum(axis=1)

            log_p_z = np.concatenate([log_p_z, partial_log_p_z])
            log_q_z_given_x = np.concatenate([log_q_z_given_x, partial_log_q_z_given_x])
            log_d_x_given_z = np.concatenate([log_d_x_given_z, partial_log_d_x_given_z])
        

        log_p_x = log_p_z - log_q_z_given_x + log_d_x_given_z
        log_p_x_skewed = self.skew(log_p_x, alpha)


        idxs = np.random.choice(range(len(images)), n_sample, p=log_p_x_skewed)
        sampled_images = np.array([images[i] for i in idxs])

        return sampled_images if not debug else sampled_images, log_p_x, log_p_x_skewed

    #unormalized_log_probabilities is the sample_goals function output (sample_goals(args)[1])
    #in the case debug params is True
    def skew(self, unormalized_log_probabilities, alpha):
        log_p_x_skewed = alpha * unormalized_log_probabilities
        log_p_x_skewed = np.exp(log_p_x_skewed - np.average(log_p_x_skewed))
        p_sum = np.sum(log_p_x_skewed)
        return log_p_x_skewed / p_sum
    

class VAEWithSkewFitAnalisys(BCVAEAnalisys):
    def __init__(self, images, latent_dim = 7, 
                        epochs=30, beta = 80, n_convolutions = 3, n_hidden_layers= 3):

        

        self.latent_dim = latent_dim
        self.epochs = epochs
        self.beta = beta
        self.n_convolutions = n_convolutions
        self.n_hidden_layers= n_hidden_layers
    
        self.abst = BCVAEWithSkewFit(images, latent_dim, epochs, beta = beta, n_convolutions=n_convolutions, n_hidden_layers=n_hidden_layers)


    '''
    Description
        show how skew-fit work. It takes a number of the sample in a limited space and few sample out of it and it use skew-fit 
        to identify outliers. The method will give a high probability at the rarest data
    Input
        max_neighbors_distance: limited space radius (starting from table center)
        min_far_distance: minimum outlier distance from the table center 
    Output
        several images saved in the current directory that show how much said above 
        several numpy array that contain the data used to make the images
    '''
    def skewfit_test_with_preselected_images(self, max_neighbors_distance=0.15, min_far_distance=0.25):
        selected_images = [] 
        selected_idxs = []
        for i in range(len(dataset)): 
            if np.linalg.norm(dataset[0][2][1]['cube'] - dataset[i][2][1]['cube']) < max_neighbors_distance and np.linalg.norm(dataset[0][2][1]['cube'] - dataset[i][2][1]['cube']) > 0.001: 
                selected_images += [images[i]]
                selected_idxs += [i]

        abst = BCVAE(selected_images, latent_dim, epochs, beta = beta, n_convolutions=n_convolutions, n_hidden_layers=n_hidden_layers)

        spaced_images = [] 
        spaced_idxs = []
        n = 50
        for i in range(len(dataset)): 
            if np.linalg.norm(dataset[0][2][1]['cube'] - dataset[i][2][1]['cube']) > min_far_distance: 
                spaced_images += [images[i]]
                spaced_idxs += [i]
            if len(spaced_images) == n:
                break



        some_images = np.concatenate([selected_images, spaced_images])
        some_idxs = np.concatenate([selected_idxs, spaced_idxs])

        print("sample list size {}".format(len(some_images)))

        _, vae_distribution, skewed_vae_distribution = abst.sample_goals(some_images,100000, debug=True)

        distr = abst.skew(vae_distribution, alpha=0)
        equi_idxs = np.random.choice(range(len(images)), 100000, p=distr)
        equi_samples = np.array([dataset[i][2][1]['cube'][:2] for i in equi_idxs])
        np.save("equi_samples", equi_samples)

        x, y = [e[1] for e in equi_samples], [e[0] for e in equi_samples] 
     
        plt.hist2d(x, y, (30,30), cmap=plt.cm.jet)  
        plt.colorbar()   
        plt.xlim([-0.5,0.5])   
        plt.ylim([-0.25,0.05])
        
        plt.savefig("hist_2d_equi_samples_vae")

        distr = abst.skew(vae_distribution, alpha=-1)
        skew_idxs = np.random.choice(range(len(images)), 100000, p=distr)
        skew_samples = np.array([dataset[i][2][1]['cube'][:2] for i in skew_idxs])
        np.save("skew_samples", skew_samples)

        x, y = [e[1] for e in skew_samples], [e[0] for e in skew_samples] 
     
        plt.hist2d(x, y, (30,30), cmap=plt.cm.jet)  
        plt.colorbar()   
        plt.xlim([-0.5,0.5])   
        plt.ylim([-0.25,0.05])

        plt.savefig("hist_2d_skew_samples_vae")

        distr = abst.skew(vae_distribution, alpha=-0.5)
        half_skew_idxs = np.random.choice(range(len(images)), 100000, p=distr)
        half_skew_samples = np.array([dataset[i][2][1]['cube'][:2] for i in half_skew_idxs])
        np.save("half_skew_samples", half_skew_samples)

        x, y = [e[1] for e in half_skew_samples], [e[0] for e in half_skew_samples] 
     
        plt.hist2d(x, y, (30,30), cmap=plt.cm.jet)  
        plt.colorbar()   
        plt.xlim([-0.5,0.5])   
        plt.ylim([-0.25,0.05])

        plt.savefig("hist_2d_half_skew_samples_vae")

        distr = abst.skew(vae_distribution, alpha=-0.025)
        half_half_skew_idxs = np.random.choice(range(len(images)), 100000, p=distr)
        half_half_skew_samples = np.array([dataset[i][2][1]['cube'][:2] for i in half_half_skew_idxs])
        np.save("half_half_skew_samples", half_half_skew_samples)

        x, y = [e[1] for e in half_half_skew_samples], [e[0] for e in half_half_skew_samples] 
     
        plt.hist2d(x, y, (30,30), cmap=plt.cm.jet)  
        plt.colorbar()   
        plt.xlim([-0.5,0.5])   
        plt.ylim([-0.25,0.05])

        plt.savefig("hist_2d_half_half_skew_samples_vae")

        distr = abst.skew(vae_distribution, alpha=-0.0125)
        half_half_half_skew_idxs = np.random.choice(range(len(images)), 100000, p=distr)
        half_half_skew_samples = np.array([dataset[i][2][1]['cube'][:2] for i in half_half_skew_idxs])
        np.save("half_half_skew_samples", half_half_skew_samples)

        x, y = [e[1] for e in half_half_half_skew_samples], [e[0] for e in half_half_half_skew_samples] 
     
        plt.hist2d(x, y, (30,30), cmap=plt.cm.jet)  
        plt.colorbar()   
        plt.xlim([-0.5,0.5])   
        plt.ylim([-0.25,0.05])

        plt.savefig("hist_2d_half_half_half_skew_samples_vae")


if __name__ == "__main__":

    dataset = np.load("/DATASET/REAL-Solution/data/transitions_file.npy", allow_pickle=True)

    images = np.array([np.array(img[2][0]) for img in dataset])

    bcvae_analisys = BCVAEAnalisys(images)
    bcvae_analisys.five_random_recostructions()
