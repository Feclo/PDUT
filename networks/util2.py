import math

import numpy as np
import tensorflow as tf

from constants import SRF_GS3_U3_41S4C_BGR_31_CHANNEL_400_700NM, \
    SRF_GS3_U3_41S4C_BGR_31_CHANNEL_420_720NM, SRF_SAPPHIRE_S25A30CL_GRAY_31_CHANNEL_400_700NM, \
    SRF_MOCK_BGR_31_CHANNEL_400_700NM

SRF_BGR_31_CHANNEL_400_700_NM = SRF_GS3_U3_41S4C_BGR_31_CHANNEL_400_700NM
SRF_BGR_31_CHANNEL_420_720_NM = SRF_GS3_U3_41S4C_BGR_31_CHANNEL_420_720NM
SRF_GRAY_31_CHANNEL_400_700_NM = SRF_SAPPHIRE_S25A30CL_GRAY_31_CHANNEL_400_700NM


def simulated_rgb_camera_spectral_response_function(hyper_spectral_image):
    channel_num = 31
    masked_response_function = SRF_BGR_31_CHANNEL_400_700_NM

    red_response = hyper_spectral_image * tf.reshape(masked_response_function[2], shape=[1, 1, 1, channel_num])
    red_channel = tf.reduce_sum(red_response, axis=-1) / tf.reduce_sum(masked_response_function[2])

    green_response = hyper_spectral_image * tf.reshape(masked_response_function[1], shape=[1, 1, 1, channel_num])
    green_channel = tf.reduce_sum(green_response, axis=-1) / tf.reduce_sum(masked_response_function[1])

    blue_response = hyper_spectral_image * tf.reshape(masked_response_function[0], shape=[1, 1, 1, channel_num])
    blue_channel = tf.reduce_sum(blue_response, axis=-1) / tf.reduce_sum(masked_response_function[0])

    rgb_image = tf.stack([red_channel, green_channel, blue_channel], axis=-1)

    return rgb_image


def inverse_simulated_rgb_channel_camera_spectral_response_function(input_rgb_image,
                                                                    input_as_rggb=False):
    channel_num = 31
    masked_response_function = SRF_BGR_31_CHANNEL_400_700_NM
    green_2_channel = None
    wave_length_sum_response = tf.reduce_sum(masked_response_function, axis=0)
    if input_as_rggb:
        red_channel, green_channel, green_2_channel, blue_channel = tf.split(input_rgb_image, num_or_size_splits=4,
                                                                             axis=-1)
    else:
        red_channel, green_channel, blue_channel = tf.split(input_rgb_image, num_or_size_splits=3, axis=-1)

    red_channel_expanded = tf.repeat(red_channel, repeats=channel_num, axis=-1)
    green_channel_expanded = tf.repeat(green_channel, repeats=channel_num, axis=-1)
    blue_channel_expanded = tf.repeat(blue_channel, repeats=channel_num, axis=-1)

    red_channel_expanded = red_channel_expanded * (masked_response_function[2] / wave_length_sum_response)
    green_channel_expanded = green_channel_expanded * (masked_response_function[1] / wave_length_sum_response)
    blue_channel_expanded = blue_channel_expanded * (masked_response_function[0] / wave_length_sum_response)

    if input_as_rggb:
        assert green_2_channel is not None
        green_2_channel_expanded = tf.repeat(green_2_channel, repeats=channel_num, axis=-1) \
                                   * (masked_response_function[1] / wave_length_sum_response)
        green_channel_expanded = (green_2_channel_expanded + green_channel_expanded) / 2

    return red_channel_expanded + green_channel_expanded + blue_channel_expanded


def fai_I(I, psfs):
    pre_sensor_image = image_convolve_with_psf(I, psfs)
    J = simulated_rgb_camera_spectral_response_function(pre_sensor_image)

    return J


def faiT_J(J, psfs):
    channel_num = 31
    masked_response_function = SRF_BGR_31_CHANNEL_400_700_NM
    RGB_to_hyper = tf.transpose(masked_response_function, perm=[1, 0])
    aaa = J

    k = tf.convert_to_tensor(masked_response_function)
    r = k[2, :] / tf.reduce_sum(masked_response_function[2])
    g = k[1, :] / tf.reduce_sum(masked_response_function[1])
    b = k[0, :] / tf.reduce_sum(masked_response_function[0])
    hyper_to_RGB = tf.stack([r, g, b], axis=0)
    RGB_to_hyper = tf.transpose(hyper_to_RGB, perm=[1, 0])

    response0 = J * tf.reshape(RGB_to_hyper[0], shape=[1, 1, 1, 3])
    hyper_channel0 = tf.reduce_sum(response0, axis=-1)
    response1 = J * tf.reshape(RGB_to_hyper[1], shape=[1, 1, 1, 3])
    hyper_channel1 = tf.reduce_sum(response1, axis=-1)
    response2 = J * tf.reshape(RGB_to_hyper[2], shape=[1, 1, 1, 3])
    hyper_channel2 = tf.reduce_sum(response2, axis=-1)
    response3 = J * tf.reshape(RGB_to_hyper[3], shape=[1, 1, 1, 3])
    hyper_channel3 = tf.reduce_sum(response3, axis=-1)
    response4 = J * tf.reshape(RGB_to_hyper[4], shape=[1, 1, 1, 3])
    hyper_channel4 = tf.reduce_sum(response4, axis=-1)
    response5 = J * tf.reshape(RGB_to_hyper[5], shape=[1, 1, 1, 3])
    hyper_channel5 = tf.reduce_sum(response5, axis=-1)
    response6 = J * tf.reshape(RGB_to_hyper[6], shape=[1, 1, 1, 3])
    hyper_channel6 = tf.reduce_sum(response6, axis=-1)
    response7 = J * tf.reshape(RGB_to_hyper[7], shape=[1, 1, 1, 3])
    hyper_channel7 = tf.reduce_sum(response7, axis=-1)
    response8 = J * tf.reshape(RGB_to_hyper[8], shape=[1, 1, 1, 3])
    hyper_channel8 = tf.reduce_sum(response8, axis=-1)
    response9 = J * tf.reshape(RGB_to_hyper[9], shape=[1, 1, 1, 3])
    hyper_channel9 = tf.reduce_sum(response9, axis=-1)
    response10 = J * tf.reshape(RGB_to_hyper[10], shape=[1, 1, 1, 3])
    hyper_channel10 = tf.reduce_sum(response10, axis=-1)
    response11 = J * tf.reshape(RGB_to_hyper[11], shape=[1, 1, 1, 3])
    hyper_channel11 = tf.reduce_sum(response11, axis=-1)
    response12 = J * tf.reshape(RGB_to_hyper[12], shape=[1, 1, 1, 3])
    hyper_channel12 = tf.reduce_sum(response12, axis=-1)
    response13 = J * tf.reshape(RGB_to_hyper[13], shape=[1, 1, 1, 3])
    hyper_channel13 = tf.reduce_sum(response13, axis=-1)
    response14 = J * tf.reshape(RGB_to_hyper[14], shape=[1, 1, 1, 3])
    hyper_channel14 = tf.reduce_sum(response14, axis=-1)
    response15 = J * tf.reshape(RGB_to_hyper[15], shape=[1, 1, 1, 3])
    hyper_channel15 = tf.reduce_sum(response15, axis=-1)
    response16 = J * tf.reshape(RGB_to_hyper[16], shape=[1, 1, 1, 3])
    hyper_channel16 = tf.reduce_sum(response16, axis=-1)
    response17 = J * tf.reshape(RGB_to_hyper[17], shape=[1, 1, 1, 3])
    hyper_channel17 = tf.reduce_sum(response17, axis=-1)
    response18 = J * tf.reshape(RGB_to_hyper[18], shape=[1, 1, 1, 3])
    hyper_channel18 = tf.reduce_sum(response18, axis=-1)
    response19 = J * tf.reshape(RGB_to_hyper[19], shape=[1, 1, 1, 3])
    hyper_channel19 = tf.reduce_sum(response19, axis=-1)
    response20 = J * tf.reshape(RGB_to_hyper[20], shape=[1, 1, 1, 3])
    hyper_channel20 = tf.reduce_sum(response20, axis=-1)
    response21 = J * tf.reshape(RGB_to_hyper[21], shape=[1, 1, 1, 3])
    hyper_channel21 = tf.reduce_sum(response21, axis=-1)
    response22 = J * tf.reshape(RGB_to_hyper[22], shape=[1, 1, 1, 3])
    hyper_channel22 = tf.reduce_sum(response22, axis=-1)
    response23 = J * tf.reshape(RGB_to_hyper[23], shape=[1, 1, 1, 3])
    hyper_channel23 = tf.reduce_sum(response23, axis=-1)
    response24 = J * tf.reshape(RGB_to_hyper[24], shape=[1, 1, 1, 3])
    hyper_channel24 = tf.reduce_sum(response24, axis=-1)
    response25 = J * tf.reshape(RGB_to_hyper[25], shape=[1, 1, 1, 3])
    hyper_channel25 = tf.reduce_sum(response25, axis=-1)
    response26 = J * tf.reshape(RGB_to_hyper[26], shape=[1, 1, 1, 3])
    hyper_channel26 = tf.reduce_sum(response26, axis=-1)
    response27 = J * tf.reshape(RGB_to_hyper[27], shape=[1, 1, 1, 3])
    hyper_channel27 = tf.reduce_sum(response27, axis=-1)
    response28 = J * tf.reshape(RGB_to_hyper[28], shape=[1, 1, 1, 3])
    hyper_channel28 = tf.reduce_sum(response28, axis=-1)
    response29 = J * tf.reshape(RGB_to_hyper[29], shape=[1, 1, 1, 3])
    hyper_channel29 = tf.reduce_sum(response29, axis=-1)
    response30 = J * tf.reshape(RGB_to_hyper[30], shape=[1, 1, 1, 3])
    hyper_channel30 = tf.reduce_sum(response30, axis=-1)
    I_pre_deconv = tf.stack(
        [hyper_channel0, hyper_channel1, hyper_channel2, hyper_channel3, hyper_channel4, hyper_channel5, hyper_channel6,
         hyper_channel7, hyper_channel8, hyper_channel9, hyper_channel10, hyper_channel11, hyper_channel12,
         hyper_channel13, hyper_channel14, hyper_channel15, hyper_channel16, hyper_channel17, hyper_channel18,
         hyper_channel19, hyper_channel20, hyper_channel21, hyper_channel22, hyper_channel23, hyper_channel24,
         hyper_channel25, hyper_channel26, hyper_channel27, hyper_channel28, hyper_channel29, hyper_channel30], axis=-1)

    I_pre_deconv2 = inverse_simulated_rgb_channel_camera_spectral_response_function(J)

    psfs = tf.transpose(psfs, perm=[0, 1, 2, 3])
    psfs = tf.transpose(psfs, perm=[1, 2, 0, 3])
    I = image_convolve_with_psf(I_pre_deconv2, psfs)
    return I


def transpose_2d_fft(a_tensor, dtype=tf.complex64):
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_tensor_transp = tf.transpose(a=a_tensor, perm=[0, 3, 1, 2])
    a_fft2d = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a=a_fft2d, perm=[0, 2, 3, 1])
    return a_fft2d


def transpose_2d_ifft(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a=a_tensor, perm=[0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor)
    a_ifft2d = tf.transpose(a=a_ifft2d_transp, perm=[0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d


def complex_exponent_tf(phase, dtype=tf.complex64, name='complex_exp'):
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)


def laplacian_filter_tf(img_batch):
    laplacian_filter = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])
    laplacian_filter = tf.cast(laplacian_filter, tf.float32)
    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(input=filter_input, filters=laplacian_filter, padding="SAME")
    return filtered_batch


def laplace_l1_regularizer(scale=100):
    def laplace_l1(a_tensor):
        laplace_filtered = laplacian_filter_tf(a_tensor)
        laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
        return tf.reshape(
            tf.multiply(scale, tf.reduce_mean(input_tensor=tf.abs(laplace_filtered)),
                        name="laplace_l1_regularized"), [])

    return laplace_l1


def laplace_l2_regularizer(scale):
    def laplace_l2(a_tensor):
        with tf.compat.v1.name_scope('laplace_l2_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
            return scale * tf.reduce_mean(input_tensor=tf.square(laplace_filtered))

    return laplace_l2


def get_one_phase_shift_thickness(wave_lengths, refractive_index):
    """
    Calculate the thickness (in meter) of a phase-shift of 2pi.
    """
    delta_n = refractive_index - 1.
    wave_nos = 2. * np.pi / wave_lengths
    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_n)
    return two_pi_thickness


def fft_shift_2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        temp_list = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, temp_list, axis=axis)
    return new_tensor


def ifft_shift_2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        temp_list = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, temp_list, axis=axis)
    return new_tensor


def psf2otf(input_filter, output_size):
    """
    Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, b, c)
    :param output_size: Size of the output OTF.
    :return: The otf.
    """
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(tensor=input_filter, paddings=[[pad_top, pad_bottom],
                                                       [pad_left, pad_right], [0, 0], [0, 0]], mode="CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(a=padded, perm=[2, 0, 1, 3])
    padded = ifft_shift_2d_tf(padded)
    padded = tf.transpose(a=padded, perm=[1, 2, 0, 3])

    tmp = tf.transpose(a=padded, perm=[2, 3, 0, 1])
    tmp = tf.signal.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(a=tmp, perm=[2, 3, 0, 1])


def next_power_of_two(number):
    closest_pow = np.power(2, np.ceil(np.math.log(number, 2)))
    return closest_pow


def image_convolve_with_psf(img, psf, otf=None, adjoint=False, img_shape=None):
    img = tf.convert_to_tensor(value=img, dtype=tf.float32)
    psf = tf.convert_to_tensor(value=psf, dtype=tf.float32)
    if img_shape is None:
        img_shape = img.shape.as_list()
    psf_shape = psf.shape.as_list()

    circular = False

    if psf_shape[0] != img_shape[1] or psf_shape[1] != img_shape[2]:

        target_side_length = psf_shape[1]
        height_pad = (target_side_length - img_shape[1]) / 2
        width_pad = (target_side_length - img_shape[1]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))
        img = tf.pad(tensor=img, paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                     mode="SYMMETRIC")
        img_shape = img.shape.as_list()
    else:
        circular = True

    img_fft = transpose_2d_fft(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(a=otf, perm=[2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)

    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transpose_2d_ifft(img_fft * tf.math.conj(otf))
    else:
        result = transpose_2d_ifft(img_fft * otf)
    result = tf.cast(tf.math.real(result), tf.float32)

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]
    return result


def depth_dep_convolution(img, psfs, disc_depth_map):
    img = tf.cast(img, dtype=tf.float32)
    input_shape = img.shape.as_list()

    zeros_tensor = tf.zeros_like(img, dtype=tf.float32)
    disc_depth_map = tf.tile(tf.cast(disc_depth_map, tf.int16),
                             multiples=[1, 1, 1, input_shape[3]])
    blurred_imgs = []
    for depth_idx, psf in enumerate(psfs):
        psf = tf.cast(psf, dtype=tf.float32)
        condition = tf.equal(disc_depth_map, tf.convert_to_tensor(value=depth_idx, dtype=tf.int16))
        blurred_img = image_convolve_with_psf(img, psf)
        blurred_imgs.append(tf.where(condition, blurred_img, zeros_tensor))

    result = tf.reduce_sum(input_tensor=blurred_imgs, axis=0)
    return result


def get_spherical_wavefront_phase(resolution, physical_size, wave_lengths, source_distance):
    source_distance = tf.cast(source_distance, tf.float64)
    physical_size = tf.cast(physical_size, tf.float64)
    wave_lengths = tf.cast(wave_lengths, tf.float64)
    N, M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)
    x = x / N * physical_size
    y = y / M * physical_size
    curvature = tf.sqrt(x ** 2 + y ** 2 + source_distance ** 2)
    wave_nos = 2. * np.pi / wave_lengths
    phase_shifts = complex_exponent_tf(wave_nos * curvature)
    phase_shifts = tf.expand_dims(tf.expand_dims(phase_shifts, 0), -1)
    return phase_shifts


def least_common_multiple(a, b):
    return abs(a * b) / math.gcd(a, b) if a and b else 0


def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = tf.nn.avg_pool2d(input=input_image,
                                      ksize=[1, factor, factor, 1],
                                      strides=[1, factor, factor, 1],
                                      padding="VALID")
    else:
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize(input_image, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                        size=2 * [upsample_factor * target_side_length])
        output_img = tf.nn.avg_pool2d(input=img_upsampled, ksize=[1, upsample_factor, upsample_factor, 1],
                                      strides=[1, upsample_factor, upsample_factor, 1], padding="VALID")
    return output_img


def hyperboidal_phase(resolution, am):
    wavelength_nominal = 633E-9

    pixelsX = resolution
    pixelsY = resolution
    dx = 4e-6
    dy = 4e-6
    xa = np.linspace(0, pixelsX - 1, pixelsX) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(0, pixelsY - 1, pixelsY) * dy
    ya = ya - np.mean(ya)
    [y_mesh, x_mesh] = np.meshgrid(ya, xa)

    x_mesh = x_mesh[np.newaxis, :, :]
    y_mesh = y_mesh[np.newaxis, :, :]

    f = 50e-3

    phase_fun = True
    R = pixelsX * dx / 2.0
    phase_def = 2 * np.pi / wavelength_nominal * (f - np.sqrt(x_mesh ** 2 + y_mesh ** 2 + f ** 2))

    if phase_fun:
        m = (x_mesh ** 2 + y_mesh ** 2) / R ** 2
        phase_fun = am[0] * m ** 1 + am[1] * m ** 2 + am[2] * m ** 3 + am[3] * m ** 4 + am[4] * m ** 5 + am[
            5] * m ** 6 + am[6] * m ** 7 + am[7] * m ** 8
        phase_def = phase_def + phase_fun

    phase_def = tf.convert_to_tensor(phase_def, dtype=tf.float32)
    mask = ((x_mesh ** 2 + y_mesh ** 2) < R ** 2)
    phase_def = phase_def * mask
    return phase_def


def hyperboidal_phase_meta(resolution, am):
    wavelength_nominal = 452E-9

    pixelsX = resolution
    pixelsY = resolution
    dx = 350e-9
    dy = 350e-9
    xa = np.linspace(0, pixelsX - 1, pixelsX) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(0, pixelsY - 1, pixelsY) * dy
    ya = ya - np.mean(ya)
    [y_mesh, x_mesh] = np.meshgrid(ya, xa)

    x_mesh = x_mesh[np.newaxis, :, :]
    y_mesh = y_mesh[np.newaxis, :, :]

    f = 1e-3

    phase_fun = True
    R = pixelsX * dx / 2.0
    phase_def = 2 * np.pi / wavelength_nominal * (f - np.sqrt(x_mesh ** 2 + y_mesh ** 2 + f ** 2))

    if phase_fun:
        m = (x_mesh ** 2 + y_mesh ** 2) / R ** 2
        phase_fun = am[0] * m ** 1 + am[1] * m ** 2 + am[2] * m ** 3 + am[3] * m ** 4 + am[4] * m ** 5 + am[
            5] * m ** 6 + am[6] * m ** 7 + am[7] * m ** 8
        phase_def = phase_def + phase_fun

    phase_def = phase_def - (tf.floor(phase_def / (2 * np.pi)) * (2 * np.pi)) - (0.41 * np.pi)
    phase_def = tf.convert_to_tensor(phase_def, dtype=tf.float32)
    mask = ((x_mesh ** 2 + y_mesh ** 2) < R ** 2)
    phase_def = phase_def * mask
    return phase_def


def hyperboidal_phase_meta_traditional(resolution, am):
    wavelength_nominal = 452E-9

    pixelsX = resolution
    pixelsY = resolution
    dx = 350e-9
    dy = 350e-9
    xa = np.linspace(0, pixelsX - 1, pixelsX) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(0, pixelsY - 1, pixelsY) * dy
    ya = ya - np.mean(ya)
    [y_mesh, x_mesh] = np.meshgrid(ya, xa)

    x_mesh = x_mesh[np.newaxis, :, :]
    y_mesh = y_mesh[np.newaxis, :, :]

    f = 1e-3

    R = pixelsX * dx / 2.0
    phase_def = 2 * np.pi / wavelength_nominal * (f - np.sqrt(x_mesh ** 2 + y_mesh ** 2 + f ** 2))
    phase_def = phase_def.astype(np.float32)

    phase_def = phase_def - (tf.floor(phase_def / (2 * np.pi)) * (2 * np.pi)) - (0.41 * np.pi)
    phase_def = tf.convert_to_tensor(phase_def, dtype=tf.float32)
    mask = ((x_mesh ** 2 + y_mesh ** 2) < R ** 2)
    phase_def = phase_def * mask
    return phase_def
