import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, Input, \
    Activation, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from networks.util_25 import faiT_J, fai_I, inverse_copy_simulated_rgb_channel_camera_spectral_response_function
from einops import rearrange


def _channel_attention_block(x, kernel_size=3, name="attention_blk"):
    attention_conv = Conv1D(filters=1, kernel_size=kernel_size, strides=1,
                            use_bias=False, padding="same", name="%s_attn_conv" % name)
    avg_pool = attention_conv(
        tf.expand_dims(
            GlobalAveragePooling2D(name="%s_attn_avg_pool" % name)(x),
            axis=-1
        )
    )
    max_pool = attention_conv(
        tf.expand_dims(
            GlobalMaxPooling2D(name="%s_attn_max_pool" % name)(x),
            axis=-1
        )
    )
    attention_feature = Activation('sigmoid', name="%s_attn_sigmoid" % name)(avg_pool + max_pool)
    attention_feature = tf.expand_dims(attention_feature, axis=-1)
    attention_feature = tf.transpose(attention_feature, perm=[0, 2, 3, 1])
    return attention_feature * x


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class HS_MSA(tf.keras.layers.Layer):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=32,
            heads=8,
            size=512,
            only_local_branch=False
    ):
        super(HS_MSA, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch
        self.size = size

        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = tf.Variable(tf.random.truncated_normal((1, heads, seq_l, seq_l)))
        else:
            self.heads = heads
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = tf.Variable(tf.random.truncated_normal((1, 1, heads // 2, seq_l1, seq_l1)))
            h, w = self.size, self.size
            seq_l2 = h * w // seq_l1
            self.pos_emb2 = tf.Variable(tf.random.truncated_normal((1, 1, heads // 2, seq_l2, seq_l2)))

        inner_dim = dim_head * heads
        self.inner_dim = inner_dim  # 32

        self.to_q = tf.keras.layers.Dense(inner_dim, use_bias=False)
        self.to_kv = tf.keras.layers.Dense(inner_dim * 2, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)

    def call(self, x):
        super(HS_MSA, self).__init__()
        b, h, w, c = x.shape
        dh = self.dim // self.heads
        w_size = self.window_size
        if self.only_local_branch:
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])

            q = self.to_q(x_inp)

            k, v = tf.split(self.to_kv(x_inp), num_or_size_splits=2, axis=-1)
            q, k, v = map(lambda t: rearrange(t, 'b t n (h d) -> b t h n d', h=self.heads), (q, k, v))

            q *= self.scale
            sim = tf.einsum('b t h i d, b t h j d -> b t h i j', q, k)
            sim = sim + self.pos_emb

            attn = tf.nn.softmax(sim, axis=-1)

            out = tf.einsum('b t h i j, b t h j d -> b t h i d', attn, v)
            out = rearrange(out, 'b t h n d -> b t n (h d)')
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])

        else:

            q = self.to_q(x)

            k, v = tf.split(self.to_kv(x), num_or_size_splits=2, axis=-1)

            c = self.inner_dim
            q1, q2 = q[:, :, :, :c // 2], q[:, :, :, c // 2:]
            k1, k2 = k[:, :, :, :c // 2], k[:, :, :, c // 2:]
            v1, v2 = v[:, :, :, :c // 2], v[:, :, :, c // 2:]

            q1, k1, v1 = map(
                lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size[0], b1=w_size[1]),
                (q1, k1, v1))

            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads // 2), (q1, k1, v1))

            q1 *= self.scale
            sim1 = tf.einsum('b n h i d, b n h j d -> b n h i j', q1, k1)

            sim1 = sim1 + self.pos_emb1

            attn1 = tf.nn.softmax(sim1, axis=-1)
            out1 = tf.einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: tf.transpose(t, perm=[0, 2, 1, 3]),
                             (tf.identity(q2), tf.identity(k2), tf.identity(v2)))

            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads // 2),
                             (q2, k2, v2))  # x=tf.transpose(x, perm=[0, 1, 3, 2])  tf.transpose(t, perm=[0, 2, 1, 3])
            q2 *= self.scale
            sim2 = tf.einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = tf.nn.softmax(sim2, axis=-1)
            out2 = tf.einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = tf.transpose(out2, perm=[0, 2, 1, 3])

            out = tf.concat([out1, out2], axis=-1)
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])

        return out


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim * mult, 1, 1, padding='valid', use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(3, 1, padding='same', depth_multiplier=mult, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(dim, 1, 1, padding='valid', use_bias=False)
        ])

    def call(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """

        out = self.net(x)

        return out


class HSAB(tf.keras.Model):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,
            size=512,

    ):
        super(HSAB, self).__init__()
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads,
                                    only_local_branch=(heads == 1), size=size)),
                PreNorm(dim, FeedForward(dim=dim))
            ])
        self.dim = dim

    def call(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """

        x = tf.transpose(x, [0, 2, 1, 3])  # 0, 2, 1, 3
        for attn, ff in self.blocks:
            x = attn(x) + x

            x = ff(x) + x

        out = x
        return out


def reconxxxxxx(filter_root, depth, output_channel=25, input_size=(512, 512, 25), activation='elu',
                batch_norm=True, batch_norm_after_activation=False, final_activation='sigmoid', net_num=1,
                extra_upsampling=False, remove_first_long_connection=False, channel_attention=False,
                kernel_initializer='glorot_uniform', final_kernel_initializer='glorot_uniform'):
    input_size = (512, 512, 25)
    inputss = Input(input_size)
    x = inputss

    in_dim = 25
    out_dim = 25
    dim = 25
    num_blocks = [1, 1, 1]
    scales = len(num_blocks)

    encoder_layers = []
    dim_scale = dim

    b, h_inp, w_inp, c = x.shape
    hb, wb = 16, 16
    pad_h = (hb - h_inp % hb) % hb
    pad_w = (wb - w_inp % wb) % wb

    embedding = Conv2D(dim, 3, 1, 'same', use_bias=False)
    bottleneck = HSAB(dim=dim_scale * 4, dim_head=dim, heads=dim_scale * 4 // dim, num_blocks=num_blocks[-1],
                      size=h_inp // 4)
    mapping = Conv2D(out_dim, 3, 1, 'same', use_bias=False)

    Pre_hsab1 = embedding(x)

    x1 = HSAB(dim=dim_scale, num_blocks=1, dim_head=dim, heads=dim_scale // dim, size=h_inp)(Pre_hsab1)

    x = Conv2D(dim_scale * 2, 4, 2, 'same', use_bias=False)(x1)
    x2 = HSAB(dim=dim_scale * 2, num_blocks=1, dim_head=dim, heads=dim_scale * 2 // dim, size=h_inp // 2)(x)
    x = Conv2D(dim_scale * 4, 4, 2, 'same', use_bias=False)(x2)
    x = bottleneck(x)
    x = tf.keras.layers.Conv2DTranspose(dim_scale * 4, 2, 2, 'same')(x)
    x = Conv2D(dim_scale * 2, 1, 1, 'same', use_bias=False)(tf.concat([x, x2], axis=3))
    x = HSAB(dim=dim_scale * 2, num_blocks=1, dim_head=dim, heads=dim_scale * 2 // dim, size=h_inp // 2)(x)
    x = tf.keras.layers.Conv2DTranspose(dim_scale * 2, 2, 2, 'same')(x)
    x = Conv2D(dim_scale, 1, 1, 'same', use_bias=False)(tf.concat([x, x1], axis=3))
    x = HSAB(dim=dim_scale, num_blocks=1, dim_head=dim, heads=dim_scale // dim, size=h_inp)(x)

    out = mapping(x) + x

    model2 = Model(inputss, outputs=out, name='res-block-u-net1')
    return model2


class Res_network(Layer):
    def __init__(self, filter_root, depth, output_channel=25, input_size=(512, 512, 3), activation='elu',
                 batch_norm=True, batch_norm_after_activation=False, final_activation='sigmoid', net_num=1,
                 extra_upsampling=False, remove_first_long_connection=False, channel_attention=False,
                 kernel_initializer='glorot_uniform', final_kernel_initializer='glorot_uniform', **kwargs):
        self.depth = depth
        self.filter_root = filter_root
        self.output_channel = output_channel
        self.input_size = input_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm_after_activation = batch_norm_after_activation
        self.final_activation = final_activation
        self.net_num = net_num
        self.extra_upsampling = extra_upsampling
        self.remove_first_long_connection = remove_first_long_connection
        self.channel_attention = channel_attention
        self.kernel_initializer = kernel_initializer
        self.final_kernel_initializer = final_kernel_initializer

        super(Res_network, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = reconxxxxxx(self.filter_root, self.depth, self.output_channel, self.input_size, self.activation,
                                self.batch_norm, self.batch_norm_after_activation, self.final_activation, self.net_num,
                                self.extra_upsampling, self.remove_first_long_connection, self.channel_attention,
                                self.kernel_initializer, self.final_kernel_initializer)
        super(Res_network, self).build(input_shape)

    def call(self, inputs):
        output = self.conv(inputs)
        return output


def get_res_block_u_net(filter_root, depth, output_channel=25, input_size=(512, 512, 3), activation='elu',
                        batch_norm=True, batch_norm_after_activation=False, final_activation='sigmoid', net_num=1,
                        extra_upsampling=False, remove_first_long_connection=False, channel_attention=False,
                        kernel_initializer='glorot_uniform', final_kernel_initializer='glorot_uniform'):
    assert net_num >= 1, "There should be at least one network."
    print("Network Args:")

    print("activation=", activation)
    print("final_activation=", final_activation)
    print("kernel_initializer=", kernel_initializer)
    print("final_kernel_initializer=", final_kernel_initializer)
    input_size = (512, 512, 25)
    input_size0 = (512, 512, 3)

    inputs = Input(input_size0)
    inputs2 = Input(input_size)

    I0 = faiT_J(inputs, inputs2)  ##########################################

    print(filter_root)

    def Recon_block(It, I0, psfs, layer_no):
        deta = tf.Variable(0.1, dtype=tf.float32, name='deta_%d' % layer_no)
        eta = tf.Variable(0.9, dtype=tf.float32, name='eta_%d' % layer_no)

        channelNum = 25

        filter_size1 = 3
        filter_size2 = 1
        filter_num = 64

        depth = 7
        H = Res_network(filter_root=32, depth=7, output_channel=25, input_size=(512, 512, 3), activation='elu',
                        batch_norm=True, batch_norm_after_activation=False, final_activation='sigmoid', net_num=1,
                        extra_upsampling=False, remove_first_long_connection=False, channel_attention=False,
                        kernel_initializer='glorot_uniform', final_kernel_initializer='glorot_uniform')(It)
        psfsk = tf.transpose(psfs, perm=[1, 2, 0,
                                         3])  #######################################################################################消融实验
        faii = fai_I(It, psfsk)

        It2 = faiT_J(faii, psfs)  # PhiT*Phi*xt fai_IfaiT_J
        x = tf.scalar_mul(1 - deta * eta, It) - tf.scalar_mul(deta, It2) + tf.scalar_mul(deta, I0) + tf.scalar_mul(
            deta * eta, H)

        return x

    def inference_ista(x, n, psfs, reuse):
        xt = x

        for i in range(n):
            with tf.compat.v1.variable_scope('Phase_%d' % i, reuse=reuse):
                xt = Recon_block(xt, x, psfs, i)

                if i == 0:
                    xk1 = xt
                if i == 1:
                    xk2 = xt
                if i == 2:
                    xk3 = xt
                if i == 3:
                    xk4 = xt
                if i == 4:
                    xk5 = xt
                if i == 5:
                    xk6 = xt
                if i == 6:
                    xk7 = xt
                if i == 7:
                    xk8 = xt
                if i == 8:
                    xk9 = xt
                if i == 9:
                    xk10 = xt
        xk = tf.stack([xk1, xk2], axis=0)

        return xk

    n = 2
    output = inference_ista(I0, n, inputs2, reuse=False)

    model = Model([inputs, inputs2], outputs=[output, inputs2], name='res-block-u-net')

    for layer in model.layers:
        if "kernel_regularizer" in layer.__dict__:
            layer.kernel_regularizer = tf.keras.regularizers.l2(l2=0.0001)  # 0.0001
    return model
