import tensorflow as tf
import tf_utils_subspace
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()


class data_consistency():
    """
    Data consistency class can be used for:
        -performing E^h*E operation in the paper
        -transforming final network output to kspace
    """

    def __init__(self, sens_maps, mask, basis, stk):
        with tf.name_scope('EncoderParams'):
            self.shape_list = tf.shape(mask)
            self.sens_maps = sens_maps
            self.mask = mask
            self.basis = basis
            self.stk = stk
            self.shape_list = tf.shape(mask)
            self.scalar = tf.complex(tf.sqrt(tf.to_float(self.shape_list[0] * self.shape_list[1])), 0.)

    def EhE_Op(self, img, mu):
        """
        Performs (E^h*E+ mu*I) x
        """
        with tf.name_scope('EhE'):
            for kk in range(args.nbasis_GLOB):
                coil_imgs_ = self.sens_maps * img[..., kk]
                kspace_ = tf.expand_dims(tf_utils_subspace.tf_fftshift(tf.fft2d(tf_utils_subspace.tf_ifftshift(coil_imgs_))) / self.scalar, axis=-1)
                if kk == 0:
                    kspace = kspace_
                else:
                    kspace = tf.concat([kspace, kspace_], axis=-1)

            kspace_r = tf_utils_subspace.tf_complex2real(kspace)
            masked_kspace_r = tf.reduce_sum(tf.reshape(kspace_r[..., 0], [args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, 1, args.nbasis_GLOB]) * \
                                tf.transpose(self.stk, perm=[2, 0, 1, 3, 4]), axis=-1, keepdims=True)
            masked_kspace_i = tf.reduce_sum(tf.reshape(kspace_r[..., 1], [args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, 1, args.nbasis_GLOB]) * \
                                tf.transpose(self.stk, perm=[2, 0, 1, 3, 4]), axis=-1, keepdims=True)
            masked_kspace = tf_utils_subspace.tf_real2complex(tf.concat([masked_kspace_r, masked_kspace_i], axis=-1))
            masked_kspace = tf.reshape(masked_kspace, [args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB])

            for kk in range(args.nbasis_GLOB):
                masked_img_ = tf_utils_subspace.tf_ifftshift(tf.ifft2d(tf_utils_subspace.tf_fftshift(masked_kspace[..., kk]))) * self.scalar
                masked_img_comb_ = tf.expand_dims(tf.reduce_sum(masked_img_ * tf.conj(self.sens_maps), axis=0), axis=-1)
                if kk == 0:
                    masked_img_comb = masked_img_comb_
                else:
                    masked_img_comb = tf.concat([masked_img_comb, masked_img_comb_], axis=-1)

            ispace = masked_img_comb + mu * img

        return ispace

    def SSDU_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """

        with tf.name_scope('SSDU_kspace'):
            coil_imgs = self.sens_maps * img
            kspace = tf_utils_subspace.tf_fftshift(tf.fft2d(tf_utils_subspace.tf_ifftshift(coil_imgs))) / self.scalar
            masked_kspace = kspace * self.mask

        return masked_kspace

    def Supervised_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """

        with tf.name_scope('Supervised_kspace'):
            coil_imgs = self.sens_maps * img
            kspace = tf_utils_subspace.tf_fftshift(tf.fft2d(tf_utils_subspace.tf_ifftshift(coil_imgs))) / self.scalar

        return kspace
    
    def Supervised_single_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """

        with tf.name_scope('Supervised_kspace'):
            kspace = tf.squeeze(tf_utils_subspace.tf_fftshift(tf.fft2d(tf_utils_subspace.tf_ifftshift(tf.expand_dims(img, axis=0)))), axis=0) / self.scalar

        return kspace

    def Supervised_single_image(self, kspace):
        """
        Transforms unrolled network output to image
        """

        with tf.name_scope('Supervised_image'):
            image = tf.squeeze(tf_utils_subspace.tf_ifftshift(tf.ifft2d(tf_utils_subspace.tf_fftshift(tf.expand_dims(kspace, axis=0)))), axis=0) * self.scalar

        return image


def conj_grad(input_elems, mu_param):
    """
    Parameters
    ----------
    input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
    rhs = nrow x ncol x 2
    sens_maps : coil sensitivity maps ncoil x nrow x ncol
    mask : nrow x ncol
    mu : penalty parameter

    Encoder : Object instance for performing encoding matrix operations

    Returns
    -------
    data consistency output, nrow x ncol x 2

    """

    rhs, sens_maps, mask, basis, stk = input_elems
    mu_param = tf.complex(mu_param, 0.)
    for kk in range(args.nbasis_GLOB):
        rhs_c_ = tf.expand_dims(tf_utils_subspace.tf_real2complex(rhs[..., kk * 2:(kk + 1) * 2]), axis=-1)
        if kk == 0:
            rhs_c = rhs_c_
        else:
            rhs_c = tf.concat([rhs_c, rhs_c_], axis=-1)
    rhs = rhs_c
    Encoder = data_consistency(sens_maps, mask, basis, stk)
    cond = lambda i, *_: tf.less(i, args.CG_Iter)

    def body(i, rsold, x, r, p, mu):
        with tf.name_scope('CGIters'):
            Ap = Encoder.EhE_Op(p, mu)
            alpha = tf.complex(rsold / tf.to_float(tf.reduce_sum(tf.conj(p) * Ap)), 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = tf.to_float(tf.reduce_sum(tf.conj(r) * r))
            beta = rsnew / rsold
            beta = tf.complex(beta, 0.)
            p = r + beta * p

        return i + 1, rsnew, x, r, p, mu

    x = tf.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rsold = tf.to_float(tf.reduce_sum(tf.conj(r) * r), )
    loop_vars = i, rsold, x, r, p, mu_param
    cg_out = tf.while_loop(cond, body, loop_vars, name='CGloop', parallel_iterations=1)[2]

    for kk in range(args.nbasis_GLOB):
        cg_out_r_ = tf_utils_subspace.tf_complex2real(cg_out[..., kk])
        if kk == 0:
            cg_out_r = cg_out_r_
        else:
            cg_out_r = tf.concat([cg_out_r, cg_out_r_], axis=-1)
    return cg_out_r


def dc_block(rhs, sens_maps, mask, basis, stk, mu):
    """
    DC block employs conjugate gradient for data consistency,
    """

    def cg_map_func(input_elems):
        cg_output = conj_grad(input_elems, mu)

        return cg_output

    dc_block_output = tf.map_fn(cg_map_func, (rhs, sens_maps, mask, basis, stk), dtype=tf.float32, name='mapCG')

    return dc_block_output


def SSDU_kspace_transform(nw_output, sens_maps, mask, basis, stk):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    nw_output = tf_utils_subspace.tf_real2complex(nw_output)

    def ssdu_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc, basis, stk)
        nw_output_kspace = Encoder.SSDU_kspace(nw_output_enc)

        return nw_output_kspace

    masked_kspace = tf.map_fn(ssdu_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='ssdumapFn')

    return tf_utils_subspace.tf_complex2real(masked_kspace)


def Supervised_kspace_transform(nw_output, sens_maps, mask, basis, stk):
    """
    This function transforms unrolled network output to k-space
    """

    nw_output = tf_utils_subspace.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc, basis, stk)
        nw_output_kspace = Encoder.Supervised_kspace(nw_output_enc)

        return nw_output_kspace

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmapFn')

    return tf_utils_subspace.tf_complex2real(kspace)


def Supervised_single_kspace_transform(nw_output, sens_maps, mask, basis, stk):
    """
    This function transforms unrolled network output to k-space
    """

    nw_output = tf_utils_subspace.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc, basis, stk)
        nw_output_kspace = Encoder.Supervised_single_kspace(nw_output_enc)

        return nw_output_kspace

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmap_single_k_Fn')

    return tf_utils_subspace.tf_complex2real(kspace)


def Supervised_single_image_transform(nw_output, sens_maps, mask, basis, stk):
    """
    This function transforms unrolled network output to image
    """

    nw_output = tf_utils_subspace.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc, basis, stk)
        nw_output_image = Encoder.Supervised_single_image(nw_output_enc)

        return nw_output_image

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmap_single_i_Fn')

    return tf_utils_subspace.tf_complex2real(kspace)

