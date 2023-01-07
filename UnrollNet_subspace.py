import tensorflow as tf
import data_consistency_subspace as ssdu_dc
import tf_utils_subspace
import models.networks_subspace as networks
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()


class UnrolledNet():

    def __init__(self, input_x, sens_maps, basis, stk, trn_mask, loss_mask):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.basis = basis
        self.stk = stk
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.shape_list = tf.shape(trn_mask)
        self.scalar = tf.complex(tf.sqrt(tf.to_float(self.shape_list[1] * self.shape_list[2])), 0.)
        self.model = self.Unrolled_SSDU()

    def Unrolled_SSDU(self):
        x, denoiser_output, dc_output = self.input_x, self.input_x, self.input_x
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(args.nb_unroll_blocks)]

        mu_init = tf.constant(0., dtype=tf.float32)
        x = ssdu_dc.dc_block(self.input_x, self.sens_maps, self.trn_mask, self.basis, self.stk, mu_init)
        x0 = x

        with tf.name_scope('SSDUModel'):
            with tf.variable_scope('Weights', reuse=tf.AUTO_REUSE):
                for i in range(args.nb_unroll_blocks):
                    x = networks.ResNet(x, args.nb_res_blocks, args.nbasis_GLOB)
                    denoiser_output = x

                    mu = networks.mu_param()
                    rhs = self.input_x + mu * x

                    x = ssdu_dc.dc_block(rhs, self.sens_maps, self.trn_mask, self.basis, self.stk, mu)
                    dc_output = x

                    for kk in range(args.nbasis_GLOB):
                        denoiser_output_c_ = tf.squeeze(denoiser_output)
                        denoiser_output_c_ = tf.expand_dims(tf_utils_subspace.tf_real2complex(denoiser_output_c_[..., kk * 2:(kk + 1) * 2]), axis=-1)
                        dc_output_c_ = tf.squeeze(dc_output)
                        dc_output_c_ = tf.expand_dims(tf_utils_subspace.tf_real2complex(dc_output_c_[..., kk * 2:(kk + 1) * 2]), axis=-1)
                        if kk == 0:
                            denoiser_output_c = denoiser_output_c_
                            dc_output_c = dc_output_c_
                        else:
                            denoiser_output_c = tf.concat([denoiser_output_c, denoiser_output_c_], axis=-1)
                            dc_output_c = tf.concat([dc_output_c, dc_output_c_], axis=-1)
                    all_intermediate_results[i][0] = denoiser_output_c
                    all_intermediate_results[i][1] = dc_output_c

            for kk in range(args.nbasis_GLOB):
                kspace_ = tf.expand_dims(ssdu_dc.Supervised_kspace_transform(x[..., kk * 2:(kk + 1) * 2], self.sens_maps, self.loss_mask, self.basis, self.stk), axis=-1)
                if kk == 0:
                    kspace = kspace_
                else:
                    kspace = tf.concat([kspace, kspace_], axis=-1)
            sub_r = tf.matmul(tf.reshape(kspace[..., 0, :], [-1, args.ncoil_GLOB * args.nrow_GLOB * args.ncol_GLOB, args.nbasis_GLOB]), tf.transpose(self.basis, perm=[0, 2, 1]))
            sub_i = tf.matmul(tf.reshape(kspace[..., 1, :], [-1, args.ncoil_GLOB * args.nrow_GLOB * args.ncol_GLOB, args.nbasis_GLOB]), tf.transpose(self.basis, perm=[0, 2, 1]))
            sub = tf_utils_subspace.tf_real2complex(tf.concat([tf.expand_dims(sub_r, axis=-1), tf.expand_dims(sub_i, axis=-1)], axis=-1))
            sub = tf.reshape(sub, [-1, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB])
            masked_sub = sub * self.loss_mask

            if args.kspace_sum_over_etl == 1:
                for ee in range(args.necho_GLOB):
                    nw_kspace_output_ = tf_utils_subspace.tf_complex2real(tf.reduce_sum(masked_sub[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=-1))
                    if ee == 0:
                        nw_kspace_output = nw_kspace_output_
                    else:
                        nw_kspace_output = tf.concat([nw_kspace_output, nw_kspace_output_], axis=-1)
            else:
                for ee in range(args.netl_GLOB * args.necho_GLOB):
                    nw_kspace_output_ = tf_utils_subspace.tf_complex2real(masked_sub[..., ee])
                    if ee == 0:
                        nw_kspace_output = nw_kspace_output_
                    else:
                        nw_kspace_output = tf.concat([nw_kspace_output, nw_kspace_output_], axis=-1)

        return x, nw_kspace_output, x0, all_intermediate_results, mu
