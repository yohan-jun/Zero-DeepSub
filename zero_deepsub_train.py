import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils_subspace
import parser_ops
import UnrollNet_subspace
from multiprocessing import Pool
from multiprocessing import cpu_count
parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 42

if args.transfer_learning:
    print('Getting weights from trained model:')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    loadChkPoint_tl = tf.train.latest_checkpoint(args.TL_path)
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(args.TL_path + '/model_test.meta')
        new_saver.restore(sess, loadChkPoint_tl)
        trainable_variable_collections = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        pretrained_model_weights = [sess.run(v) for v in trainable_variable_collections]


save_dir ='saved_models'
directory = os.path.join(save_dir, 'ZS_SSL_Subspace' + args.data_opt + '_Rate'+ str(args.acc_rate)+'_'+ str(args.num_reps)+'reps')
if not os.path.exists(directory):
    os.makedirs(directory)

print('..... Create a test model for the testing \n')
test_graph_generator = tf_utils_subspace.test_graph(directory)

start_time = time.time()
print('..... ZS-SSL subspace training \n')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

print('..... Loading data for training \n')
data = sio.loadmat(args.data_dir) 
kspace_train, sens_maps, original_mask, basis = data['kspace'], data['sens_maps'], data['mask'], data['basis']

print('..... Data shape ')
print('kspace:', kspace_train.shape, ', sensitivity maps:', sens_maps.shape, ', mask:', original_mask.shape, ', basis:', basis.shape, '\n')

print('..... Normalize the kspace to 0-1 region \n')
kspace_train = kspace_train / np.max(np.abs(kspace_train[:]))

print('..... Generate validation mask \n')
cv_trn_mask = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB), dtype=np.float32)
cv_val_mask = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB), dtype=np.complex64)
remainder_mask = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB), dtype=np.float32)

if args.kspace_sum_over_etl == 1:
    for ee in range(args.necho_GLOB):
        cv_trn_mask_, cv_val_mask_ = utils.uniform_selection(np.sum(kspace_train[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=3), \
                                                                np.sum(original_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=2), \
                                                                rho=args.rho_val, \
                                                                small_acs_block=(4, 4), seed=seed)
        cv_trn_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)] = cv_trn_mask_[..., np.newaxis] * original_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)]
        cv_val_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)] = cv_val_mask_[..., np.newaxis] * original_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)]
else:
    for ee in range(args.netl_GLOB * args.necho_GLOB):
        cv_trn_mask[..., ee], cv_val_mask[..., ee] = utils.uniform_selection(kspace_train[..., ee], original_mask[..., ee], rho=args.rho_val, small_acs_block=(4, 4), seed=seed)
remainder_mask, cv_val_mask = np.copy(cv_trn_mask), np.copy(np.complex64(cv_val_mask))

print('..... Generate validation data \n')
val_stk = np.reshape(cv_trn_mask, [args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB, 1, 1]) * \
            np.reshape(basis, [1, 1, args.netl_GLOB * args.necho_GLOB, 1, args.nbasis_GLOB])
val_stk = np.expand_dims(np.sum(val_stk * np.reshape(basis, [1, 1, args.netl_GLOB * args.necho_GLOB, args.nbasis_GLOB, 1]), axis=2), axis=2)

if args.kspace_sum_over_etl == 1:
    ref_kspace_val = np.empty((args.batchSize, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
else:
    ref_kspace_val = np.empty((args.batchSize, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB * 2), dtype=np.float32)
val_img = np.empty((args.batchSize, args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB * 2), dtype=np.float32)
nw_input_val = np.empty((args.batchSize, args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB * 2), dtype=np.float32)

ref_kspace_val_ = kspace_train * np.tile(cv_val_mask[..., np.newaxis, :], (1, 1, args.ncoil_GLOB, 1))[np.newaxis]
if args.kspace_sum_over_etl == 1:
    for ee in range(args.necho_GLOB):
        ref_kspace_val[..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(np.sum(ref_kspace_val_[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=-1), (0, 3, 1, 2)))
else:
    for ee in range(args.netl_GLOB * args.necho_GLOB):
        ref_kspace_val[..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(ref_kspace_val_[..., ee], (0, 3, 1, 2)))
for ee in range(args.netl_GLOB * args.necho_GLOB):
    val_img[..., ee * 2:(ee + 1) * 2] = utils.complex2real(
                                                utils.sense1(kspace_train[..., ee] * np.tile(cv_trn_mask[..., ee][..., np.newaxis], (1, 1, args.ncoil_GLOB)), \
                                                                sens_maps)[np.newaxis])

for bb in range(args.batchSize):
    val_sub_r = np.reshape(np.matmul(np.reshape(val_img[bb, ..., 0::2], [args.nrow_GLOB * args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB]), basis), \
                            [args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB])
    val_sub_i = np.reshape(np.matmul(np.reshape(val_img[bb, ..., 1::2], [args.nrow_GLOB * args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB]), basis), \
                            [args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB])
    for kk in range(args.nbasis_GLOB):
        val_sub_ = np.concatenate((val_sub_r[..., kk][..., np.newaxis], val_sub_i[..., kk][..., np.newaxis]), axis=-1)
        if kk == 0:
            val_sub = val_sub_
        else:
            val_sub = np.concatenate((val_sub, val_sub_), axis=-1)
    nw_input_val[bb, ...] = val_sub

sens_maps_trn = np.transpose(np.tile(sens_maps[np.newaxis],(args.num_reps, 1, 1, 1)), (0, 3, 1, 2))
basis = np.tile(basis[np.newaxis], (args.num_reps, 1, 1))

print('..... Make tf. placeholder & dataset \n')
if args.kspace_sum_over_etl == 1:
    kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, args.necho_GLOB * 2), name='refkspace')
else:
    kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, args.netl_GLOB * args.necho_GLOB * 2), name='refkspace')
sens_mapsP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
basisP = tf.placeholder(tf.float32, shape=(None, None, None), name='basis')
stkP = tf.placeholder(tf.float32, shape=(None, None, None, None, None, None), name='stk')
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='trn_mask')
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='loss_mask')
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB * 2), name='nw_input')

train_dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, basisP, stkP, trn_maskP, loss_maskP)).shuffle(buffer_size = 10 * args.batchSize).batch(args.batchSize)
cv_dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, basisP, stkP, trn_maskP, loss_maskP)).shuffle(buffer_size = 10 * args.batchSize).batch(args.batchSize)
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_iterator = iterator.make_initializer(train_dataset)
cv_iterator = iterator.make_initializer(cv_dataset)

ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, basis_tensor, stk_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next('getNext')

print('..... Make training model \n')
nw_output_img, nw_output_kspace, *_ = UnrollNet_subspace.UnrolledNet(nw_input_tensor, sens_maps_tensor, basis_tensor, stk_tensor, trn_mask_tensor, loss_mask_tensor).model

scalar = tf.constant(0.5, dtype=tf.float32)

loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))

trn_mask, loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB), dtype=np.complex64), \
                                np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB), dtype=np.complex64)
trn_stk = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, 1, args.nbasis_GLOB, args.nbasis_GLOB), dtype=np.float32)

if args.kspace_sum_over_etl == 1:
    ref_kspace = np.empty((args.num_reps, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
else:
    ref_kspace = np.empty((args.num_reps, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB * 2), dtype=np.float32)
trn_img = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB * 2), dtype=np.float32)
nw_input = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB * 2), dtype=np.float32)

if args.mask_gen_parallel_computation == 0:
    for jj in range(args.num_reps):
        tic = time.time()
        if args.kspace_sum_over_etl == 1:
            for ee in range(args.necho_GLOB):
                trn_mask_, loss_mask_ = utils.uniform_selection(np.sum(kspace_train[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=3), \
                                                                np.sum(remainder_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=2), \
                                                                rho=args.rho_train, \
                                                                small_acs_block=(4, 4))
                trn_mask[jj, ..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)] = trn_mask_[..., np.newaxis] * original_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)]
                loss_mask[jj, ..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)] = loss_mask_[..., np.newaxis] * original_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)]
        else:
            for ee in range(args.netl_GLOB * args.necho_GLOB):
                trn_mask[jj, ..., ee], loss_mask[jj, ..., ee] = utils.uniform_selection(kspace_train[..., ee], remainder_mask[..., ee], rho=args.rho_train, small_acs_block=(4, 4))

        sub_kspace = kspace_train * np.tile(trn_mask[jj, ..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))
        ref_kspace_ = kspace_train * np.tile(loss_mask[jj, ..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))

        trn_stk_ = np.reshape(np.real(trn_mask[jj, ...]), [args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB, 1, 1]) * \
                            np.reshape(basis[jj, ...], [1, 1, args.netl_GLOB * args.necho_GLOB, 1, args.nbasis_GLOB])
        trn_stk[jj, ...] = np.expand_dims(np.sum(trn_stk_ * np.reshape(basis[jj, ...], [1, 1, args.netl_GLOB * args.necho_GLOB, args.nbasis_GLOB, 1]), axis=2), axis=2)

        if args.kspace_sum_over_etl == 1:
            for ee in range(args.necho_GLOB):
                ref_kspace[jj, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(np.sum(ref_kspace_[np.newaxis, ..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=-1), (0, 3, 1, 2)))
        else:
            for ee in range(args.netl_GLOB * args.necho_GLOB):
                ref_kspace[jj, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(ref_kspace_[np.newaxis, ..., ee], (0, 3, 1, 2)))

        for ee in range(args.netl_GLOB * args.necho_GLOB):
            trn_img[..., ee * 2:(ee + 1) * 2] = utils.complex2real(utils.sense1(sub_kspace[..., ee], sens_maps))

        trn_sub_r = np.reshape(np.matmul(np.reshape(trn_img[..., 0::2], [args.nrow_GLOB * args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB]), basis[jj, ...]), \
                                [args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB])
        trn_sub_i = np.reshape(np.matmul(np.reshape(trn_img[..., 1::2], [args.nrow_GLOB * args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB]), basis[jj, ...]), \
                                [args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB])
        for kk in range(args.nbasis_GLOB):
            trn_sub_ = np.concatenate((trn_sub_r[..., kk][..., np.newaxis], trn_sub_i[..., kk][..., np.newaxis]), axis=-1)
            if kk == 0:
                trn_sub = trn_sub_
            else:
                trn_sub = np.concatenate((trn_sub, trn_sub_), axis=-1)
        nw_input[jj, ...] = trn_sub
        toc = time.time() - tic
        print("..... making multi-mask:", jj, "elapsed_time = ""{:.2f}".format(toc))

def make_data_reps(jj):
    tic = time.time()

    trn_mask_reps = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB), dtype=np.complex64)
    loss_mask_reps = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB), dtype=np.complex64)
    trn_stk_reps = np.empty((args.nrow_GLOB, args.ncol_GLOB, 1, args.nbasis_GLOB, args.nbasis_GLOB), dtype=np.float32)
    if args.kspace_sum_over_etl == 1:
        ref_kspace_reps = np.empty((args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
    else:
        ref_kspace_reps = np.empty((args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB * 2), dtype=np.float32)
    nw_input_reps = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB * 2), dtype=np.float32)

    if args.kspace_sum_over_etl == 1:
        for ee in range(args.necho_GLOB):
            trn_mask_, loss_mask_ = utils.uniform_selection(np.sum(kspace_train[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=3), \
                                                            np.sum(remainder_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=2), \
                                                            rho=args.rho_train, \
                                                            small_acs_block=(4, 4))
            trn_mask_reps[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)] = trn_mask_[..., np.newaxis] * original_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)]
            loss_mask_reps[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)] = loss_mask_[..., np.newaxis] * original_mask[..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)]
    else:
        for ee in range(args.netl_GLOB * args.necho_GLOB):
            trn_mask_reps[..., ee], loss_mask_reps[..., ee] = utils.uniform_selection(kspace_train[..., ee], remainder_mask[..., ee], rho=args.rho_train, small_acs_block=(4, 4))

    sub_kspace = kspace_train * np.tile(trn_mask_reps[..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))
    ref_kspace_ = kspace_train * np.tile(loss_mask_reps[..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))

    trn_stk_ = np.reshape(np.real(trn_mask_reps), [args.nrow_GLOB, args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB, 1, 1]) * \
                        np.reshape(basis[jj, ...], [1, 1, args.netl_GLOB * args.necho_GLOB, 1, args.nbasis_GLOB])
    trn_stk_reps = np.expand_dims(np.sum(trn_stk_ * np.reshape(basis[jj, ...], [1, 1, args.netl_GLOB * args.necho_GLOB, args.nbasis_GLOB, 1]), axis=2), axis=2)

    if args.kspace_sum_over_etl == 1:
        for ee in range(args.necho_GLOB):
            ref_kspace_reps[..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(np.sum(ref_kspace_[np.newaxis, ..., args.netl_GLOB * ee:args.netl_GLOB * (ee + 1)], axis=-1), (0, 3, 1, 2)))
    else:
        for ee in range(args.netl_GLOB * args.necho_GLOB):
            ref_kspace_reps[..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(ref_kspace_[np.newaxis, ..., ee], (0, 3, 1, 2)))

    for ee in range(args.netl_GLOB * args.necho_GLOB):
        trn_img[..., ee * 2:(ee + 1) * 2] = utils.complex2real(utils.sense1(sub_kspace[..., ee], sens_maps))

    trn_sub_r = np.reshape(np.matmul(np.reshape(trn_img[..., 0::2], [args.nrow_GLOB * args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB]), basis[jj, ...]), \
                            [args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB])
    trn_sub_i = np.reshape(np.matmul(np.reshape(trn_img[..., 1::2], [args.nrow_GLOB * args.ncol_GLOB, args.netl_GLOB * args.necho_GLOB]), basis[jj, ...]), \
                            [args.nrow_GLOB, args.ncol_GLOB, args.nbasis_GLOB])
    for kk in range(args.nbasis_GLOB):
        trn_sub_ = np.concatenate((trn_sub_r[..., kk][..., np.newaxis], trn_sub_i[..., kk][..., np.newaxis]), axis=-1)
        if kk == 0:
            trn_sub = trn_sub_
        else:
            trn_sub = np.concatenate((trn_sub, trn_sub_), axis=-1)
    nw_input_reps = trn_sub
    toc = time.time() - tic
    print("..... making multi-mask:", jj, "elapsed_time = ""{:.2f}".format(toc))
    return trn_mask_reps, loss_mask_reps, trn_stk_reps, ref_kspace_reps, nw_input_reps

if args.mask_gen_parallel_computation == 1:
    num_parallel = min([int(cpu_count()/2), args.num_reps])
    pool = Pool(num_parallel)
    trn_mask_reps, loss_mask_reps, trn_stk_reps, ref_kspace_reps, nw_input_reps = zip(*pool.map(make_data_reps, range(args.num_reps)))
    pool.close()
    for jj in range(args.num_reps):
        trn_mask[jj, ...], loss_mask[jj, ...], trn_stk[jj, ...], ref_kspace[jj, ...], nw_input[jj, ...] = \
        trn_mask_reps[jj], loss_mask_reps[jj], trn_stk_reps[jj], ref_kspace_reps[jj], nw_input_reps[jj]

    del trn_mask_reps, loss_mask_reps, trn_stk_reps, ref_kspace_reps, nw_input_reps

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=1)
sess_trn_filename = os.path.join(directory, 'model')
totalLoss, totalTime = [], []
total_val_loss = []
avg_cost = 0

print('..... Start tf.session \n')
lowest_val_loss = np.inf
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('Number of trainable parameters: ', sess.run(all_trainable_vars), '\n')
    if args.mask_gen_in_each_iter == 0:
        feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps_trn, basisP: basis, stkP: trn_stk}

    print('..... Training \n')
    if args.transfer_learning:
        print('transferring weights from pretrained model to the new model:')
        trainable_collection_test = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        initialize_model_weights = [v for v in trainable_collection_test]
        for ii in range(len(initialize_model_weights)):
            sess.run(initialize_model_weights[ii].assign(pretrained_model_weights[ii]))

    np.random.seed(seed)
    ep, val_loss_tracker = 0, 0
    while ep<args.epochs and val_loss_tracker<args.stop_training:
        if args.mask_gen_in_each_iter == 1:
            np.random.seed()
            jj = np.random.randint(args.num_reps)
            ref_kspace_batch, nw_input_batch = ref_kspace[jj, ...][np.newaxis], nw_input[jj, ...][np.newaxis]
            trn_mask_batch, loss_mask_batch = trn_mask[jj, ...][np.newaxis], loss_mask[jj, ...][np.newaxis]
            sens_maps_trn_batch, basis_batch, trn_stk_batch = sens_maps_trn[jj, ...][np.newaxis], basis[jj, ...][np.newaxis], trn_stk[jj, ...][np.newaxis]
            feedDict = {kspaceP: ref_kspace_batch, nw_inputP: nw_input_batch, trn_maskP: trn_mask_batch, loss_maskP: loss_mask_batch, \
                        sens_mapsP: sens_maps_trn_batch, basisP: basis_batch, stkP: trn_stk_batch}

        sess.run(train_iterator, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:
            if args.mask_gen_in_each_iter == 0:
                for jj in range(args.num_reps):
                    tmp, _, _ = sess.run([loss, update_ops, optimizer])
                    avg_cost += tmp / args.num_reps
            else:
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp

            toc = time.time() - tic
            totalLoss.append(avg_cost)
        except tf.errors.OutOfRangeError:
            pass
        sess.run(cv_iterator, feed_dict={kspaceP: ref_kspace_val, nw_inputP: nw_input_val, trn_maskP: cv_trn_mask[np.newaxis], loss_maskP: cv_val_mask[np.newaxis], \
                                            sens_mapsP: sens_maps_trn[0][np.newaxis], basisP: basis[0][np.newaxis], stkP: val_stk[np.newaxis]})
        val_loss = sess.run([loss])[0]
        total_val_loss.append(val_loss)
        print("Epoch:", ep, "elapsed_time =""{:.2f}".format(toc), "trn loss =", "{:.5f}".format(avg_cost)," val loss =", "{:.5f}".format(val_loss))
        if val_loss<=lowest_val_loss:
            lowest_val_loss = val_loss    
            saver.save(sess, sess_trn_filename, global_step=ep)
            val_loss_tracker = 0
        else:
            val_loss_tracker += 1
        sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'trn_loss': totalLoss, 'val_loss': total_val_loss})
        ep += 1

end_time = time.time()
print('Training completed in  ', str(ep), ' epochs, ',((end_time - start_time) / 60), ' minutes')
