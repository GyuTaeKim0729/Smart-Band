
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import time

from six.moves import xrange
import tensorflow as tf
import quant_models as models
from tensorflow.python.framework import dtypes


def run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                        window_size_ms, window_stride_ms, dct_coefficient_count,
                        model_architecture, model_size_info):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
      wanted_words: Comma-separated list of the words we're trying to recognize.
      sample_rate: How many samples per second are in the input audio files.
      clip_duration_ms: How many samples to analyze for the audio pattern.
      window_size_ms: Time slice duration to estimate frequencies from.
      window_stride_ms: How far apart time slices should be.
      dct_coefficient_count: Number of frequency bands to analyze.
      model_architecture: Name of the kind of model to generate.
      model_size_info: Model dimensions : different lengths for different models
    """

    all_training_fingerprints = np.load('model/segmentTest/65/x_train.npy')
    all_training_ground_truth = np.load('model/segmentTest/65/y_train.npy')
    all_test_fingerprints = np.load('model/segmentTest/65/x_test.npy')
    all_test_ground_truth = np.load('model/segmentTest/65/y_test.npy')
    print(all_training_fingerprints.shape())
    print(all_test_fingerprints.shape())	
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    '''
    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count)

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    '''
    model_settings = {
        'fingerprint_size': 65*9,
        'label_count': 30,
        'spectrogram_length': 65,
        'dct_coefficient_count': 9,
    }

    label_count = model_settings['label_count']
    fingerprint_size = model_settings['fingerprint_size']

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        FLAGS.model_size_info,
        FLAGS.act_max,
        is_training=False)

    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)


    # Quantize weights to 8-bits using (min,max) and write to file
    f = open('weights.h', 'wb')
    f.close()

    for v in tf.trainable_variables():
        var_name = str(v.name)
        var_values = sess.run(v)
        min_value = var_values.min()
        max_value = var_values.max()
        int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
        dec_bits = 7-int_bits
        # convert to [-128,128) or int8
        var_values = np.round(var_values*2**dec_bits)
        var_name = var_name.replace('-', '_')
        var_name = var_name.replace('/', '_')
        var_name = var_name.replace(':', '_')
        with open('weights.h','a') as f:
            f.write('#define '+var_name+' {')
        if(len(var_values.shape)>2): #convolution layer weights
            transposed_wts = np.transpose(var_values,(3,0,1,2))
        else: #fully connected layer weights or biases of any layer
            transposed_wts = np.transpose(var_values)
        with open('weights.h','a') as f:
            transposed_wts.tofile(f,sep=", ",format="%d")
            f.write('}\n')
        # convert back original range but quantized to 8-bits or 256 levels
        var_values = var_values/(2**dec_bits)
        # update the weights in tensorflow graph for quantizing the activations
        var_values = sess.run(tf.assign(v,var_values))
        print(var_name+' number of wts/bias: '+str(var_values.shape)+ \
              ' dec bits: '+str(dec_bits)+ \
              ' max: ('+str(var_values.max())+','+str(max_value)+')'+ \
              ' min: ('+str(var_values.min())+','+str(min_value)+')')

    # training set
    '''
    set_size = audio_processor.set_size('training')
    '''
    '''
    set_size = int(all_training_fingerprints.shape[0])

    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(FLAGS.batch_size, all_training_fingerprints.shape[0], FLAGS.batch_size):
        training_fingerprints = all_training_fingerprints[i - FLAGS.batch_size:i]
        training_ground_truth = all_training_ground_truth[i - FLAGS.batch_size:i]

        training_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: training_fingerprints,
                ground_truth_input: training_ground_truth,
            })
        total_accuracy += (training_accuracy * FLAGS.batch_size) / all_training_fingerprints.shape[0]
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Training accuracy = %.2f%% (N=%d)' %
                    (total_accuracy * 100, set_size))
    '''
    '''
    # validation set
    set_size = audio_processor.set_size('validation')
    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
        validation_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
            })
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                    (total_accuracy * 100, set_size))
    '''

    # test set
    '''
    set_size = audio_processor.set_size('testing')
    '''

    set_size = int(all_test_fingerprints.shape[0])

    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(FLAGS.batch_size, all_test_fingerprints.shape[0], FLAGS.batch_size):
        test_fingerprints = all_test_fingerprints[i - FLAGS.batch_size:i]
        test_ground_truth = all_test_ground_truth[i - FLAGS.batch_size:i]

        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
            })
        total_accuracy += (test_accuracy * FLAGS.batch_size) / all_test_fingerprints.shape[0]
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                       set_size))

  print(sess.run(logits, feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
            })
  print(sess.run(debugs, feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
            })
'''
    st = time.time()
    set_size = int(test_fingerprints.shape[0])
    for i in xrange(0, set_size):
        test_accuracy = sess.run(
            [evaluation_step],
            feed_dict={
                fingerprint_input: np.array([test_fingerprints[i]]),
                ground_truth_input: np.array([test_ground_truth[i]]),
            })
    et = time.time()
    print(et - st)
    print((et - st) / set_size)
'''
def main(_):

    # Create the model, load weights from checkpoint and run on train/val/test
    run_quant_inference(FLAGS.wanted_words, FLAGS.sample_rate,
                        FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
                        FLAGS.model_architecture, FLAGS.model_size_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=40.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=20.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=28,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='model/segmentTest/50/ds_cnn_9916.ckpt-100_bnfused',
        help='Checkpoint to load the weights from.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='ds_cnn',
        help='What model architecture to use')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1],
        help='Model dimensions - different for various models')
    parser.add_argument(
        '--act_max',
        type=float,
        nargs="+",
        default=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        help='activations max')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
