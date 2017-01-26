from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, midi_io

SAMPLES = 4*4*8
NUM_OUTPUTS = 1
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
WINDOW = 4*4*8
LOGDIR = './logdir'
WAVENET_PARAMS = './midi-wavenet_params.json'


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How long a sequence of waveform samples to generate')
    parser.add_argument(
        '--num_outputs',
        type=int,
        default=NUM_OUTPUTS,
        help='How many output sequences to generate')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--output_prefix',
        type=str,
        default=STARTED_DATESTRING,
        help='Prefix of file path where the output sequences to be saved')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=False,
        help='Use fast generation. Do not use when num_outputs is larger than 1')
    return parser.parse_args()


def write_midi(waveform, filename, encoding):
    seq = midi_io.seq_to_midi_file(waveform, filename)
    print(seq)
    print('Write midi file at {}'.format(filename))


def main():
    args = get_arguments()
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        encoding=wavenet_params["encoding"])

    samples = tf.placeholder(tf.int32)

    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples)
    else:
        next_sample = net.predict_proba(samples)

    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = samples
    quantization_channels = wavenet_params['quantization_channels']
    digits = len(str(args.num_outputs))

    for i in range(args.num_outputs):
        waveform = [62.]
        for step in range(args.samples):
            if args.fast_generation:
                outputs = [next_sample]
                outputs.extend(net.push_ops)
                window = waveform[-1]
            else:
                if len(waveform) > args.window:
                    window = waveform[-args.window:]
                else:
                    window = waveform
                outputs = [next_sample]

            # Run the WaveNet to predict the next sample.
            prediction = sess.run(outputs, feed_dict={samples: window})[0]
            sample = np.random.choice(
                np.arange(quantization_channels), p=prediction)
            waveform.append(sample)

        # Introduce a newline to clear the carriage return from the progress.
        print()

        # Save the result as a midi file.
        out = sess.run(decode, feed_dict={samples: waveform})
        filepath = '{}_{}.mid'.format(args.output_prefix, str(i + 1).zfill(digits))
        write_midi(out, filepath)

    print('Finished generating.')


if __name__ == '__main__':
    main()
