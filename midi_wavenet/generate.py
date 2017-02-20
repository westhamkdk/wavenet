from __future__ import division
from __future__ import print_function

import argparse
import json
from datetime import datetime

import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel
from midi_wavenet import midi_io
from midi_reader import melody_to_represenatation

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
    return parser.parse_args()


def write_midi(waveform, filename):
    seq = midi_io.seq_to_midi_file(waveform, filename)
    print(seq)
    print('Write midi file at {}'.format(filename))


def decode(waveform, encoding):
    if encoding == 'time_single':
        return np.array(waveform) - 2


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
        use_biases=wavenet_params['use_biases'])

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

    quantization_channels = wavenet_params['quantization_channels']
    digits = len(str(args.num_outputs))

    for i in range(args.num_outputs):
        waveform = melody_to_represenatation([60], wavenet_params['midi_encoding'])
        for step in range(args.samples):
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
        out = sess.run(samples, feed_dict={samples: waveform})
        decoded = decode(out, encoding=wavenet_params['midi_encoding'])
        filepath = '{}_{}.mid'.format(args.output_prefix, str(i + 1).zfill(digits))
        write_midi(decoded, filepath)

    print('Finished generating.')


if __name__ == '__main__':
    main()
