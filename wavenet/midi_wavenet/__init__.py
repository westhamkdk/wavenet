import os

from wavenet import PROJECT_ROOT

LOGDIR_ROOT = os.path.join(PROJECT_ROOT, 'logdir')
WAVENET_PARAMS = os.path.join(os.path.dirname(__file__), 'midi-wavenet_params.json')
