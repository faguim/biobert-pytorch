#!/bin/bash
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
export TRAIN_PATH=dataset/train1.0.json
export DEV_PATH=dataset/dev1.0.json
export TEST_PATH=dataset/test1.0.json

python3 dataset-code/describe_data.py -train_file TRAIN_PATH -dev_file DEV_PATH -test_file TEST_PATH
