#!/usr/bin/env sh

caffe_home="/home/simon/Workspaces/caffe/"

${caffe_home}build/tools/caffe train --solver=./training/bvlc_alexnet/solver.prototxt
