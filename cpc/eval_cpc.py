# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import os
import numpy as np
import torch
import time
from copy import deepcopy
import random
import psutil
import sys

import cpc.criterion as cr
import cpc.model as model
import cpc.utils.misc as utils
import cpc.feature_loader as fl
from cpc.cpc_default_config import set_default_cpc_config
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels


def getCriterion(args, downsampling, nSpeakers, nPhones):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if not args.supervised:
        if args.cpc_mode == 'none':
            cpcCriterion = cr.NoneCriterion()
        else:
            sizeInputSeq = (args.sizeWindow // downsampling)
            cpcCriterion = cr.CPCUnsupersivedCriterion(args.nPredicts,
                                                       args.hiddenGar,
                                                       args.hiddenEncoder,
                                                       args.negativeSamplingExt,
                                                       mode=args.cpc_mode,
                                                       rnnMode=args.rnnMode,
                                                       dropout=args.dropout,
                                                       nSpeakers=nSpeakers,
                                                       speakerEmbedding=args.speakerEmbedding,
                                                       sizeInputSeq=sizeInputSeq)
    elif args.pathPhone is not None:
        if not args.CTC:
            cpcCriterion = cr.PhoneCriterion(dimFeatures,
                                             nPhones, args.onEncoder,
                                             nLayers=args.nLevelsPhone)
        else:
            cpcCriterion = cr.CTCPhoneCriterion(dimFeatures,
                                                nPhones, args.onEncoder)
    else:
        cpcCriterion = cr.SpeakerCriterion(dimFeatures, nSpeakers)
    return cpcCriterion


def loadCriterion(pathCheckpoint, downsampling, nSpeakers, nPhones):
    _, _, locArgs = fl.getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion

def valStep(dataLoader_list,
            cpcModel,
            cpcCriterion):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    iter = 0

    for i, dataLoader in enumerate(dataLoader_list):
        step = 0
        for fulldata in dataLoader:

            batchData, label = fulldata

            batchData = batchData.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            with torch.no_grad():
                c_feature, encoded_data, label = cpcModel(batchData, label)
                allLosses, allAcc = cpcCriterion(c_feature, encoded_data, label)

            if f"locLoss_val_{i}" not in logs:
                logs[f"locLoss_val_{i}"] = np.zeros(allLosses.size(1))
                logs[f"locAcc_val_{i}"] = np.zeros(allLosses.size(1))

            iter += 1
            logs[f"locLoss_val_{i}"] += allLosses.mean(dim=0).cpu().numpy()
            logs[f"locAcc_val_{i}"] += allAcc.mean(dim=0).cpu().numpy()
            step += 1
        logs[f"locLoss_val_{i}"] = logs[f"locLoss_val_{i}"] / step
        logs[f"locAcc_val_{i}"] = logs[f"locAcc_val_{i}"] / step

    logs["locLoss_val"] = np.mean([logs[f"locLoss_val_{i}"] for i in range(len(dataLoader_list))], axis=0)
    logs["locAcc_val"] = np.mean([logs[f"locAcc_val_{i}"] for i in range(len(dataLoader_list))], axis=0)
    logs["iter"] = iter
    utils.show_logs("Validation loss:", logs)
    return logs


def main(args):
    args = parseArgs(args)
    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    utils.set_seed(args.random_seed)

    # Load labels (if neeeded)
    phoneLabels, nPhones = None, None
    if args.supervised and args.pathPhone is not None:
        print("Loading the phone labels at " + args.pathPhone)
        phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
        print(f"{nPhones} phones found")

    # Load datasets
    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache,
                                     speaker_level=args.speaker_level)

    print(f'Found files: {len(seqNames)} seqs, {len(speakers)} speakers')
    if args.pathVal is None:
        seqVal_list = [seqNames]
    else:
        seqVal_list = []
        for pathseqVal in args.pathVal:
            seqVal = filterSeqs(pathseqVal, seqNames)
            seqVal_list.append(seqVal)
    if args.debug:
        for i in range(len(seqVal_list)):
            seqVal_list[i] = seqVal_list[i][-100:]
    print(f'Validation dataset: {len(seqVal_list)} subsets | Number of sequences: {[len(item) for item in seqVal_list]}')
    print("")
    print(f'Loading audio data at {args.pathDB}')
    print("Loading the validation datasets")
    valDataset_list = []
    for seqVal in seqVal_list:
        valDataset = AudioBatchData(args.pathDB,
                                args.sizeWindow,
                                seqVal,
                                phoneLabels,
                                len(speakers),
                                nProcessLoader=args.n_process_loader)
        valDataset_list.append(valDataset)
    print("Validation datasets loaded")
    print("")

    # Get checkpoints infomation
    if not os.path.isdir(args.pathCheckpoint):
        print(f"Checkpoint directory {args.pathCheckpoint} not found!")
        return
    checkpoints = [x for x in os.listdir(args.pathCheckpoint)
                   if os.path.splitext(x)[1] == '.pt'
                   and os.path.splitext(x[11:])[0].isdigit()]
    if len(checkpoints) == 0:
        print("No checkpoints found at " + args.pathCheckpoint)
        return
    else:
        print(f"Found {len(checkpoints)} checkpoints found at {args.pathCheckpoint}")
    checkpoints.sort(key=lambda x: int(os.path.splitext(x[11:])[0]))

    # Create eval dir
    if not os.path.isdir(os.path.join(args.pathCheckpoint, "eval")):
        os.mkdir(os.path.join(args.pathCheckpoint, "eval"))

    # Write args
    with open(os.path.join(args.pathCheckpoint, "eval", "eval_args.json"), 'w') as file:
        json.dump(vars(args), file, indent=2)

    # Prepare dataLoader
    batchSize = args.nGPU * args.batchSizeGPU
    valLoader_list = []
    for valDataset in valDataset_list:
        valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                            numWorkers=0)
        valLoader_list.append(valLoader)
    print(f"Validation datasets {[len(item) for item in valLoader_list]} batches, batch size {batchSize}")

    # Do evaluation for each checkpoint
    logs = {"epoch": []}
    epochs = []
    valAccuracyList = []
    for checkpoint in checkpoints:
        start_time = time.time()

        print("")
        print(f"Run evaluation for {checkpoint}")
        ckptPath = os.path.join(args.pathCheckpoint, checkpoint)

        # Get epoch number
        num_epoch = int(os.path.splitext(checkpoint[11:])[0])
        logs["epoch"].append(num_epoch)
        epochs.append(num_epoch)

        # Load checkpoint
        cpcModel, _, _ = fl.loadModel([ckptPath])
        cpcModel.supervised = args.supervised

        # Load criterion
        cpcCriterion = loadCriterion(ckptPath, cpcModel.gEncoder.DOWNSAMPLING,
                                     len(speakers), nPhones)

        # Move to cuda
        cpcModel.cuda()
        cpcCriterion.cuda()
        
        cpcModel = torch.nn.DataParallel(cpcModel,
                                     device_ids=range(args.nGPU)).cuda()
        cpcCriterion = torch.nn.DataParallel(cpcCriterion,
                                         device_ids=range(args.nGPU)).cuda()

        locLogsVal = valStep(valLoader_list, cpcModel, cpcCriterion)

        print(f'Ran evaluation for {checkpoint} in {time.time() - start_time:.2f} seconds')

        torch.cuda.empty_cache()

        currentAccuracy = float(locLogsVal["locAcc_val"].mean())
        valAccuracyList.append(100*currentAccuracy)
        
        for key, value in locLogsVal.items():
            if key not in logs:
                logs[key] = []
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        utils.save_logs(logs, os.path.join(args.pathCheckpoint, "eval",  "eval_logs.json"))

        with open(os.path.join(args.pathCheckpoint, "eval",  "eval_valAcc_info.txt"), 'w') as file:
            outLines = [f"Epoch {ep} : {acc}" for ep, acc in zip(epochs, valAccuracyList)] + \
                       [f"Best valAcc : checkpoint_{epochs[np.argmax(valAccuracyList)]}"]
            file.write("\n".join(outLines))


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    # Default arguments:
    parser = set_default_cpc_config(parser)

    group_db = parser.add_argument_group('Dataset')
    group_db.add_argument('--pathDB', type=str, default=None,
                          help='Path to the directory containing the '
                          'data.')
    group_db.add_argument('--file_extension', type=str, default=".flac",
                          help="Extension of the audio files in the dataset.")
    group_db.add_argument('--pathVal', type=str, default=None,
                          help='List of .txt files containing the list of the '
                          'validation sequences (delimited by ,).')
    group_db.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    group_db.add_argument('--ignore_cache', action='store_true',
                          help='Activate if the dataset has been modified '
                          'since the last training session.')
    group_db.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    group_db.add_argument('--speaker_level', type=int, default=1,
                          help="Level of speaker in the training directory.")
    group_supervised = parser.add_argument_group(
        'Supervised mode (depreciated)')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='(Depreciated) Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the speaker classification.')
    group_supervised.add_argument('--pathPhone', type=str, default=None,
                                  help='(Supervised mode only) Path to a .txt '
                                  'containing the phone labels of the dataset. If given '
                                  'and --supervised, will train the model using a '
                                  'phone classification task.')
    group_supervised.add_argument('--CTC', action='store_true')

    group_save = parser.add_argument_group('Save')
    group_save.add_argument('--pathCheckpoint', type=str, default=None,
                            help="Path of the output directory.")

    group_gpu = parser.add_argument_group('GPUs')
    group_gpu.add_argument('--nGPU', type=int, default=-1,
                           help="Number of GPU to use (default: use all "
                           "available GPUs)")
    group_gpu.add_argument('--batchSizeGPU', type=int, default=8,
                           help='Number of batches per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    args = parser.parse_args(argv)

    assert args.pathDB is not None
    assert args.pathCheckpoint is not None
    args.pathCheckpoint = os.path.abspath(args.pathCheckpoint)
    if not os.path.isdir(args.pathCheckpoint):
        args.pathCheckpoint = os.path.dirname(args.pathCheckpoint)

    if args.pathVal is not None:
        args.pathVal = args.pathVal.split(",")

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(),\
        f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"
    print(f"Let's use {args.nGPU} GPUs!")

    # set it up if needed, so that it is dumped along with other args
    if args.random_seed is None:
        args.random_seed = random.randint(0, 2**31)

    return args


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
