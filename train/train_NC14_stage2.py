"""
Stage 1
Input: 1 kb*250=250 kb
Target: 2000*2000 125 bp resolution Hi-C matrix
with denet_1 and denet_2 trained
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import time
import torch
from torch import nn

import selene_sdk
from selene_sdk.samplers.dataloader import SamplerDataLoader

sys.path.append("../")
from selene_utils3 import (
    MemmapGenome,
    Genomic2DFeatures,
    RandomPositions1DIntervalsSampler,
)
from orca_modules_1kb import Decoder, Encoder, Encoder2_1kb

res = 125
modelstr = "NC14_res125.MP.woCentro.stage2"
pre_modelstr = "NC14_res125.MP.woCentro.stage1"
sequence_length = 250000
val_size = 50
seed = 3141
n_gpus = 4
val_n_gpus = 4
mat_size = sequence_length // res
retrain = False
DATASET = "/path/to/dataset"  # replace with actual path

torch.set_default_tensor_type("torch.FloatTensor")
os.makedirs("./models/", exist_ok=True)
os.makedirs("./png/", exist_ok=True)

if __name__ == "__main__":

    smooth_diag = np.load(f"{DATASET}/NC14_{res}bp_madmaxiter0_diag0_expected.tsv.npy")
    normmat = np.exp(
        smooth_diag[np.abs(np.arange(8000)[:, None] - np.arange(8000)[None, :])]
    )

    net0 = nn.DataParallel(Encoder())
    net = nn.DataParallel(Encoder2_1kb())
    denet_1 = nn.DataParallel(Decoder())
    denet_2 = nn.DataParallel(Decoder())
    denet_4 = nn.DataParallel(Decoder())
    denet_8 = nn.DataParallel(Decoder())

    if retrain:
        net0.load_state_dict(
            torch.load("./models/model_" + modelstr + ".net0.checkpoint")
        )
        net.load_state_dict(torch.load("./models/model_" + modelstr + ".checkpoint"))
        denet_1.load_state_dict(
            torch.load("./models/model_" + modelstr + ".d1.checkpoint")
        )
        denet_2.load_state_dict(
            torch.load("./models/model_" + modelstr + ".d2.checkpoint")
        )
        denet_4.load_state_dict(
            torch.load("./models/model_" + modelstr + ".d4.checkpoint")
        )
        denet_8.load_state_dict(
            torch.load("./models/model_" + modelstr + ".d8.checkpoint")
        )
        print("loaded model states")
    else:
        net0.load_state_dict(
            torch.load("./models/pretrained/model_" + pre_modelstr + ".net0.checkpoint")
        )

        denet_1.load_state_dict(
            torch.load("./models/pretrained/model_" + pre_modelstr + ".d1.checkpoint")
        )
        denet_2.load_state_dict(
            torch.load("./models/pretrained/model_" + pre_modelstr + ".d2.checkpoint")
        )
        net_dict = net.state_dict()
        pretrained_dict = torch.load(
            "./models/pretrained/model_" + pre_modelstr + ".checkpoint"
        )
        for key in pretrained_dict.keys():
            if key in net_dict.keys():
                net_dict[key] = pretrained_dict[key]

    ## generate validation data
    t_val = Genomic2DFeatures(
        [f"{DATASET}/NC14_125bp_madmaxiter0.mcool::/resolutions/{res}"],
        [f"r{res}"],
        (mat_size, mat_size),
        cg=True,
        filt_tresh=0.3,
    )
    val_sampler = RandomPositions1DIntervalsSampler(
        reference_sequence=MemmapGenome(
            input_path=f"{DATASET}/dm6/dm6.fa",
            memmapfile=f"{DATASET}/dm6/dm6.fa.mmap",
        ),
        target=t_val,
        features=[f"r{res}"],
        intervals_path=f"{DATASET}/intervals_woCentromere.csv",
        test_holdout="interval",
        validation_holdout="interval",
        sequence_length=sequence_length,
        position_resolution=res,
        random_shift=0,
        random_strand=True,
        seed=seed,
    )

    val_sampler.mode = "validate"
    dataloader = SamplerDataLoader(val_sampler, num_workers=24, batch_size=1, seed=seed)

    validation_sequences = []
    validation_targets = []
    i = 0
    for ss, tt in dataloader:
        validation_sequences.append(ss)
        validation_targets.append(tt)
        i += 1
        if i == val_size:
            break

    validation_sequences = np.vstack(validation_sequences)
    validation_targets = np.vstack(validation_targets)
    del val_sampler

    t = Genomic2DFeatures(
        [f"{DATASET}/NC14_125bp_madmaxiter0.mcool::/resolutions/{res}"],
        [f"r{res}"],
        (mat_size, mat_size),
        cg=True,
        filt_tresh=0.5,
    )
    sampler = RandomPositions1DIntervalsSampler(
        reference_sequence=MemmapGenome(
            input_path=f"{DATASET}/dm6/dm6.fa",
            memmapfile=f"{DATASET}/dm6/dm6.fa.mmap",
        ),
        target=t,
        # target_1d=target_1d,
        features=[f"r{res}"],
        intervals_path=f"{DATASET}/intervals_woCentromere.csv",
        test_holdout="interval",
        validation_holdout="interval",
        sequence_length=sequence_length,
        position_resolution=res,
        random_shift=res // 4,
        random_strand=True,
    )

    sampler.mode = "train"
    dataloader = SamplerDataLoader(sampler, num_workers=24, batch_size=1, seed=seed)

    def figshow(x, np=False):
        if np:
            plt.imshow(x.squeeze())
        else:
            plt.imshow(x.squeeze().cpu().detach().numpy())
        plt.show()

    net0.cuda()
    net.cuda()
    denet_1.cuda()
    denet_2.cuda()
    denet_4.cuda()
    denet_8.cuda()

    net0.eval()
    net.train()
    denet_1.train()
    denet_2.train()
    denet_4.train()
    denet_8.train()

    for p in net.module.lblocks[0].parameters():
        p.requires_grad = False
    for p in net.module.blocks[0].parameters():
        p.requires_grad = False

    params = (
        [p for p in net.parameters() if p.requires_grad]
        + [p for p in denet_1.parameters() if p.requires_grad]
        + [p for p in denet_2.parameters() if p.requires_grad]
        + [p for p in denet_4.parameters() if p.requires_grad]
        + [p for p in denet_8.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.98)
    if retrain:
        optimizer_bak = torch.load("./models/model_" + modelstr + ".optimizer")
        optimizer.load_state_dict(optimizer_bak)
        print("loaded optimizer state")

    i = 0
    normmat_r1 = (
        np.reshape(normmat[:250, :250], (250, 1, 250, 1)).mean(axis=1).mean(axis=2)
    )
    normmat_r2 = (
        np.reshape(normmat[:500, :500], (250, 2, 250, 2)).mean(axis=1).mean(axis=2)
    )
    normmat_r4 = (
        np.reshape(normmat[:1000, :1000], (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
    )
    normmat_r8 = (
        np.reshape(normmat[:2000, :2000], (250, 8, 250, 8)).mean(axis=1).mean(axis=2)
    )
    normmat_r16 = (
        np.reshape(normmat[:4000, :4000], (250, 16, 250, 16)).mean(axis=1).mean(axis=2)
    )
    normmat_r32 = (
        np.reshape(normmat[:8000, :8000], (250, 32, 250, 32)).mean(axis=1).mean(axis=2)
    )
    eps1 = np.min(normmat_r1)
    eps2 = np.min(normmat_r2)
    eps4 = np.min(normmat_r4)
    eps8 = np.min(normmat_r8)
    eps16 = np.min(normmat_r16)
    eps32 = np.min(normmat_r32)

    normmats = {
        1: normmat_r1,
        2: normmat_r2,
        4: normmat_r4,
        8: normmat_r8,
        16: normmat_r16,
        32: normmat_r32,
    }
    epss = {1: eps1, 2: eps2, 4: eps4, 8: eps8, 16: eps16, 32: eps32}
    denets = {
        1: denet_1,
        2: denet_2,
        4: denet_4,
        8: denet_8,
    }

    out = open(f"log_{modelstr}.txt", "w")
    past_losses = {1: [], 2: [], 4: [], 8: [], 16: [], 32: []}
    scaler = torch.amp.GradScaler("cuda")

    stime = time.time()
    sequence = torch.zeros((n_gpus, sequence_length, 4), dtype=torch.float32)
    target = torch.zeros((n_gpus, mat_size, mat_size), dtype=torch.float32)
    while True:
        fillind = 0
        optimizer.zero_grad()
        for ss, tt in dataloader:
            if np.isnan(tt.numpy()).mean() > 0.5:
                continue
            sequence[fillind, :] = ss
            target[fillind, :] = tt
            if fillind == sequence.shape[0] - 1:
                fillind = 0
            else:
                fillind += 1
                continue

            # with torch.no_grad():
            with torch.autocast(enabled=True, device_type="cuda"):
                encoding0 = net0(torch.Tensor(sequence.float()).transpose(1, 2).cuda())
                (
                    encoding1,
                    encoding2,
                    encoding4,
                    encoding8,
                ) = net(encoding0)
                encodings = {
                    1: encoding1,
                    2: encoding2,
                    4: encoding4,
                    8: encoding8,
                }

                def train_step(level, start, coarse_pred=None):
                    target_np = target[
                        :, start : start + 250 * level, start : start + 250 * level
                    ]
                    target_r = torch.nanmean(
                        torch.nanmean(
                            torch.reshape(
                                target_np, (target.shape[0], 250, level, 250, level)
                            ),
                            axis=4,
                        ),
                        axis=2,
                    )
                    distenc = torch.log(
                        torch.FloatTensor(normmats[level][None, None, :, :]).cuda()
                    ).expand(sequence.shape[0], 1, 250, 250)

                    if coarse_pred is not None:

                        pred = denets[level].forward(
                            encodings[level][
                                :, :, int(start / level) : int(start / level) + 250
                            ],
                            distenc,
                            coarse_pred,
                        )
                    #
                    else:
                        pred = denets[level].forward(
                            encodings[level][
                                :, :, int(start / level) : int(start / level) + 250
                            ],
                            distenc,
                        )

                    target_cuda = torch.log(
                        ((target_r + epss[level]) / (normmats[level] + epss[level]))
                    )[:, 0:, 0:].cuda()

                    loss = (
                        (
                            pred[:, 0, 0:, 0:][~torch.isnan(target_cuda)]
                            - target_cuda[~torch.isnan(target_cuda)]
                        )
                        ** 2
                    ).mean()
                    loss = loss
                    past_losses[level].append(loss.detach().cpu().numpy())
                    return loss, pred

                start = 0
                loss8, pred = train_step(8, start)
                r = np.random.randint(0, 125)
                start = start + r * 8
                loss4, pred = train_step(
                    4, start, pred[:, :, r : r + 125, r : r + 125].detach()
                )
                r = np.random.randint(0, 125)
                start = start + r * 4
                loss2, pred = train_step(
                    2, start, pred[:, :, r : r + 125, r : r + 125].detach()
                )
                r = np.random.randint(0, 125)
                start = start + r * 2
                loss1, pred = train_step(
                    1, start, pred[:, :, r : r + 125, r : r + 125].detach()
                )
                loss = loss8 + loss4 + loss2 + loss1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            del encodings
            del encoding1
            del encoding2
            del encoding4
            del encoding8
            del pred

            if i % 500 == 0:
                print("l1:" + str(np.mean(past_losses[1][-500:])), flush=True)
                print("l2:" + str(np.mean(past_losses[2][-500:])), flush=True)
                print("l4:" + str(np.mean(past_losses[4][-500:])), flush=True)
                print("l8:" + str(np.mean(past_losses[8][-500:])), flush=True)
                out.write("l1:" + str(np.mean(past_losses[1][-500:])) + "\n")
                out.write("l2:" + str(np.mean(past_losses[2][-500:])) + "\n")
                out.write("l4:" + str(np.mean(past_losses[4][-500:])) + "\n")
                out.write("l8:" + str(np.mean(past_losses[8][-500:])) + "\n")
                out.flush()

            if i % 500 == 0:
                torch.save(
                    net0.state_dict(), "./models/model_" + modelstr + ".net0.checkpoint"
                )
                torch.save(
                    net.state_dict(), "./models/model_" + modelstr + ".checkpoint"
                )
                torch.save(
                    denet_1.state_dict(),
                    "./models/model_" + modelstr + ".d1.checkpoint",
                )
                torch.save(
                    denet_2.state_dict(),
                    "./models/model_" + modelstr + ".d2.checkpoint",
                )
                torch.save(
                    denet_4.state_dict(),
                    "./models/model_" + modelstr + ".d4.checkpoint",
                )
                torch.save(
                    denet_8.state_dict(),
                    "./models/model_" + modelstr + ".d8.checkpoint",
                )
                torch.save(
                    optimizer.state_dict(), "./models/model_" + modelstr + ".optimizer"
                )

            if i % 2000 == 0:
                net0.eval()
                net.eval()
                denet_1.eval()
                denet_2.eval()
                denet_4.eval()
                denet_8.eval()

                count = 0
                mses = {1: [], 2: [], 4: [], 8: [], 16: [], 32: []}
                corrs = {1: [], 2: [], 4: [], 8: [], 16: [], 32: []}
                sequence = torch.zeros(
                    (val_n_gpus, sequence_length, 4), dtype=torch.float32
                )
                target = torch.zeros(
                    (val_n_gpus, mat_size, mat_size), dtype=torch.float32
                )
                fillind = 0
                with torch.no_grad():
                    for ss, tt in zip(
                        np.array_split(validation_sequences, val_size),
                        np.array_split(validation_targets, val_size),
                    ):
                        if np.isnan(tt).mean() > 0.3:
                            continue
                        sequence[fillind, :] = torch.FloatTensor(ss)
                        target[fillind, :] = torch.FloatTensor(tt)
                        if fillind == sequence.shape[0] - 1:
                            fillind = 0
                        else:
                            fillind += 1
                            continue

                        count += 1
                        (
                            encoding1,
                            encoding2,
                            encoding4,
                            encoding8,
                        ) = net(
                            net0(torch.Tensor(sequence.float()).transpose(1, 2).cuda())
                        )

                        encodings = {
                            1: encoding1,
                            2: encoding2,
                            4: encoding4,
                            8: encoding8,
                        }

                        def eval_step(level, start, coarse_pred=None, plot=False):
                            target_r = np.nanmean(
                                np.nanmean(
                                    np.reshape(
                                        target[
                                            :,
                                            start : start + 250 * level,
                                            start : start + 250 * level,
                                        ].numpy(),
                                        (target.shape[0], 250, level, 250, level),
                                    ),
                                    axis=4,
                                ),
                                axis=2,
                            )
                            distenc = torch.log(
                                torch.FloatTensor(
                                    normmats[level][None, None, :, :]
                                ).cuda()
                            ).expand(sequence.shape[0], 1, 250, 250)
                            if coarse_pred is not None:
                                if level == 1:
                                    pred = denets[level].forward(
                                        encodings[level][
                                            :,
                                            :,
                                            int(start / level) : int(start / level)
                                            + 250,
                                        ],
                                        distenc,
                                        coarse_pred,
                                    )
                                else:
                                    pred = denets[level].forward(
                                        encodings[level][
                                            :,
                                            :,
                                            int(start / level) : int(start / level)
                                            + 250,
                                        ],
                                        distenc,
                                        coarse_pred,
                                    )
                            else:
                                pred = denets[level].forward(
                                    encodings[level][
                                        :,
                                        :,
                                        int(start / level) : int(start / level) + 250,
                                    ],
                                    distenc,
                                )
                            target_cuda = torch.Tensor(
                                np.log(
                                    (
                                        (target_r + epss[level])
                                        / (normmats[level] + epss[level])
                                    )
                                )[:, 0:, 0:]
                            ).cuda()
                            loss = (
                                (
                                    pred[:, 0, 0:, 0:][~torch.isnan(target_cuda)]
                                    - target_cuda[~torch.isnan(target_cuda)]
                                )
                                ** 2
                            ).mean()

                            mses[level].append(loss.detach().cpu().numpy())
                            pred_np = (
                                pred[:, 0, 0:, 0:]
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape((pred.shape[0], -1))
                            )
                            target_np = np.log(
                                (target_r + epss[level])
                                / (normmats[level] + epss[level])
                            )[:, 0:, 0:].reshape((pred.shape[0], -1))

                            for j in range(pred_np.shape[0]):
                                validinds = ~np.isnan(target_np[j, :])
                                if np.mean(validinds) > 0.3:
                                    corrs[level].append(
                                        pearsonr(
                                            pred_np[j, validinds],
                                            target_np[j, validinds],
                                        )[0]
                                    )
                                else:
                                    corrs[level].append(np.nan)
                            if plot:
                                for ii in range(pred.shape[0]):
                                    figshow(pred[ii, 0, :, :])
                                    plt.savefig(
                                        "./png/model_"
                                        + modelstr
                                        + ".test"
                                        + str(ii)
                                        + "."
                                        + str(count)
                                        + ".level"
                                        + str(level)
                                        + ".pred.png"
                                    )
                                    plt.clf()
                                    figshow(
                                        np.log(
                                            (
                                                (target_r + epss[level])
                                                / (normmats[level] + epss[level])
                                            )
                                        )[ii, :, :],
                                        np=True,
                                    )
                                    plt.savefig(
                                        "./png/model_"
                                        + modelstr
                                        + ".test"
                                        + str(ii)
                                        + "."
                                        + str(count)
                                        + ".level"
                                        + str(level)
                                        + ".label.png"
                                    )
                            return pred

                        start = 0
                        pred = eval_step(8, start, plot=(count == 2))
                        start = start + 62 * 8
                        pred = eval_step(
                            4, start, pred[:, :, 62:187, 62:187], plot=(count == 2)
                        )
                        start = start + 62 * 4
                        pred = eval_step(
                            2, start, pred[:, :, 62:187, 62:187], plot=(count == 2)
                        )
                        start = start + 62 * 2
                        pred = eval_step(
                            1, start, pred[:, :, 62:187, 62:187], plot=(count == 2)
                        )

                        del encodings
                        del encoding1
                        del encoding2
                        del encoding4
                        del encoding8
                        del pred

                print(
                    "Cor1 {0}, Cor2 {1}, Cor4 {2}, Cor8 {3}".format(
                        np.nanmean(corrs[1]),
                        np.nanmean(corrs[2]),
                        np.nanmean(corrs[4]),
                        np.nanmean(corrs[8]),
                    )
                )
                print(
                    "MSE1 {0}, MSE2 {1}, MSE4 {2}, MSE8 {3}".format(
                        np.nanmean(mses[1]),
                        np.nanmean(mses[2]),
                        np.nanmean(mses[4]),
                        np.nanmean(mses[8]),
                    )
                )
                out.write(
                    "Cor1 {0}, Cor2 {1}, Cor4 {2}, Cor8 {3}".format(
                        np.nanmean(corrs[1]),
                        np.nanmean(corrs[2]),
                        np.nanmean(corrs[4]),
                        np.nanmean(corrs[8]),
                    )
                    + "\n"
                )
                out.write(
                    "MSE1 {0}, MSE2 {1}, MSE4 {2}, MSE8 {3}".format(
                        np.nanmean(mses[1]),
                        np.nanmean(mses[2]),
                        np.nanmean(mses[4]),
                        np.nanmean(mses[8]),
                    )
                    + "\n"
                )
                out.flush()
                net0.eval()
                net.train()
                denet_1.train()
                denet_2.train()
                denet_4.train()
                denet_8.train()

            sequence = torch.zeros((n_gpus, sequence_length, 4), dtype=torch.float32)
            target = torch.zeros((n_gpus, mat_size, mat_size), dtype=torch.float32)
            i += 1
            print(f"itr {i} elasped time: {time.time() - stime}")
            stime = time.time()
