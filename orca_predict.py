"""
This module provides functions for using Orca models for various
types of the predictions. This is the main module that you need for
interacting with Orca models.

To use any of the prediction functions, `load_resources` has to be
called first to load the necessary resources.

The coordinates used in Orca are 0-based, inclusive for the start
coordinate and exclusive for the end coordinate, consistent with
python conventions.
"""

import os
import pathlib

import numpy as np
import torch

from selene_sdk.sequences import Genome

from orca_utils import (
    genomeplot,
    StructuralChange2,
    coord_clip,
    coord_round,
)

try:
    from seqstr import seqstr
except ImportError:
    seqstr = None

ORCA_PATH = str(pathlib.Path(__file__).parent.absolute())
target_dict_global = {}
# fly_resources_global = {}
fly_model_dict_global = {}

FLY_SEQUENCE_LENGTH = 250000
FLY_WINDOW_RADIUS = FLY_SEQUENCE_LENGTH // 2
FLY_BASE_RESOLUTION = 125
FLY_LEVELS = (1, 2, 4, 8)


def load_resources(models=["NC12"], use_cuda=True, use_memmapgenome=True):
    """
    Load Orca-Fly resources based on predict.ipynb.

    Parameters
    ----------
    models : list(str or dict)
        Each entry can be a string model name (e.g., "NC12", "NC14") or a dict
        with keys: {"name", "work_dir", "modelstr", "expected_diag_path"}.
    use_cuda : bool, optional
        Default is True. If true, loaded models are moved to GPU.
    use_memmapgenome : bool, optional
        Unused for Orca-Fly. Present for API compatibility.
    """
    global dm6, target_available

    fly_model_dict_global.clear()
    target_dict_global.clear()

    def _resolve_fly_model_spec(model):
        if isinstance(model, dict):
            required = ["name", "work_dir", "modelstr", "expected_diag_path"]
            missing = [k for k in required if k not in model]
            if missing:
                raise ValueError(f"Model spec missing keys: {missing}")
            return {
                "name": model["name"],
                "work_dir": model["work_dir"],
                "modelstr": model["modelstr"],
                "expected_diag_path": model["expected_diag_path"],
                "genome_fasta": model.get("genome_fasta", "dm6/dm6.fa"),
                "genome_memmap": model.get("genome_memmap", "dm6/dm6.fa.mmap"),
            }

        if not isinstance(model, str):
            raise ValueError("Model spec must be a string or dict.")

        name = model
        work_dir = os.environ.get("ORCA_FLY_WORK_DIR")
        modelstr = os.environ.get(f"ORCA_FLY_MODELSTR_{name}") or os.environ.get(
            "ORCA_FLY_MODELSTR"
        )
        expected_diag_path = os.environ.get(
            f"ORCA_FLY_EXPECTED_DIAG_{name}"
        ) or os.environ.get("ORCA_FLY_EXPECTED_DIAG")
        genome_fasta = os.environ.get("ORCA_FLY_GENOME_FASTA", "dm6/dm6.fa")
        genome_memmap = os.environ.get("ORCA_FLY_GENOME_MEMMAP", "dm6/dm6.fa.mmap")

        if not work_dir or not modelstr or not expected_diag_path:
            raise ValueError(
                "Fly resources require ORCA_FLY_WORK_DIR, ORCA_FLY_MODELSTR[_NAME], "
                "and ORCA_FLY_EXPECTED_DIAG[_NAME] environment variables, or pass "
                "a dict model spec with these paths."
            )

        return {
            "name": name,
            "work_dir": work_dir,
            "modelstr": modelstr,
            "expected_diag_path": expected_diag_path,
            "genome_fasta": genome_fasta,
            "genome_memmap": genome_memmap,
        }

    if not models:
        models = ["NC12"]

    for model in models:
        spec = _resolve_fly_model_spec(model)
        resources = load_fly_resources(
            spec["work_dir"],
            spec["modelstr"],
            spec["expected_diag_path"],
            model_name=spec["name"],
            genome_fasta=spec["genome_fasta"],
            genome_memmap=spec["genome_memmap"],
            use_cuda=use_cuda,
        )
        fly_model_dict_global[spec["name"]] = resources

    dm6 = next(iter(fly_model_dict_global.values()))["genome"]
    target_available = bool(target_dict_global)
    return fly_model_dict_global, target_dict_global


def load_fly_resources(
    work_dir,
    modelstr,
    expected_diag_path,
    model_name=None,
    genome_fasta="dm6/dm6.fa",
    genome_memmap="dm6/dm6.fa.mmap",
    use_cuda=True,
    use_dataparallel=True,
):
    """
    Load Orca-Fly 1kb/125bp prediction resources based on predict.ipynb.

    Parameters
    ----------
    work_dir : str
        Base directory containing the model checkpoints and genome files.
    modelstr : str
        Model string used in checkpoint filenames.
    expected_diag_path : str
        Path to the expected diagonal npy file.
    model_name : str, optional
        Model name (e.g., "NC12", "NC14") for loading target datasets.
    genome_fasta : str, optional
        Relative path to FASTA file (from work_dir).
    genome_memmap : str, optional
        Relative path to memmap file (from work_dir).
    use_cuda : bool, optional
        Default is True. If False, use CPU.
    use_dataparallel : bool, optional
        Default is True. If True, wrap modules in DataParallel.
    """
    from orca_modules_1kb import Encoder2_1kb, Decoder, Encoder
    from selene_utils3 import MemmapGenome as FlyMemmapGenome, Genomic2DFeatures

    fly_resources_global = {}
    net0 = Encoder()
    net = Encoder2_1kb()
    denet_1 = Decoder()
    denet_2 = Decoder()
    denet_4 = Decoder()
    denet_8 = Decoder()

    if use_dataparallel:
        net0 = torch.nn.DataParallel(net0)
        net = torch.nn.DataParallel(net)
        denet_1 = torch.nn.DataParallel(denet_1)
        denet_2 = torch.nn.DataParallel(denet_2)
        denet_4 = torch.nn.DataParallel(denet_4)
        denet_8 = torch.nn.DataParallel(denet_8)

    map_location = None if use_cuda else "cpu"
    net0.load_state_dict(
        torch.load(
            os.path.join(work_dir, f"resources/model_{modelstr}.net0.checkpoint"),
            map_location=map_location,
        )
    )
    net.load_state_dict(
        torch.load(
            os.path.join(work_dir, f"resources/model_{modelstr}.checkpoint"),
            map_location=map_location,
        )
    )
    denet_1.load_state_dict(
        torch.load(
            os.path.join(work_dir, f"resources/model_{modelstr}.d1.checkpoint"),
            map_location=map_location,
        )
    )
    denet_2.load_state_dict(
        torch.load(
            os.path.join(work_dir, f"resources/model_{modelstr}.d2.checkpoint"),
            map_location=map_location,
        )
    )
    denet_4.load_state_dict(
        torch.load(
            os.path.join(work_dir, f"resources/model_{modelstr}.d4.checkpoint"),
            map_location=map_location,
        )
    )
    denet_8.load_state_dict(
        torch.load(
            os.path.join(work_dir, f"resources/model_{modelstr}.d8.checkpoint"),
            map_location=map_location,
        )
    )

    if use_cuda:
        net0.cuda()
        net.cuda()
        denet_1.cuda()
        denet_2.cuda()
        denet_4.cuda()
        denet_8.cuda()
    else:
        net0.cpu()
        net.cpu()
        denet_1.cpu()
        denet_2.cpu()
        denet_4.cpu()
        denet_8.cpu()

    net0.eval()
    net.eval()
    denet_1.eval()
    denet_2.eval()
    denet_4.eval()
    denet_8.eval()

    smooth_diag = np.load(expected_diag_path)
    normmat = np.exp(
        smooth_diag[np.abs(np.arange(8000)[:, None] - np.arange(8000)[None, :])]
    )

    def _downsample_normmat(level):
        size = 250 * level
        return (
            np.reshape(normmat[:size, :size], (250, level, 250, level))
            .mean(axis=1)
            .mean(axis=2)
        )

    normmats = {level: _downsample_normmat(level) for level in FLY_LEVELS}
    epss = {level: np.min(normmats[level]) for level in FLY_LEVELS}

    genome = FlyMemmapGenome(
        input_path=os.path.join(work_dir, genome_fasta),
        memmapfile=os.path.join(work_dir, genome_memmap),
    )

    target = None
    if model_name in {"NC12", "NC14"}:
        target_path = os.path.join(
            work_dir, "resources", f"{model_name}_125bp_madmaxiter0.mcool"
        )
        if os.path.exists(target_path):
            target = Genomic2DFeatures(
                [f"{target_path}::/resolutions/125"],
                ["r125"],
                (2000, 2000),
                cg=True,
            )
            target_dict_global[model_name] = target

    fly_resources_global.update(
        {
            "net0": net0,
            "net": net,
            "denets": {1: denet_1, 2: denet_2, 4: denet_4, 8: denet_8},
            "normmats": normmats,
            "epss": epss,
            "genome": genome,
            "modelstr": modelstr,
            "work_dir": work_dir,
            "target": target,
        }
    )
    return fly_resources_global


def pred_fly_1kb(sequence, resources, use_cuda=True):
    """
    Predict 1kb contact map (250x250) for a 250kb sequence.

    Parameters
    ----------
    sequence : numpy.ndarray or torch.Tensor
        One-hot sequence encoding of shape (1, 250000, 4) or (1, 4, 250000).
    resources : dict
        Output of load_fly_resources.
    use_cuda : bool, optional
        Default is True. If False, use CPU.
    """
    res = resources
    if not res:
        raise ValueError("Fly resources are not loaded. Call load_fly_resources first.")

    seq = torch.as_tensor(sequence, dtype=torch.float32)
    if seq.ndim != 3:
        raise ValueError(
            "Expected 3D sequence tensor with shape (1, L, 4) or (1, 4, L)."
        )
    if seq.shape[1] != 4:
        seq = seq.transpose(1, 2)
    if use_cuda:
        seq = seq.cuda()

    with torch.no_grad():
        encoding0 = res["net0"](seq)
        encoding1, encoding2, encoding4, encoding8 = res["net"](encoding0)
        encodings = {1: encoding1, 2: encoding2, 4: encoding4, 8: encoding8}

        level = 8
        start = 0
        distenc = torch.log(
            torch.FloatTensor(res["normmats"][level][None, None, :, :]).cuda()
            if use_cuda
            else torch.FloatTensor(res["normmats"][level][None, None, :, :])
        ).expand(seq.shape[0], 1, 250, 250)

        pred = res["denets"][level].forward(
            encodings[level][:, :, int(start / level) : int(start / level) + 250],
            distenc,
        )

    return pred


def pred_fly_125(sequence, resources, depth=3, use_cuda=True):
    """
    Predict 125bp contact map (250x250) by zooming into the center.

    Parameters
    ----------
    sequence : numpy.ndarray or torch.Tensor
        One-hot sequence encoding of shape (1, 250000, 4) or (1, 4, 250000).
    resources : dict
        Output of load_fly_resources.
    depth : int, optional
        Number of refinement steps. Default is 3 (levels 4,2,1).
    use_cuda : bool, optional
        Default is True. If False, use CPU.
    """
    res = resources
    if not res:
        raise ValueError("Fly resources are not loaded. Call load_fly_resources first.")

    seq = torch.as_tensor(sequence, dtype=torch.float32)
    if seq.ndim != 3:
        raise ValueError(
            "Expected 3D sequence tensor with shape (1, L, 4) or (1, 4, L)."
        )
    if seq.shape[1] != 4:
        seq = seq.transpose(1, 2)
    if use_cuda:
        seq = seq.cuda()

    with torch.no_grad():
        encoding0 = res["net0"](seq)
        encoding1, encoding2, encoding4, encoding8 = res["net"](encoding0)
        encodings = {1: encoding1, 2: encoding2, 4: encoding4, 8: encoding8}

        level = 8
        start = 0
        distenc = torch.log(
            torch.FloatTensor(res["normmats"][level][None, None, :, :]).cuda()
            if use_cuda
            else torch.FloatTensor(res["normmats"][level][None, None, :, :])
        ).expand(seq.shape[0], 1, 250, 250)

        pred = res["denets"][level].forward(
            encodings[level][:, :, int(start / level) : int(start / level) + 250],
            distenc,
        )

        for level in [4, 2, 1][:depth]:
            r = 63
            start = start + r * level * 2
            distenc = torch.log(
                torch.FloatTensor(res["normmats"][level][None, None, :, :]).cuda()
                if use_cuda
                else torch.FloatTensor(res["normmats"][level][None, None, :, :])
            ).expand(seq.shape[0], 1, 250, 250)
            pred = res["denets"][level].forward(
                encodings[level][:, :, int(start / level) : int(start / level) + 250],
                distenc,
                pred[:, :, r : r + 125, r : r + 125].detach(),
            )

    return pred


def genomepredict(
    sequence,
    mchr,
    mpos=-1,
    wpos=-1,
    models=None,
    targets=None,
    annotation=None,
    use_cuda=True,
    nan_thresh=1,
):
    """
    Fly prediction for a 250kb sequence (1kb/125bp model).

    Parameters
    ----------
    sequence : numpy.ndarray or torch.Tensor
        One-hot sequence encoding of shape 1 x 250000 x 4.
    mchr : str
        Chromosome name (used for output only).
    mpos : int, optional
        Zoom-in coordinate. Defaults to the sequence center.
    wpos : int, optional
        Center coordinate of the sequence. Defaults to the sequence center.
    models : list(dict or str), optional
        Models to use. Strings are resolved from loaded fly models.
    targets : list(numpy.ndarray or torch.Tensor), optional
        Observed balanced contact matrices at 125bp resolution. If provided,
        length must match the number of models.
    annotation : unused
    use_cuda : bool, optional
        Default is True. If False, use CPU.
    nan_thresh : int, optional
        Threshold for NaN proportion when downsampling targets.
    """
    if wpos == -1:
        wpos = FLY_WINDOW_RADIUS
    if mpos == -1:
        mpos = wpos

    model_objs = []
    if models is None:
        model_objs = list(fly_model_dict_global.values())
    else:
        for m in models:
            if isinstance(m, dict):
                model_objs.append(m)
            elif isinstance(m, str):
                if m in fly_model_dict_global:
                    model_objs.append(fly_model_dict_global[m])
                else:
                    raise ValueError(f"Unknown fly model: {m}")
            else:
                raise ValueError("Models must be dicts or strings.")

    if not model_objs:
        raise ValueError("No fly models loaded. Call load_resources first.")

    if targets is not None and len(targets) != len(model_objs):
        raise ValueError("Length of targets must match the number of models.")

    seq = torch.as_tensor(sequence, dtype=torch.float32)
    if seq.ndim != 3:
        raise ValueError(
            "Expected 3D sequence tensor with shape (1, L, 4) or (1, 4, L)."
        )
    if seq.shape[1] != 4:
        seq = seq.transpose(1, 2)
    if seq.shape[2] != FLY_SEQUENCE_LENGTH:
        raise ValueError(f"Expected sequence length {FLY_SEQUENCE_LENGTH}.")
    if use_cuda:
        seq = seq.cuda()

    levels = [8, 4, 2, 1]
    start_bins = []
    start = 0
    for level in levels:
        if level != 8:
            start += 63 * level * 2
        start_bins.append(start)

    preds_by_model = []
    targets_by_model = []
    for model_index, model in enumerate(model_objs):
        with torch.no_grad():
            encoding0 = model["net0"](seq)
            encoding1, encoding2, encoding4, encoding8 = model["net"](encoding0)
            encodings = {1: encoding1, 2: encoding2, 4: encoding4, 8: encoding8}

            preds = []
            coarse = None
            for i, level in enumerate(levels):
                start = start_bins[i]
                distenc = torch.log(
                    torch.FloatTensor(model["normmats"][level][None, None, :, :]).cuda()
                    if use_cuda
                    else torch.FloatTensor(model["normmats"][level][None, None, :, :])
                ).expand(seq.shape[0], 1, 250, 250)

                if coarse is not None:
                    pred = model["denets"][level].forward(
                        encodings[level][
                            :, :, int(start / level) : int(start / level) + 250
                        ],
                        distenc,
                        coarse,
                    )
                else:
                    pred = model["denets"][level].forward(
                        encodings[level][
                            :, :, int(start / level) : int(start / level) + 250
                        ],
                        distenc,
                    )

                preds.append(pred)
                coarse = pred[:, :, 63 : 63 + 125, 63 : 63 + 125].detach()

        preds_by_model.append(
            [
                (
                    p.cpu().detach().numpy()[0, 0, :, :]
                    if p.shape[1] == 1
                    else p.cpu().detach().numpy()[0]
                )
                for p in preds
            ]
        )

        if targets is not None:
            target = targets[model_index]
            if torch.is_tensor(target):
                target_np = target.detach().cpu().numpy()
            else:
                target_np = target
            if target_np.ndim == 3:
                target_np = target_np[0]
            ts = []
            for i, level in enumerate(levels):
                start = start_bins[i]
                size = 250 * level
                target_window = target_np[
                    start : start + size,
                    start : start + size,
                ]
                target_r = np.nanmean(
                    np.nanmean(
                        np.reshape(target_window, (250, level, 250, level)),
                        axis=3,
                    ),
                    axis=1,
                )
                target_nan = np.mean(
                    np.mean(
                        np.isnan(np.reshape(target_window, (250, level, 250, level))),
                        axis=3,
                    ),
                    axis=1,
                )
                target_r[target_nan > nan_thresh] = np.nan
                eps = model["epss"][level]
                target_log = np.log((target_r + eps) / (model["normmats"][level] + eps))
                ts.append(target_log)
            targets_by_model.append(ts)

    start_coords = [
        wpos - FLY_WINDOW_RADIUS + s * FLY_BASE_RESOLUTION for s in start_bins
    ]
    end_coords = [
        start_coords[i] + 250 * levels[i] * FLY_BASE_RESOLUTION
        for i in range(len(levels))
    ]

    output = {
        "predictions": preds_by_model,
        "experiments": targets_by_model if targets is not None else None,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "chr": mchr,
        "annos": None,
        "normmats": [
            [model["normmats"][lvl] for lvl in levels] for model in model_objs
        ],
    }
    return output


def _retrieve_multi(*args, **kwargs):
    """Not supported for Orca-Fly models."""
    raise ValueError("_retrieve_multi is not supported for Orca-Fly models.")


def _validate_fly_window(window_radius):
    if window_radius is None:
        window_radius = FLY_WINDOW_RADIUS
    if window_radius != FLY_WINDOW_RADIUS:
        raise ValueError(
            f"Only window_radius {FLY_WINDOW_RADIUS} is supported for Orca-Fly models."
        )
    return window_radius


def _get_fly_genome(genome):
    if genome is not None:
        return genome
    if fly_model_dict_global:
        return next(iter(fly_model_dict_global.values()))["genome"]
    raise ValueError("Fly genome not available. Call load_resources or pass genome.")


def _resolve_fly_models(custom_models, model_labels=None):
    if custom_models is None:
        if fly_model_dict_global:
            models = list(fly_model_dict_global.values())
            labels = list(fly_model_dict_global.keys())
        else:
            raise ValueError("No fly models loaded. Call load_resources.")
    else:
        models = custom_models
        labels = model_labels or [f"Model {i}" for i in range(len(models))]
    return models, labels


def _sequence_from_structural_change(sc, genome, start, end, ins_seq=None):
    sequence_parts = []
    for chrm, s, e, strand in sc[start:end]:
        if ins_seq is not None and str(chrm).startswith("ins"):
            seq = Genome.sequence_to_encoding(ins_seq[s:e])
        else:
            seq = genome.get_encoding_from_coords(chrm, s, e)
        if strand == "-":
            seq = seq[None, ::-1, ::-1]
        else:
            seq = seq[None, :, :]
        sequence_parts.append(seq)
    if not sequence_parts:
        raise ValueError("Empty sequence extracted from structural change.")
    return np.concatenate(sequence_parts, axis=1)


def process_region(
    mchr,
    mstart,
    mend,
    genome,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=FLY_WINDOW_RADIUS,
    padding_chr="chr1",
    model_labels=None,
    use_cuda=True,
):
    """
    Generate fly predictions for the specified region.
    """
    genome = _get_fly_genome(genome)
    window_radius = _validate_fly_window(window_radius)
    models, model_labels = _resolve_fly_models(custom_models, model_labels)

    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()
    mpos = int((int(mstart) + int(mend)) / 2)
    wpos = coord_clip(mpos, chrlen, window_radius=window_radius)

    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]

    outputs_ref = genomepredict(
        sequence,
        mchr,
        mpos,
        wpos,
        models=models,
        targets=False,
        use_cuda=use_cuda,
    )

    return outputs_ref


def process_dup(
    mchr,
    mstart,
    mend,
    genome,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=FLY_WINDOW_RADIUS,
    padding_chr="chr1",
    model_labels=None,
    use_cuda=True,
):
    """
    Generate fly predictions for a duplication variant.
    """
    genome = _get_fly_genome(genome)
    window_radius = _validate_fly_window(window_radius)
    models, model_labels = _resolve_fly_models(custom_models, model_labels)

    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    target_dict = None
    if isinstance(target, dict):
        target_dict = target
        target = False

    if target:
        try:
            if target is True:
                target = [
                    label for label in model_labels if label in target_dict_global
                ]
            from selene_utils3 import Genomic2DFeatures

            target = [
                t if isinstance(t, Genomic2DFeatures) else target_dict_global[t]
                for t in target
            ]
        except Exception as e:
            print(e)
            target = False

    # ref.l
    wpos = coord_clip(mstart, chrlen, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]
    outputs_ref_l = genomepredict(
        sequence,
        mchr,
        mstart,
        wpos,
        models=models,
        targets=(
            target_dict.get("ref_l")
            if target_dict is not None
            else (
                [
                    torch.FloatTensor(
                        t.get_feature_data(
                            mchr,
                            coord_round(wpos - window_radius),
                            coord_round(wpos + window_radius),
                        )[None, :]
                    )
                    for t in target
                ]
                if target
                else None
            )
        ),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_l,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".ref.l.pdf",
        )

    # ref.r
    wpos = coord_clip(mend, chrlen, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]
    outputs_ref_r = genomepredict(
        sequence,
        mchr,
        mend,
        wpos,
        models=models,
        targets=(
            target_dict.get("ref_r")
            if target_dict is not None
            else (
                [
                    torch.FloatTensor(
                        t.get_feature_data(
                            mchr,
                            coord_round(wpos - window_radius),
                            coord_round(wpos + window_radius),
                        )[None, :]
                    )
                    for t in target
                ]
                if target
                else None
            )
        ),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_r,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".ref.r.pdf",
        )

    # alt
    s = StructuralChange2(mchr, chrlen)
    s.duplicate(mstart, mend)
    chrlen_alt = chrlen + mend - mstart
    wpos = coord_clip(mend, chrlen_alt, window_radius=window_radius)
    sequence = _sequence_from_structural_change(
        s, genome, wpos - window_radius, wpos + window_radius
    )
    outputs_alt = genomepredict(
        sequence,
        mchr,
        mend,
        wpos,
        models=models,
        targets=None,
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_alt,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".alt.pdf",
        )

    return outputs_ref_l, outputs_ref_r, outputs_alt


def process_del(
    mchr,
    mstart,
    mend,
    genome,
    cmap=None,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=FLY_WINDOW_RADIUS,
    padding_chr="chr1",
    model_labels=None,
    use_cuda=True,
):
    """
    Generate fly predictions for a deletion variant.
    """
    genome = _get_fly_genome(genome)
    window_radius = _validate_fly_window(window_radius)
    models, model_labels = _resolve_fly_models(custom_models, model_labels)

    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    target_dict = None
    if isinstance(target, dict):
        target_dict = target
        target = False

    if target:
        try:
            if target is True:
                target = [
                    label for label in model_labels if label in target_dict_global
                ]
            from selene_utils3 import Genomic2DFeatures

            target = [
                t if isinstance(t, Genomic2DFeatures) else target_dict_global[t]
                for t in target
            ]
        except Exception as e:
            print(e)
            target = False

    def _pick_targets(key):
        if isinstance(target, dict):
            return target.get(key)
        if isinstance(target, list):
            return target
        return None

    # ref.l
    wpos = coord_clip(mstart, chrlen, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]
    outputs_ref_l = genomepredict(
        sequence,
        mchr,
        mstart,
        wpos,
        models=models,
        targets=(
            target_dict.get("ref_l")
            if target_dict is not None
            else (
                [
                    torch.FloatTensor(
                        t.get_feature_data(
                            mchr,
                            coord_round(wpos - window_radius),
                            coord_round(wpos + window_radius),
                        )[None, :]
                    )
                    for t in target
                ]
                if target
                else None
            )
        ),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_l,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            cmap=cmap,
            model_labels=model_labels,
            file=file + ".ref.l.pdf",
        )

    # ref.r
    wpos = coord_clip(mend, chrlen, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]
    outputs_ref_r = genomepredict(
        sequence,
        mchr,
        mend,
        wpos,
        models=models,
        targets=(
            target_dict.get("ref_r")
            if target_dict is not None
            else (
                [
                    torch.FloatTensor(
                        t.get_feature_data(
                            mchr,
                            coord_round(wpos - window_radius),
                            coord_round(wpos + window_radius),
                        )[None, :]
                    )
                    for t in target
                ]
                if target
                else None
            )
        ),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_r,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            cmap=cmap,
            model_labels=model_labels,
            file=file + ".ref.r.pdf",
        )

    # alt
    s = StructuralChange2(mchr, chrlen)
    s.delete(mstart, mend)
    chrlen_alt = chrlen - (mend - mstart)
    wpos = coord_clip(mstart, chrlen_alt, window_radius=window_radius)
    sequence = _sequence_from_structural_change(
        s, genome, wpos - window_radius, wpos + window_radius
    )
    outputs_alt = genomepredict(
        sequence,
        mchr,
        mstart,
        wpos,
        models=models,
        targets=None,
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_alt,
            show_coordinates=True,
            cmap=cmap,
            model_labels=model_labels,
            file=file + ".alt.pdf",
        )

    return outputs_ref_l, outputs_ref_r, outputs_alt


def process_inv(
    mchr,
    mstart,
    mend,
    genome,
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=FLY_WINDOW_RADIUS,
    padding_chr="chr1",
    model_labels=None,
    use_cuda=True,
):
    """
    Generate fly predictions for an inversion variant.
    """
    genome = _get_fly_genome(genome)
    window_radius = _validate_fly_window(window_radius)
    models, model_labels = _resolve_fly_models(custom_models, model_labels)

    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    target_dict = None
    if isinstance(target, dict):
        target_dict = target
        target = False

    if target:
        try:
            if target is True:
                target = [
                    label for label in model_labels if label in target_dict_global
                ]
            from selene_utils3 import Genomic2DFeatures

            target = [
                t if isinstance(t, Genomic2DFeatures) else target_dict_global[t]
                for t in target
            ]
        except Exception as e:
            print(e)
            target = False

    # ref.l
    wpos = coord_clip(mstart, chrlen, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]
    outputs_ref_l = genomepredict(
        sequence,
        mchr,
        mstart,
        wpos,
        models=models,
        targets=(
            target_dict.get("ref_l")
            if target_dict is not None
            else (
                [
                    torch.FloatTensor(
                        t.get_feature_data(
                            mchr,
                            coord_round(wpos - window_radius),
                            coord_round(wpos + window_radius),
                        )[None, :]
                    )
                    for t in target
                ]
                if target
                else None
            )
        ),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_l,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".ref.l.pdf",
        )

    # ref.r
    wpos = coord_clip(mend, chrlen, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]
    outputs_ref_r = genomepredict(
        sequence,
        mchr,
        mend,
        wpos,
        models=models,
        targets=(
            target_dict.get("ref_r")
            if target_dict is not None
            else (
                [
                    torch.FloatTensor(
                        t.get_feature_data(
                            mchr,
                            coord_round(wpos - window_radius),
                            coord_round(wpos + window_radius),
                        )[None, :]
                    )
                    for t in target
                ]
                if target
                else None
            )
        ),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_r,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".ref.r.pdf",
        )

    # alt.l / alt.r
    s = StructuralChange2(mchr, chrlen)
    s.invert(mstart, mend)
    chrlen_alt = chrlen

    wpos = coord_clip(mstart, chrlen_alt, window_radius=window_radius)
    sequence = _sequence_from_structural_change(
        s, genome, wpos - window_radius, wpos + window_radius
    )
    outputs_alt_l = genomepredict(
        sequence,
        mchr,
        mstart,
        wpos,
        models=models,
        targets=None,
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_alt_l,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".alt.l.pdf",
        )

    wpos = coord_clip(mend, chrlen_alt, window_radius=window_radius)
    sequence = _sequence_from_structural_change(
        s, genome, wpos - window_radius, wpos + window_radius
    )
    outputs_alt_r = genomepredict(
        sequence,
        mchr,
        mend,
        wpos,
        models=models,
        targets=None,
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_alt_r,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".alt.r.pdf",
        )

    return outputs_ref_l, outputs_ref_r, outputs_alt_l, outputs_alt_r


def process_ins(
    mchr,
    mpos,
    ins_seq,
    genome,
    strand="+",
    file=None,
    custom_models=None,
    target=True,
    show_genes=True,
    show_tracks=False,
    window_radius=FLY_WINDOW_RADIUS,
    padding_chr="chr1",
    model_labels=None,
    use_cuda=True,
):
    """
    Generate fly predictions for an insertion variant.
    """
    genome = _get_fly_genome(genome)
    window_radius = _validate_fly_window(window_radius)
    models, model_labels = _resolve_fly_models(custom_models, model_labels)

    chrlen = [l for c, l in genome.get_chr_lens() if c == mchr].pop()

    target_dict = None
    if isinstance(target, dict):
        target_dict = target
        target = False

    if target:
        try:
            if target is True:
                target = [
                    label for label in model_labels if label in target_dict_global
                ]
            from selene_utils3 import Genomic2DFeatures

            target = [
                t if isinstance(t, Genomic2DFeatures) else target_dict_global[t]
                for t in target
            ]
        except Exception as e:
            print(e)
            target = False

    # ref
    wpos = coord_clip(mpos, chrlen, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        mchr, wpos - window_radius, wpos + window_radius
    )[None, :]
    outputs_ref = genomepredict(
        sequence,
        mchr,
        mpos,
        wpos,
        models=models,
        targets=(
            target_dict.get("ref")
            if target_dict is not None
            else (
                [
                    torch.FloatTensor(
                        t.get_feature_data(
                            mchr,
                            coord_round(wpos - window_radius),
                            coord_round(wpos + window_radius),
                        )[None, :]
                    )
                    for t in target
                ]
                if target
                else None
            )
        ),
        use_cuda=use_cuda,
    )

    # alt
    s = StructuralChange2(mchr, chrlen)
    s.insert(mpos, len(ins_seq), strand=strand)
    chrlen_alt = chrlen + len(ins_seq)

    wpos = coord_clip(mpos, chrlen_alt, window_radius=window_radius)
    sequence = _sequence_from_structural_change(
        s, genome, wpos - window_radius, wpos + window_radius, ins_seq=ins_seq
    )
    outputs_alt_l = genomepredict(
        sequence, mchr, mpos, wpos, models=models, targets=None, use_cuda=use_cuda
    )

    wpos = coord_clip(mpos + len(ins_seq), chrlen_alt, window_radius=window_radius)
    sequence = _sequence_from_structural_change(
        s, genome, wpos - window_radius, wpos + window_radius, ins_seq=ins_seq
    )
    outputs_alt_r = genomepredict(
        sequence,
        mchr,
        mpos + len(ins_seq),
        wpos,
        models=models,
        targets=None,
        use_cuda=use_cuda,
    )

    return outputs_ref, outputs_alt_l, outputs_alt_r


def process_custom(
    region_list,
    ref_region_list,
    mpos,
    genome,
    ref_mpos_list=None,
    anno_list=None,
    ref_anno_list=None,
    custom_models=None,
    target=True,
    file=None,
    show_genes=True,
    show_tracks=False,
    window_radius=FLY_WINDOW_RADIUS,
    model_labels=None,
    use_cuda=True,
):
    """
    Generate fly predictions for a custom variant by an ordered list of genomic segments.
    """
    genome = _get_fly_genome(genome)
    window_radius = _validate_fly_window(window_radius)
    models, model_labels = _resolve_fly_models(custom_models, model_labels)

    outputs_ref = []
    if ref_region_list:
        for i, (chrm, start, end, strand) in enumerate(ref_region_list):
            seq = genome.get_encoding_from_coords(chrm, start, end)
            if strand == "-":
                seq = seq[None, ::-1, ::-1].copy()
            else:
                seq = seq[None, :, :]
            mpos_ref = (
                ref_mpos_list[i]
                if ref_mpos_list is not None and i < len(ref_mpos_list)
                else start + window_radius
            )
            wpos_ref = start + window_radius
            outputs_ref.append(
                genomepredict(
                    seq,
                    chrm,
                    mpos_ref,
                    wpos_ref,
                    models=models,
                    targets=False,
                    use_cuda=use_cuda,
                )
            )

    sequence = []
    for chrm, start, end, strand in region_list:
        seq = genome.get_encoding_from_coords(chrm, start, end)
        if strand == "-":
            seq = seq[None, ::-1, ::-1].copy()
        else:
            seq = seq[None, :, :]
        sequence.append(seq)
    alt_sequence = np.concatenate(sequence, axis=1)

    alt_mpos = mpos if mpos is not None else window_radius
    outputs_alt = genomepredict(
        alt_sequence,
        "chimeric",
        alt_mpos,
        window_radius,
        models=models,
        targets=False,
        use_cuda=use_cuda,
    )

    return outputs_ref, outputs_alt


def process_single_breakpoint(
    chr1,
    pos1,
    chr2,
    pos2,
    orientation1,
    orientation2,
    genome,
    custom_models=None,
    target=True,
    file=None,
    show_genes=True,
    show_tracks=False,
    window_radius=FLY_WINDOW_RADIUS,
    padding_chr="chr1",
    model_labels=None,
    use_cuda=True,
):
    """
    Generate fly predictions for a simple translocation event.
    """
    genome = _get_fly_genome(genome)
    window_radius = _validate_fly_window(window_radius)
    models, model_labels = _resolve_fly_models(custom_models, model_labels)

    chrlen1 = [l for c, l in genome.get_chr_lens() if c == chr1].pop()
    chrlen2 = [l for c, l in genome.get_chr_lens() if c == chr2].pop()

    def _pick_targets(key):
        if isinstance(target, dict):
            return target.get(key)
        if isinstance(target, list):
            return target
        return None

    # ref 1
    wpos1 = coord_clip(pos1, chrlen1, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        chr1, wpos1 - window_radius, wpos1 + window_radius
    )[None, :]
    outputs_ref_1 = genomepredict(
        sequence,
        chr1,
        pos1,
        wpos1,
        models=models,
        targets=_pick_targets("ref_1"),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_1,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".ref.1.pdf",
            colorbar=True,
        )

    # ref 2
    wpos2 = coord_clip(pos2, chrlen2, window_radius=window_radius)
    sequence = genome.get_encoding_from_coords(
        chr2, wpos2 - window_radius, wpos2 + window_radius
    )[None, :]
    outputs_ref_2 = genomepredict(
        sequence,
        chr2,
        pos2,
        wpos2,
        models=models,
        targets=_pick_targets("ref_2"),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_ref_2,
            show_genes=show_genes,
            show_tracks=show_tracks,
            show_coordinates=True,
            model_labels=model_labels,
            file=file + ".ref.2.pdf",
            colorbar=True,
        )

    # alt junction
    pos1 = coord_clip(pos1, chrlen1, window_radius=window_radius)
    pos2 = coord_clip(pos2, chrlen2, window_radius=window_radius)

    if orientation1 == "+":
        seg1 = [chr1, pos1 - window_radius, pos1, "+"]
    else:
        seg1 = [chr1, pos1, pos1 + window_radius, "-"]

    if orientation2 == "+":
        seg2 = [chr2, pos2 - window_radius, pos2, "+"]
    else:
        seg2 = [chr2, pos2, pos2 + window_radius, "-"]

    sequence = []
    for chrm, start, end, strand in [seg1, seg2]:
        seq = genome.get_encoding_from_coords(chrm, start, end)
        if strand == "-":
            seq = seq[None, ::-1, ::-1].copy()
        else:
            seq = seq[None, :, :]
        sequence.append(seq)
    alt_sequence = np.concatenate(sequence, axis=1)

    outputs_alt = genomepredict(
        alt_sequence,
        "junction",
        window_radius,
        window_radius,
        models=models,
        targets=_pick_targets("alt"),
        use_cuda=use_cuda,
    )
    if file is not None:
        genomeplot(
            outputs_alt,
            show_coordinates=False,
            model_labels=model_labels,
            file=file + ".alt.pdf",
            colorbar=True,
        )

    return outputs_ref_1, outputs_ref_2, outputs_alt


def process_seqstr(
    seqstr_input,
    file=None,
    mpos=None,
    custom_models=None,
    model_labels=None,
    use_cuda=True,
):
    """
    Predict fly interactions for a 250kb sequence from Seqstr input.
    """
    if seqstr is None:
        raise ImportError(
            "Seqstr is not installed. Please install it first. pip install seqstr"
        )

    seqstrout = seqstr(seqstr_input)
    sequence_str = seqstrout[0].Seq

    if len(sequence_str) < FLY_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length needs to be at least {FLY_SEQUENCE_LENGTH} bp. "
            f"Current length is {len(sequence_str)}."
        )
    if len(sequence_str) > FLY_SEQUENCE_LENGTH:
        midpoint = int(len(sequence_str) / 2)
        sequence_str = sequence_str[
            midpoint - FLY_WINDOW_RADIUS : midpoint + FLY_WINDOW_RADIUS
        ]

    if custom_models is None:
        models = list(fly_model_dict_global.values())
    else:
        models = custom_models
        if model_labels is None:
            model_labels = [f"Model {i}" for i in range(len(models))]

    wpos = FLY_WINDOW_RADIUS
    if mpos is None:
        mpos = FLY_WINDOW_RADIUS

    sequence = Genome.sequence_to_encoding(sequence_str)[None, :, :]

    outputs_ref = genomepredict(
        sequence,
        "customized seq",
        mpos,
        wpos,
        models=models,
        targets=False,
        use_cuda=use_cuda,
    )

    return outputs_ref


if __name__ == "__main__":
    from docopt import docopt
    import sys
    import os
    import re

    doc = """
    Orca-Fly prediction tool.

    Usage:
    orca_predict region [options] <coordinate> <output_dir>
    orca_predict del [options] <coordinate> <output_dir>
    orca_predict dup [options] <coordinate> <output_dir>
    orca_predict inv [options] <coordinate> <output_dir>
    orca_predict break [options] <coordinate> <output_dir>

    Options:
    -h --help        Show this screen.
    --nocuda         Use CPU implementation.
    --coor_filename  Include coordinate in the output filenames.
    --version        Show version.
    """
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    arguments = docopt(doc, version="Orca-Fly v0.1")
    use_cuda = not arguments["--nocuda"]
    coor_filename = arguments["--coor_filename"]

    load_resources(use_cuda=use_cuda)

    if arguments["region"]:
        predtype = "region"
    elif arguments["del"]:
        predtype = "del"
    elif arguments["dup"]:
        predtype = "dup"
    elif arguments["inv"]:
        predtype = "inv"
    elif arguments["break"]:
        predtype = "break"
    else:
        raise ValueError("Unexpected prediction type!")

    if coor_filename:
        suffix = "_" + re.sub(r'[\/*?:"<>|]', "_", arguments["<coordinate>"])
    else:
        suffix = ""

    def predict(chrm, start, end, savedir):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        outputs = process_region(
            chrm,
            start,
            end,
            dm6,
            use_cuda=use_cuda,
        )
        torch.save(outputs, savedir + "/orca_pred" + suffix + ".pth")
        return None

    def get_interactions(predtype, content, savedir):
        if predtype == "region":
            chrstr, coordstr = str(content).split(":")
            chrstr = "chr" + chrstr.replace("chr", "")
            coord_s, coord_e = coordstr.split("-")
            predict(chrstr, int(coord_s), int(coord_e), savedir)
        elif predtype in ["dup", "del"]:
            chrstr, coordstr = str(content).split(":")
            chrstr = "chr" + chrstr.replace("chr", "")
            coord_s, coord_e = coordstr.split("-")

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            if predtype == "dup":
                outputs_ref_l, outputs_ref_r, outputs_alt = process_dup(
                    chrstr,
                    int(coord_s),
                    int(coord_e),
                    dm6,
                    use_cuda=use_cuda,
                )
            else:
                outputs_ref_l, outputs_ref_r, outputs_alt = process_del(
                    chrstr,
                    int(coord_s),
                    int(coord_e),
                    dm6,
                    use_cuda=use_cuda,
                )
            torch.save(
                {
                    "outputs_ref_l": outputs_ref_l,
                    "outputs_ref_r": outputs_ref_r,
                    "outputs_alt": outputs_alt,
                },
                savedir + "/orca_pred" + suffix + ".pth",
            )
        elif predtype == "inv":
            chrstr, coordstr = str(content).split(":")
            chrstr = "chr" + chrstr.replace("chr", "")
            coord_s, coord_e = coordstr.split("-")

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            outputs_ref_l, outputs_ref_r, outputs_alt_l, outputs_alt_r = process_inv(
                chrstr,
                int(coord_s),
                int(coord_e),
                dm6,
                use_cuda=use_cuda,
            )
            torch.save(
                {
                    "outputs_ref_l": outputs_ref_l,
                    "outputs_ref_r": outputs_ref_r,
                    "outputs_alt_l": outputs_alt_l,
                    "outputs_alt_r": outputs_alt_r,
                },
                savedir + "/orca_pred" + suffix + ".pth",
            )
        elif predtype == "break":
            chr_coord_1, chr_coord_2, orientations = str(
                content.replace("	", " ")
            ).split(" ")
            chr1, coord1 = chr_coord_1.split(":")
            chr2, coord2 = chr_coord_2.split(":")
            chr1 = "chr" + chr1.replace("chr", "")
            chr2 = "chr" + chr2.replace("chr", "")
            orientation1, orientation2 = orientations.split("/")

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            outputs_ref_1, outputs_ref_2, outputs_alt = process_single_breakpoint(
                chr1,
                int(coord1),
                chr2,
                int(coord2),
                orientation1,
                orientation2,
                dm6,
                use_cuda=use_cuda,
            )

            torch.save(
                {
                    "outputs_ref_1": outputs_ref_1,
                    "outputs_ref_2": outputs_ref_2,
                    "outputs_alt": outputs_alt,
                },
                savedir + "/orca_pred" + suffix + ".pth",
            )
        else:
            raise ValueError("Unexpected prediction type!")
        return None

    get_interactions(predtype, arguments["<coordinate>"], arguments["<output_dir>"])
