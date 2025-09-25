# -*- coding: utf-8 -*-
"""
File Name:              json_support.py
Description:            Provides JSON-based serialization and deserialization
                        utilities for saving and loading MDN model parameters,
                        replacing pickle-based workflows.

Date Created:           September 23rd, 2025
Author:                 Arun M Saranathan
Email:                  arun.saranathan@ssaihq.com
                        fnu.arunmuralidharansaranathan@nasa.gov
"""

import importlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.base import TransformerMixin

from .transformers import TransformerPipeline


# -------------------
# Helpers for sklearn Transformers
# -------------------

def serialize_list(lst):
    """Recursively convert a list with numpy types into JSON-serializable form."""
    if isinstance(lst, list):
        return [serialize_list(x) for x in lst]
    elif isinstance(lst, np.ndarray):
        return lst.tolist()
    elif isinstance(lst, (np.integer, int)):
        return int(lst)
    elif isinstance(lst, (np.floating, float)):
        return float(lst)
    elif isinstance(lst, (np.bool_, bool)):
        return bool(lst)
    else:
        return lst

def serialize_transformer(transformer):
    """Convert any sklearn TransformerMixin into a JSON-safe dict."""
    params = make_serializable(transformer.get_params(deep=False))
    state = {}
    for attr, val in transformer.__dict__.items():
        if isinstance(val, np.ndarray):
            state[attr] = val.tolist()
        elif isinstance(val, (np.generic,)):  # np.int32, np.float64, etc.
            state[attr] = val.item()
        elif isinstance(val, list):
            state[attr] = serialize_list(val)
        else:
            try:
                json.dumps(val)  # check if natively serializable
                state[attr] = val
            except TypeError:
                pass  # skip non-serializable internals

    cls = transformer.__class__
    return {
        "_kind": "SklearnTransformer",
        "class_path": cls.__module__ + "." + cls.__name__,
        "params": params,
        "state": state,
    }


def deserialize_transformer(data):
    """Recreate any sklearn TransformerMixin from serialized form."""
    module_name, cls_name = data["class_path"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)

    transformer = cls(**data.get("params", {}))

    # restore learned attributes
    for attr, val in data.get("state", {}).items():
        setattr(transformer, attr, np.array(val) if isinstance(val, list) else val)

    return transformer


# ----------------------------
# Handle TransformerPipeline
# ----------------------------

def serialize_pipeline(pipeline):
    return {
        "_kind": "TransformerPipeline",
        "scalers": [make_serializable(s) for s in pipeline.scalers],
    }


def deserialize_pipeline(data):
    scalers = [restore_object(s) for s in data["scalers"]]
    return TransformerPipeline(scalers=scalers)


# ----------------------------
# General JSON (de)serializers
# ----------------------------

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, Path):
        return {"_kind": "Path", "data": str(obj)}
    elif isinstance(obj, slice):
        return {"_kind": "Slice", "start": int(obj.start), "stop": int(obj.stop), "step": obj.step}
    elif isinstance(obj, np.random.RandomState):
        state = obj.get_state()
        state_list = list(state)
        state_list[1] = state_list[1].tolist()  # ndarray -> list
        return {"_kind": "NumpyRandomState", "data": state_list}
    elif isinstance(obj, tf.random.Generator):
        return {
            "_kind": "TFGenerator",
            "state": obj.state.numpy().tolist(),
            "algorithm": int(obj.algorithm),
        }
    elif isinstance(obj, TransformerPipeline):
        return serialize_pipeline(obj)
    elif isinstance(obj, TransformerMixin):
        return serialize_transformer(obj)
    else:
        return obj


def restore_object(obj):
    if isinstance(obj, dict) and "_kind" in obj:
        kind = obj["_kind"]

        if kind == "Path":
            return Path(obj["data"])

        elif kind == "Slice":
            return slice(int(obj["start"]), int(obj["stop"]), obj["step"])

        elif kind == "NumpyRandomState":
            state = list(tuple(obj["data"]))
            state[1] = np.array(state[1], dtype=np.uint32)  # convert list -> ndarray
            rs = np.random.RandomState()
            rs.set_state(tuple(state))
            return rs

        elif kind == "TFGenerator":
            return tf.random.Generator.from_state(
                state=tf.constant(obj["state"], dtype=tf.uint64),
                alg=obj["algorithm"]
            )

        elif kind == "TransformerPipeline":
            return deserialize_pipeline(obj)

        elif kind == "SklearnTransformer":
            return deserialize_transformer(obj)

    elif isinstance(obj, dict):
        return {k: restore_object(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [restore_object(v) for v in obj]

    else:
        return obj


# --------------------------
# I/O
# --------------------------

def store_json(path, obj):
    serializable = make_serializable(obj)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def read_json(path):
    with open(path, "r") as f:
        obj = json.load(f)
    return restore_object(obj)
