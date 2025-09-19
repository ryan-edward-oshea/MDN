#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name:              compat_patch.py
Description:            This code file will be used apply patches to enable the compatibility of pickle load with the
                        old version of numpy and Tensorflow

Date Created:   September 9th, 2025
Author:         Arun M Saranathan
Email:          arun.saranathan@ssaihq.com/
                fnu.arunmuralidharansaranathan@nasa.gov
"""

import sys, types
from packaging import version

def patch_numpy():
    """Patch NumPy for missing _randomstate_ctor in versions >= 1.19."""
    try:
        import numpy as np
        import numpy.random
        numpy_version = np.__version__
        print(f"[compat_patch] NumPy version: {numpy_version}")

        if version.parse(numpy_version) >= version.parse("1.19"):
            if not hasattr(np.random, "_randomstate_ctor"):
                def _randomstate_ctor(*args, **kwargs):
                    return np.random.RandomState(*args, **kwargs)
                np.random._randomstate_ctor = _randomstate_ctor
                print("[compat_patch] ✅ Applied NumPy _randomstate_ctor patch.")
    except ImportError:
        print("[compat_patch] ⚠️ NumPy not installed, skipping patch.")

def patch_tensorflow():
    """Patch TensorFlow for missing TrackableReference in versions >= 2.11."""
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        print(f"[compat_patch] TensorFlow version: {tf_version}")

        if version.parse(tf_version) >= version.parse("2.11.0"):
            # Full module path needed by older pickles
            base_module_name = "tensorflow.python.training.tracking.base"

            if base_module_name not in sys.modules:
                import types
                sys.modules[base_module_name] = types.ModuleType(base_module_name)

            base_module = sys.modules[base_module_name]

            if not hasattr(base_module, "TrackableReference"):
                class TrackableReference:
                    def __init__(self, *args, **kwargs):
                        pass
                setattr(base_module, "TrackableReference", TrackableReference)

                print("[compat_patch] ✅ Applied TensorFlow TrackableReference patch (tracking.base).")
    except ImportError:
        print("[compat_patch] ⚠️ TensorFlow not installed, skipping patch.")


def apply_all():
    """Apply all compatibility patches."""
    patch_numpy()
    patch_tensorflow()

# Run automatically on import
apply_all()