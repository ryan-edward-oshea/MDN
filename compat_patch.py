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

import sys

from packaging import version


def patch_numpy():
    """
    Patch NumPy for compatibility across versions:
    - Add np.random._randomstate_ctor if missing (NumPy >=1.19).
    - Shim numpy._core and its submodules for NumPy <2.0 to support pickles.
    - Stub new BitGenerators (MT19937, PCG64, Philox, SFC64) with RandomState fallback.
      NOTE: RNG state will NOT be preserved exactly, but unpickling will succeed.
    """
    try:
        import sys
        import types
        import numpy as np
        import numpy.random
        from packaging import version

        numpy_version = np.__version__
        print(f"[compat_patch] NumPy version: {numpy_version}")

        # Patch missing _randomstate_ctor
        if version.parse(numpy_version) >= version.parse("1.19"):
            if not hasattr(np.random, "_randomstate_ctor"):
                def _randomstate_ctor(*args, **kwargs):
                    return np.random.RandomState(*args, **kwargs)

                np.random._randomstate_ctor = _randomstate_ctor
                print("[compat_patch] ✅ Applied NumPy _randomstate_ctor patch.")

        # Shim numpy._core for NumPy <2.0
        major = int(numpy_version.split(".")[0])
        if major < 2:
            if "numpy._core" not in sys.modules:
                # Create fake numpy._core package
                np_core = types.ModuleType("numpy._core")
                sys.modules["numpy._core"] = np_core

                # Explicitly map submodules used in pickles
                sys.modules["numpy._core.multiarray"] = np.core.multiarray
                sys.modules["numpy._core.numerictypes"] = np.core.numerictypes
                if hasattr(np.core, "overrides"):
                    sys.modules["numpy._core.overrides"] = np.core.overrides

                print("[compat_patch] ✅ Applied NumPy _core shim (multiarray, numerictypes, overrides).")

        # Stub new BitGenerators for pickle loading
        class RNGStub:
            """Fallback to np.random.RandomState for unpickling old pickles"""

            def __init__(self, *args, **kwargs):
                self.rng = np.random.RandomState(42)

            def __getattr__(self, attr):
                return getattr(self.rng, attr)

        bitgenerators = {
            "numpy.random._mt19937": "MT19937",
            "numpy.random._pcg64": "PCG64",
            "numpy.random._philox": "Philox",
            "numpy.random._sfc64": "SFC64",
        }

        for module_name, class_name in bitgenerators.items():
            if module_name not in sys.modules:
                fake_module = types.ModuleType(module_name)
                setattr(fake_module, class_name, RNGStub)
                sys.modules[module_name] = fake_module
                print(f"[compat_patch] ⚠️ Stubbed {module_name}.{class_name} with RNGStub.")

    except ImportError:
        print("[compat_patch] ⚠️ NumPy not installed, skipping patch.")


def patch_tensorflow():
    """Patch TensorFlow for missing Trackable Reference in versions >= 2.11."""
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
