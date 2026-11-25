# Input-independent code (line numbers in this file):
#   reference binding and local allocation: 16–18.

import os
import io
import uuid
import json
import sys
from . import storage


def handler(event):
    # ---- Input-independent BEGIN: reference binding and local allocation ----
    # With DASH, frames can be bound as a reference before upstream predictions are available.
    # Therefore, this binding and the local allocation are input-independent and can execute without waiting for upstream output.
    frames = event["frames"]

    logs = {}
    # ---- Input-independent END: reference binding and local allocation ----

    # ---- Input-dependent BEGIN: merge upstream predictions ----
    for xs in frames:
      for key,value in xs.items():
        logs[key] = value

    return logs
    # ---- Input-dependent END: merge upstream predictions ----

