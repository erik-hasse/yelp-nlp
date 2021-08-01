#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh

conda activate hf

uvicorn --host 0.0.0.0 review_app:app
