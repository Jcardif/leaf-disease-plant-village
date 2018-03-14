#!/usr/bin/env bash
python save_resnet18_test_scores.py --dataseparate '10-90' && \
    python save_resnet18_test_scores.py --dataseparate '20-80' && \
    python save_resnet18_test_scores.py --dataseparate '40-60' && \
    python save_resnet18_test_scores.py --dataseparate '50-50' && \
    python save_resnet18_test_scores.py --dataseparate '60-40'&& \
    python save_resnet18_test_scores.py --dataseparate '80-20'

