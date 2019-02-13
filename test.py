#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:13:34 2019

@author: rishialluri
"""
from imaging.file_operations import file_operations

fo = file_operations()

fo.read_rois("data/roi.1024.tif")
fo.read_video("data/video_3007.nd2")
