#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 02:55:02 2019

@author: ravi
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM


TOP_DOWN_PYRAMID_SIZE = 256


RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
RPN_ANCHOR_STRIDE = 1





def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth_of_image, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * D * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * D * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * D * anchors_per_location, (dy, dx, dz, log(dh), log(dw), log(dd))] Deltas to be
                  applied to anchors.
    """
    
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv3D(512, (3, 3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, depth, anchors per location * 2].
    x = KL.Conv3D(2 * anchors_per_location, (1,1,1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1,  2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, D, anchors per location * depth]
    # where depth is [x, y, z, log(w), log(h), log(d)]
    x = KL.Conv3D(anchors_per_location * 4, (1,1,1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)
    # Reshape to [batch, anchors, 8]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 8]))(x)
    return [rpn_class_logits, rpn_probs, rpn_bbox]

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * D * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * D * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * D * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")

rpn = build_rpn_model(RPN_ANCHOR_STRIDE,
                              len(RPN_ANCHOR_RATIOS), TOP_DOWN_PYRAMID_SIZE)