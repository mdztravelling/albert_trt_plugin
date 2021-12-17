* Note

  TensorRT albert plugin for TensorRT-8.0.3.4.

* How to use it?
  - copy plugin/embLayerNormPlugin/embFactorizedLayerNormPlugin.* and embLayerNormKernel.cu to tensorrt-oss plugin/embLayerNormPlugin.
  - build tensorrt-oss and copy libnvinfer_plugin* to TensorRT-8.0.3.4/lib.
  - demo/BERT/builder_fixedseq.py is convert tensorflow model to tensorrt model.

