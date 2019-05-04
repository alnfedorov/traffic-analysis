This repository contains all source codes relevant to the NAME research paper. 
We used official 
[Detectron realization](https://github.com/facebookresearch/maskrcnn-benchmark/tree/f917a555bc422ed5e06a402e739da0e21b00d0b5) 
from Facebook as a start point. Here is an example of how implemented system works at [day](examples/day.mp4) and at 
[night](examples/night.mp4)(better to download and open in local video player). 
If you are interest in dataset and/or our trained models, please email us(canxes@mail.ru, nikolskaiaki@susu.ru).

One should not consider provided sources as an off-the shelf implementation. It is not a production-ready solution and 
still only a draft of what can be achieved with modern convolutional networks in the field of traffic analysis. 
Provided project is intended to serve as a complete reference about implementation details and neural network 
architecture/hyperparameters.

Short summary of the most relevant directories and files:
- **traffic/models** - config files for used models. Start from here if you are interest in precise architecture 
of the detection network.
- **traffic/utils** - sources for SORT tracker, drawing procedures, model inference and statistics building.
- **traffic/scripts/plot_predicts.py** - run trained model on a small video fragment to get 
per frame rendered predictions. Be aware that pretty matplotlib rendering takes time.
- **traffic/scripts/parse_is_archives.py** - API calls to download and concat video fragments from is74 archive, 
this archive was used as our main data source in our work. Requires login and password to access.

Refer to the **maskrcnn_benchmark/modeling** for realization of the feature aggregation pooling and focal loss. 

Note, that one should be careful with absolute paths in [config](traffic/models) files and *[path_catalog.py](traffic/paths_catalog.py)*. 
In addition, many scripts in this repository should be run block-by-block rather than in a console mode.
