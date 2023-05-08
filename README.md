# CasViGE: Learning Robust Point Cloud Registration with Cascaded Visual-Geometric Encoding

## Data Preparation

First, download preprocessed 3DMatch point cloud data from (Predator)[https://github.com/prs-eth/OverlapPredator].

```bash
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/Predator/data.zip
```

And also download the preprocessed RGB images from (Google Drive)[https://drive.google.com/file/d/17K6n4-Xjn9n0qc9sEmUqsQ_lrZnlw7bv/view?usp=share_link].

Finally, place the downloaded data in `data/3DMatch/data` and organize the data as follows:

```text
--data--3DMatch--metadata
              |--data--train--7-scenes-chess--cloud_bin_0.pth
                    |      |               |--...
                    |      |--...
                    |--test--7-scenes-redkitchen--cloud_bin_0.pth
                    |     |                    |--...
                    |     |--...
                    |--frames--7-scenes--chess--cloud_bin_0_0.color.png
                            |                |--cloud_bin_0_0.depth.png
                            |                |--cloud_bin_0_0.pose.txt
                            |                |--...
                            |--...
```

## Dependencies

Install `https://github.com/qinzheng93/vision3d`.

## Testing

We provide the implementation of Predator-based CasViGE. The pretrained model is placed in `weights`.

Run the following command to evaluate the pretrained model on 3DMatch:

```bash
cd experiments/predator.casvige.3dmatch.gcn.prob_sel
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/absolute/path/to/casvige/weights/predator-casvige.pth --benchmark=3DMatch
python eval.py --benchmark=3DMatch --num_keypoints=5000 --method=ransac --mutual=True
```

Replace `--benchmark=3DMatch` with `--benchmark=3DLoMatch` to evaluate on 3DLoMatch.

Note: as there is randomness in keypoint sampling and RANSAC, the results could be slightly different from the paper.

## Training

Run the following command to train from scratch.

```bash
cd experiments/predator.casvige.3dmatch.gcn.prob_sel
CUDA_VISIBLE_DEVICES=0 python trainval.py
```
