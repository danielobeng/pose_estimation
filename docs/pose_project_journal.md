# Pose estimation and tracking in PyTorch

## Problem statement

- The goal of this project in mainly to improve on the default results achievable in PyTorch's `keypointRCNN` model, using various techniques, as well as creating a comparable model for 23 keypoints using the foot dataset (which is smaller)

## Model architecture

- The model architecture is PyTorch's implementation of keypoints RCNN with some modifications to take advantage of the pre-trained models but to train on 23 keypoints

## Data

We used 2 sources of data, and merged

- The general format of the keypoint data is
  - `[x, y, v]`
  - `x` and `y` are the coordinates of the keypoint, `v` indicates whether a keypoint is visible in the image, occluded or outside the image frame

### EDA

- first look
- examples (image, labels)
- summary statistics
- graphs

  - sns distribution plots
  - scatter plot
    - fit regressions to see if correlations
  - violin plot

- Looking at the data, notice that all the keypoints are treated independently. In reality, we know that keypoints are structurally connected by the skeletal structure of the body, yet we only take this into consideration when visualizing the data. This seems like valuable information that could improve the results

### Feature Engineering

- is there anything we can do with this data to improve the model before we even run it
- better quality labels brute force
- anything else?

### Data munging

- In order to train the model on 23 keypoints, we need a dataset that contains all 23 keypoints
- Fortunately, there exists a dataset for this: https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/ which contains an additional 6 keypoints, formatted in the COCO format
- Combining these datasets requires therefore finding the keypoints from the foot dataset with matching IDs from the original COCO dataset and producing a new dataset containing both by appending the feet keypoints to the end of the keypoint list

## Metrics

- what does success look like for this model? beyond the metrics, does this create value in some way?

### Object Keypoint Similarity (OKS)

- The team behind the COCO dataset created a metric called Object Keypoint Similarity
  - The "Keypoint Similarity" part is essentially a Euclidean distance metric calculation between the coordinates of the ground truth label and the predicted keypoint coordinates with some additional considerations
    - When using this methodology, it is implied that multiple annotators will annotate the same keypoints on the same image, such that a mean and standard deviation of their annotation locations can be generated
    - A Gaussian distribution with this mean and standard deviation is used to delimit an "area" within which the keypoint coordinate can lie
    - ![Pasted image 20220830001043.png](../media/Pasted%20image%2020220830001043.png?raw=true)
      (Image source source from [Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation](https://openaccess.thecvf.com/content_ICCV_2017/papers/Ronchi_Benchmarking_and_Error_ICCV_2017_paper.pdf))
    - This implies that, despite the average label coordinate for a keypoint being a single pixel location, a broader definition is used to define a "correct" keypoint classification. There are many pixels in any given image that are considered part of the eye. There must still however be some threshold to define what is part of the eye and what is not.
    - One method could be to do image segmentation on each joint area, but this is more time consuming and not the approach used here.
    - By considering the variance of the keypoint annotators ( $k_i^2$ ), who inevitably place (inter-rater and intra-rater) the keypoint in different locations, a sort of "wisdom of the crowds" approach is used to delimit the keypoints:
      - Keypoints that are easier to locate precisely will have smaller variances (e.g. eyes), harder to locate or poorly defined joints ("hip") end up with a larger variance. The larger the variance, the more tolerance there will be on the final measure score
      - During inference, the closer the predicted keypoint is to it's annotation mean value, the higher the keypoint similarity with a maximum value of 1 (when a prediction = mean annotation value) - the spread of the Gaussian determines how quickly the keypoint similarity score increases
      - If the prediction falls outside of the defined threshold set by the Gaussian curve, the prediction is truncated/rounded to 0
- we are limited by the depth of field and resolution of any given image, that is the point at which an image's blur circle is larger than the size of a pixel - given there is an inherent variance in the focal lengths of the cameras used in the images which we do not know, we should expect additional variance to be caused from this - even if we could specify a specific location, human anatomy varies per person, creating ambiguity on what that point really is

#### Calculating Keypoint Similarity ( $ks$ )

$$
ks(\hat\theta_i^{(p)}, \theta_i^{(p)}) = e^{-\frac{||\hat\theta_i^{(p)} - \theta_i^{(p)}||^2_2}{2s^2k^2_i}}
$$

- $ks$ is measured for each keypoint $i$ on each person $p$ by finding the L2 norm between the predicted ( $\hat{\theta}$ ) and the ground truth ( $\theta$ ) coordinate values. ( $\theta$ ) is the mean annotation value
- $s$ is the scaling factor - the square root of the object segment area, which is the area of each individual bounding box. This is for scale invariance between each person detected.
  - Why do we need scale invariance?
  - The smallest unit of length in a digital image is a pixel. However, the distance each pixel represents in an image differs. A pixel far in the background of an image represents a larger area than a pixel of something close up to the camera lens.
  - Therefore, a variance of one pixel should be more significant for a person far into the background of an image versus a person very close to the camera.
- $k_i$ is the standard deviation from the keypoint annotators, specific to each keypoint $i$
  - The values for $k_i$ are in the table below. Most were calculated by the MS COCO team as the original annotators. The last 6 keypoints representing foot locations are assumed to have similar variance to wrists, as gathering more accurate values would require running a study with annotators based on where they place the markers.

| keypoint        | std  |
| --------------- | ---- |
| nose            | 0.26 |
| left_eye        | 0.25 |
| right_eye       | 0.25 |
| left_ear        | 0.35 |
| right_ear       | 0.35 |
| left_shoulder   | 0.79 |
| right_shoulder  | 0.79 |
| left_elbow      | 0.72 |
| right_elbow     | 0.72 |
| left_wrist      | 0.62 |
| right_wrist     | 0.62 |
| left_hip        | 1.07 |
| right_hip       | 1.07 |
| left_knee       | 0.87 |
| right_knee      | 0.87 |
| left_ankle      | 0.89 |
| right_ankle     | 0.89 |
| left_big_toe    | 0.62 |
| left_small_toe  | 0.62 |
| left_heel       | 0.62 |
| right_big_toe   | 0.62 |
| right_small_toe | 0.62 |
| right_heel      | 0.62 |

- Euler's number and the negative exponentiation are used to normalise values between 0 and 1

#### Calculating $OKS$

- To find the final $OKS$ score, we take the average keypoint similarity scores for each person $p$ for all visible keypoints ( $v_i$ > 0 where $v_i=1$ means occluded keypoint, $v_i=2$ means visible keypoint, $\delta$ function returns 1 if $v_i > 0$ )

$$
OKS(\hat\theta_i^{(p)}, \theta_i^{(p)}) = \frac{\sum_i{ks(\hat\theta_i^{(p)}, \theta_i^{(p)})\delta(v_i>0)}}{\sum_i{\delta(v_i>0)}}
$$

### $OKS$ Thresholding

- Using $OKS$ values, we can set a threshold at which a predicted keypoint is deemed a 'correct' prediction. This is means, if the prediction is within a certain 'standard deviation'
  - this turns our problem from a regression problem into a classification problem, allowing us to measure an accuracy, precision and recall as we have can measure true positive, false positive and false negative values
    - In our case accuracy is less helpful because we are also interested in the keypoints we miss (false negatives) as much as how precise we are with the ones we detect. Accuracy is not a good metric when a high number of false negatives are predicted. Precision and recall however are very useful.
  - We can then create precision recall curves and use the AUC as our single metric to track how well the model is making predictions

### Mean Average Precision mAP

- Precision-recall curve

  - the curve represents how many misses (false negatives) our model makes for a given confidence level
  - there are 2 thresholds
    - the confidence the model allows for which improves the recall value
    - oks threshold represented by the different lines
  - the reason why the PR curve might not reach a recall value of 1 is because there are keypoints (people) that are never detected no matter how uncertain the model is about a prediction.

- How the area under the curve is calculated for the precision recall values matters - the COCO implementation uses interpolation between points which is an optimistic answer
  - We can choose different OKS thresholds (eg OKs = 0.5, 0.6, 0.7, 0.8, 0.9, 0.95) to understand the trade-off between making correct and precise predictions and how confident the model is in it's predictions. If less confident predictions are allowed, precision will drop but fewer keypoints will be missed.

## Model Training

- Pytorch's keypoint RCNN model pre-trained on 17 keypoints was used initially but modified for 23 keypoints.
  - It is likely that the initial layers of the nnet already know all the shapes that might also be useful for detecting feet keypoints, so we can focus the training on the later layers of the model by only training from the roi

```python
for name, param in self.named_parameters():
	if "roi" not in name:
		param.requires_grad = False
```

### Overfitting the model

- Initially, the model is trained on 11 examples only for the training set and 11 examples for the validation set, and run for 500 epochs
- This is performed to ensure the model can be trained without issue - if the model cannot be overfit, then the model is unable to reach losses low enough to provide a useful solution
- As the training loss continues to decrease, the validation loss rises - this tells us that our model is overfit
  ![[W&B Chart 8_29_2022, 11_34_23 PM (1).svg]]

- This is also an opportunity to test the analysis metrics
- Below is the precision-recall curve for the predictions on the training data. It is possible to achieve 1.0 PR score for Oks@0.5, but interestingly we cannot seem to find any keypoints when 0ks > 0.9
- It is unclear why this may be, it could be due to this being such a high standard applied to a small number of examples (which may indicate the difficulty of achieving this level of precision and recall for this model)
  ![[Pasted image 20220829235043.png|500]]

### AdamW vs SGD

- Using 1 cycle policy on the pre-trained model, it doesn't seem like it matters whether we used AdamW or SGD
  ![[W&B Chart 9_1_2022, 1_35_08 PM (1).svg|500]]

### Adaptive learning rate

- Using an adaptive learning rate makes sense - In this case, using the 1cycle policy is used from https://arxiv.org/abs/1803.09820 - The idea is to train the model with 3 phases of learning rates 1. Increasing learning rates 2. Decreasing learning rates 3. Lower learning rate than the starting rate
  ![[W&B Chart 9_1_2022, 12_08_28 PM.svg|500]]
- Intuitively, this makes a lot of sense - Smith observed that midway through training, models can get stuck in a local minima too early - with a higher learning rate it makes it easier for the model to find a better minimum and then anneal the learning rate to get as close to the final local minimum as possible
- In theory, this allows to both train the model with higher learning rates and reduce overfitting, because a high learning rate acts as a regulariser, it keeps the model from immediately settling

- Scheduling 1 cycle vs no scheduling
  ![[W&B Chart 9_1_2022, 7_39_52 PM.svg|500]]
  ![[W&B Chart 9_1_2022, 7_40_05 PM.svg|500]]
- Gradient accumulation
  - show gradient accum vs non gradient accum graps

### Error analysis

#### Model debugging

##### Activations

- Checking the model activations are relatively smooth (especially not collapsing to 0) and are not mostly 0. If a large percentage (90%+, although the lower the better) of the activations end up being close to 0 (using a threshold of 0.05), then
- Pytorch hooks are used to get this information from the model on a few selected layers

####

- It seems like
  ![[Pasted image 20221010153443.png|500]] ![[Pasted image 20220903112023.png|500]]

- The first thing required is to implement a tool for error analysis on the validation set
  - this will need to be run after training is complete so that we get the parameters after training
  - We then need to calculate the relevant metrics for each example (OKS) and choose something like the 100 worst to look at
  - We should also include keypoint scores, because we want to see the confidence the algo had in its decisions

#### What the error types are and where they occur (train set, test set)

- Examples of error types we can see in the worst fit examples on our **validation set** - False positive human detections
  ![[Pasted image 20220903105025.png|500]]
- Making detections on more keypoints than are visible
  ![[Pasted image 20220903105131.png|500]] ![[Pasted image 20220903111353.png|500]]

![[Pasted image 20220903115803.png|500]] ![[Pasted image 20220903115723.png|500]] ![[Pasted image 20220903115630.png|500]]

#### What can I do to solve them and how easy each solution is ( how confident you are in the solution working - you might need to assess this somehow)

     ![[Screenshot 2022-07-10 at 10.56.22.png|500]]

- error analysis on entire pipeline "augmentative analysis/enrichment analysis"

  - create a diagram of ml pipeline even through the model (as has backbone)
  - eg perfect bounding boxes to see how much of the error is caused by bounding box being incorrect

- Progressively correct different types of errors to see what is the worst
- Progressively correcting/looking at what keypoint causes the most error

- Ablative analysis

  - remove components from pipeline one at a time to see where it breaks

- fastai implementation

### Changing the model size

- Until now, a resnet50 model has been used as the backbone for training the model as it is faster it iterate. It is worth experimenting with different models and model sizes that can produce more accurate results

### Resnet-152

- The changes for using a resnet 152 model are straightforward, the backbone of the model must be changed

```python
backbone = resnet152(
pretrained=True,
progress=True,
norm_layer=misc_nn_ops.FrozenBatchNorm2d,
)
```

- Given the rest of the model is built off the keypoint rcnn model in pytorch, using any existing weight from the rest of the network is not possible for fine tuning, only the resnet part of the model does not need to be trained

```python
for name, param in self.named_parameters():
	if "backbone" in name:
		param.requires_grad = False
```

#### Overfitting the model

- To ensure the model will train correctly, the model is overfit by training on only 11 examples for 500 epochs - The image below shows that the model can be overfit (training loss decreasing, validation loss increasing) and is able to overfit more than the smaller resnet-50 model. This would be expected given it has more parameters that can be overfit
  ![[Pasted image 20221003140243.png|1000]]

### Post-process improvements

- We know that most of the errors we can face fall into one of a few categories: swaps, jitter, misses, inversions. We can do some post-processing to address some of these to improve the final output

### Jitters

- Given that small movements between the predicted location of joints are arbitrary, we can smooth the positioning with a simple weighted average smoothing

- For smoothing Locally Weighted Scatterplot Smooting (LOWESS) is used as it is a non-parametric
  - a non-parametric strategy is required because how the coordinates jitter (up or down) is unknown in advance. Therefore a parametric curve fit would only provide a good fit if the number of large jitters matched the number of parameters
  - LOWESS is simply linear regression applied over a sliding window, sliding over each point
- The image below shows examples of the smoothing on a person's right knee in the x and y coordinates respectively - The green marker shows the predictions from the model and where it sometimes predicts sudden deviations from what is likely - Such values are outliers and are replaced with an interpolation from the previous coordinates - We can define an outlier as a keypoint coordinate that deviates from it's nearest neighbour by more that 1 std (which is show by the orange line and )
  ![[Pasted image 20220927140538.png|1200]]

## Experiments

==implement ideas in this paper Clustering Anchor for Faster R-CNN to Improve Detection Results==

- use adaptiveconcatpooling - implement yourself https://docs.fast.ai/layers.html#adaptiveconcatpool2d

## Testing

### Pre-train tests

- Overfitting
- - Checking if any labels are missing in your training and validation datasets
- Check the single gradient to find the loss of data

## Results

## Unaddressed issues

- Currently it seems the model cannot decide to only place some keypoints and tries to place all keypoints even if only part of a person is visible on the image - it might not be able to set coordinates to 0 on x-y well because that's the lowest possible value so it will always be biased above? Not sure needs investigating
- If we wanted to fine tune the model on specific data type eg people running, we could be more precise with anchor box choice in terms of aspec ratios and size for example because a person running will only be upright (by definition )

### self-supervised learning

- this technique is particuarly interesting because if it is possible to train a model merely on data with no direct annotations where the more data it is given, the better the results, we can save a lot of annotating time

[1]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Ronchi_Benchmarking_and_Error_ICCV_2017_paper.pdf
